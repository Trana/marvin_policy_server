"""Stateless CPU inference engines beneath the existing stateful policy runners.

TorchScript remains the immutable source format. ONNX graphs are derived and
validated in memory before readiness. No bundle writes or automatic fallback.
"""

import hashlib
import io
from types import SimpleNamespace

import numpy as np

import torch

DEFAULT_ENGINE = 'onnxruntime'
ENGINES = ('onnxruntime', 'torchscript')


def validate_engine(engine):
    """Reject unknown engines before loading a source artifact."""
    if engine not in ENGINES:
        raise ValueError(f'inference_engine must be one of {ENGINES}, got {engine!r}')
    return engine


def engine_identity(model):
    """Keep derived graph identities separate from source artifact identities."""
    import torch
    return getattr(model, 'inference_engine_info', {
        'engine': 'torchscript', 'version': torch.__version__, 'provider': 'cpu',
    })


class _Graph:
    def __init__(self, source, examples, output_shapes):
        import torch
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                'ONNX engine dependencies are missing; install requirements-onnx.txt '
                'or explicitly select inference_engine=torchscript'
            ) from exc
        self.input_shapes = [tuple(value.shape) for value in examples]
        self.output_shapes = output_shapes
        self.names = [f'input_{index}' for index in range(len(examples))]
        buffer = io.BytesIO()
        with torch.inference_mode():
            torch.onnx.export(
                source, examples, buffer, opset_version=17, dynamo=False,
                input_names=self.names,
            )
        content = buffer.getvalue()
        onnx.checker.check_model(onnx.load_model_from_string(content))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            content, sess_options=options, providers=['CPUExecutionProvider']
        )
        self.session.disable_fallback()
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('ONNX policy requires CPUExecutionProvider only')
        self.active_inputs = {item.name for item in self.session.get_inputs()}
        if not self.active_inputs <= set(self.names):
            raise ValueError('Unexpected ONNX graph inputs')
        self.inference_engine_info = {
            'engine': 'onnxruntime', 'version': ort.__version__, 'onnx_version': onnx.__version__,
            'provider': 'CPUExecutionProvider', 'threads': 1, 'opset': 17,
            'graph_sha256': hashlib.sha256(content).hexdigest(),
        }
        # Compare the exact exported operation against its unchanged Torch source.
        generator = torch.Generator().manual_seed(59)
        with torch.inference_mode():
            for scale in (0.0, 0.1, 1.0, 3.0):
                inputs = tuple(
                    torch.randn(value.shape, generator=generator) * scale for value in examples
                )
                expected = source(*inputs)
                expected = expected if isinstance(expected, tuple) else (expected,)
                actual = self(*inputs)
                actual = actual if isinstance(actual, tuple) else (actual,)
                if len(actual) != len(expected):
                    raise ValueError('ONNX output count differs from the source')
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
            for _ in range(20):
                self(*examples)

    def __call__(self, *inputs):
        import torch
        if len(inputs) != len(self.input_shapes):
            raise ValueError('ONNX input count mismatch')
        feeds = {}
        for name, value, shape in zip(self.names, inputs, self.input_shapes):
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.float32 or tuple(value.shape) != shape):
                raise ValueError('ONNX input must match its CPU float32 shape')
            values = value.detach().numpy()
            if not np.isfinite(values).all():
                raise ValueError('ONNX input contains non-finite values')
            if name in self.active_inputs:
                feeds[name] = np.ascontiguousarray(values)
        outputs = self.session.run(None, feeds)
        if len(outputs) != len(self.output_shapes):
            raise ValueError('ONNX output count mismatch')
        for values, shape in zip(outputs, self.output_shapes):
            if values.dtype != np.float32 or values.shape != shape or not np.isfinite(values).all():
                raise ValueError('ONNX output has invalid shape, dtype or non-finite values')
        tensors = tuple(torch.from_numpy(value) for value in outputs)
        return tensors[0] if len(tensors) == 1 else tensors


def prepare_forward(source, examples, output_shapes, *, engine=DEFAULT_ENGINE):
    """Adapt one stateless operation, including the external velocity estimator."""
    validate_engine(engine)
    return source if engine == 'torchscript' else _Graph(source, examples, output_shapes)


class _SnapshotExport(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, observation, history):
        return self.source.inference_snapshot(observation, history)


class _EstimatorExport(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, history):
        return self.source.actor_estimates(history)


class _Combined:
    def __init__(self, graph):
        self.graph = graph
        self.inference_engine_info = dict(graph.inference_engine_info)

    def __call__(self, observation, history):
        return self.graph(observation, history)


class _Snapshot(_Combined):
    def inference_snapshot(self, observation, history):
        return self.graph(observation, history)

    def __call__(self, observation, history):
        action, state, _ = self.inference_snapshot(observation, history)
        return action, state


def prepare_policy(source, *, manifest, observation_count, action_count, engine=DEFAULT_ENGINE):
    """Prepare every supported operation before readiness, without runner state."""
    import torch
    validate_engine(engine)
    if engine == 'torchscript':
        return source
    observation = torch.zeros(1, observation_count)
    if manifest is None:
        return prepare_forward(source, (observation,), [(1, action_count)], engine=engine)
    history = torch.zeros(1, manifest.history_length, manifest.raw_feature_count)
    shapes = [(1, action_count), (1, manifest.estimated_state_count)]
    snapshot = hasattr(source, 'inference_snapshot')
    graph = _Graph(torch.jit.script(_SnapshotExport(source).eval()) if snapshot else source,
                   (observation, history), shapes + [(1, 4, 3)] if snapshot else shapes)
    model = _Snapshot(graph) if snapshot else _Combined(graph)
    if hasattr(source, 'estimator') and hasattr(source, 'actor'):
        estimator = _Graph(torch.jit.script(_EstimatorExport(source.estimator).eval()), (history,),
                           [(1, 3), (1, 4), (1, 4), (1, 4, 3)])
        actor = _Graph(source.actor, (torch.zeros(1, manifest.actor_observation_count),),
                       [(1, action_count)])
        model.estimator = SimpleNamespace(actor_estimates=estimator)
        model.actor = actor
        model.inference_engine_info['estimator_graph_sha256'] = (
            estimator.inference_engine_info['graph_sha256']
        )
        model.inference_engine_info['actor_graph_sha256'] = (
            actor.inference_engine_info['graph_sha256']
        )
    return model
