import ast
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

from marvin_policy_server.artifact_identity import artifact_identity
from marvin_policy_server.env_config_loader import EnvConfigLoader


def test_identity_describes_loaded_bytes_not_current_file(tmp_path):
    path = tmp_path / 'policy.pt'
    content = b'loaded model bytes'
    path.write_bytes(b'replaced on disk')
    receipt = artifact_identity(path, content)
    assert receipt == {
        'path': str(path),
        'sha256': hashlib.sha256(content).hexdigest(),
        'size': len(content),
    }


def test_environment_receipt_remains_pinned_to_parsed_bytes(tmp_path):
    path = tmp_path / 'env.yaml'
    content = b'scene: {robot: {init_state: {joint_pos: {joint: 0.0}}}}\n'
    path.write_bytes(content)
    loader = EnvConfigLoader(path)
    assert loader.loaded_artifact is None
    assert loader.get_joint_names() == ['joint']
    path.write_text('not the loaded environment')
    assert loader.get_joint_names() == ['joint']
    assert loader.loaded_artifact == artifact_identity(path, content)


def test_policy_receipt_hashes_the_buffer_given_to_torch(tmp_path):
    path = tmp_path / 'policy.pt'
    content = b'model bytes'
    path.write_bytes(content)
    source = Path(__file__).parents[1] / 'marvin_policy_server/marvin_policy_server.py'
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'load_policy'
    )

    def load(buffer):
        assert buffer.read() == content
        path.write_bytes(b'replaced after load')
        return object()

    scope = {
        'io': io,
        'torch': SimpleNamespace(jit=SimpleNamespace(load=load)),
        'artifact_identity': artifact_identity,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    server = SimpleNamespace(policy_path=str(path), _loaded_artifacts={})
    scope['load_policy'](server)
    assert server._loaded_artifacts['policy.pt'] == artifact_identity(path, content)
