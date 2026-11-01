"""Run with Isaac's USD Python environment; no simulator or ROS process starts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from pxr import Sdf, Usd
except ImportError:
    Usd = None

spec = importlib.util.spec_from_file_location('routing',
    Path(__file__).parents[1] / 'scripts/prepare_joint_test_scene.py')
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


@unittest.skipIf(Usd is None, 'Requires Isaac USD Python environment')
class RoutingTest(unittest.TestCase):
    def fixture(self, folder):
        source = Path(folder) / 'source.usda'
        stage = Usd.Stage.CreateNew(str(source))
        stage.SetDefaultPrim(stage.DefinePrim('/World'))
        stage.SetMetadata('metersPerUnit', 1.)
        stage.SetMetadata('upAxis', 'Z')
        stage.SetTimeCodesPerSecond(60.)
        context = stage.DefinePrim('/World/context')
        context.CreateAttribute('node:type', Sdf.ValueTypeNames.Token).Set(
            'isaacsim.ros2.bridge.ROS2Context')
        context.CreateAttribute('inputs:domain_id', Sdf.ValueTypeNames.UChar).Set(10)
        stage.DefinePrim('/World/qos').CreateAttribute('node:type', Sdf.ValueTypeNames.Token).Set(
            'isaacsim.ros2.bridge.ROS2QoSProfile')
        for name, kind in [('isaac_joint_states', 'ROS2PublishJointState'),
                           ('isaac_joint_commands', 'ROS2SubscribeJointState')]:
            prim = stage.DefinePrim('/World/' + name)
            prim.CreateAttribute('node:type', Sdf.ValueTypeNames.Token).Set('isaacsim.ros2.bridge.' + kind)
            prim.CreateAttribute('inputs:context', Sdf.ValueTypeNames.UInt64).SetConnections(
                ['/World/context.outputs:context'])
            prim.CreateAttribute('inputs:topicName', Sdf.ValueTypeNames.String).Set(name)
        stage.DefinePrim('/World/physics').CreateAttribute(
            'test:unchanged', Sdf.ValueTypeNames.String).Set('delay and physics')
        stage.GetRootLayer().Save()
        return source, stage

    def test_routing_preserves_source_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            source, original = self.fixture(folder)
            before = source.read_bytes()
            output = Path(folder) / 'isolated.usda'
            routing.prepare(source, output)
            stage = Usd.Stage.Open(str(output))
            self.assertEqual(source.read_bytes(), before)
            for key in ('metersPerUnit', 'upAxis', 'timeCodesPerSecond', 'defaultPrim'):
                self.assertEqual(stage.GetMetadata(key), original.GetMetadata(key))
            self.assertEqual(stage.GetAttributeAtPath('/World/context.inputs:domain_id').Get(), 231)
            self.assertEqual(stage.GetAttributeAtPath('/World/context.inputs:domain_id').GetTypeName(),
                             Sdf.ValueTypeNames.UChar)
            self.assertFalse(stage.GetAttributeAtPath('/World/context.inputs:useDomainIDEnvVar').Get())
            self.assertEqual(stage.GetAttributeAtPath('/World/isaac_joint_commands.inputs:topicName').Get(),
                             '/joint_test/isaac_joint_commands')
            self.assertEqual(stage.GetAttributeAtPath('/World/physics.test:unchanged').Get(), 'delay and physics')
            with self.assertRaises(ValueError):
                routing.prepare(source, output)

    def test_unverified_context_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, stage = self.fixture(folder)
            stage.GetAttributeAtPath('/World/isaac_joint_commands.inputs:context').ClearConnections()
            stage.GetRootLayer().Save()
            output = Path(folder) / 'isolated.usda'
            with self.assertRaises(ValueError):
                routing.prepare(source, output)
            self.assertFalse(output.exists())

    def test_connected_routing_inputs_rejected_before_output(self):
        # Graph connections take precedence over the fixed isolation values.
        cases = [('/World/context', 'inputs:domain_id', Sdf.ValueTypeNames.UChar),
                 ('/World/context', 'inputs:useDomainIDEnvVar', Sdf.ValueTypeNames.Bool),
                 ('/World/isaac_joint_commands', 'inputs:topicName', Sdf.ValueTypeNames.String),
                 ('/World/isaac_joint_commands', 'inputs:nodeNamespace', Sdf.ValueTypeNames.String)]
        for path, name, value_type in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                source, stage = self.fixture(folder)
                stage.GetPrimAtPath(path).CreateAttribute(name, value_type).SetConnections(
                    ['/World/other.outputs:value'])
                stage.GetRootLayer().Save()
                output = Path(folder) / 'isolated.usda'
                with self.assertRaisesRegex(ValueError, 'Connected routing input'):
                    routing.prepare(source, output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
