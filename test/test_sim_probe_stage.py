"""Real USD integration test. Also runnable with Isaac's bundled Python."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
except ImportError:
    Usd = None

spec = importlib.util.spec_from_file_location(
    'probe', Path(__file__).parents[1] / 'scripts/prepare_sim_delay_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@unittest.skipIf(Usd is None, 'Requires Isaac USD Python environment')
class StageMetadataTest(unittest.TestCase):
    def test_stage_metadata_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'reference.usda'
            output = Path(temporary) / 'probe.usda'
            stage = Usd.Stage.CreateNew(str(source))
            stage.SetDefaultPrim(stage.DefinePrim('/World'))
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.SetStageKilogramsPerUnit(stage, 2.)
            stage.SetTimeCodesPerSecond(60.)
            stage.SetFramesPerSecond(30.)
            stage.SetStartTimeCode(5.)
            stage.SetEndTimeCode(300.)
            stage.GetRootLayer().customLayerData = {'fixture': 'preserve'}
            script = ('import collections\ndef setup(db):\n    pass\n'
                      'def compute(db):\n    delay_steps = 6\n'
                      '    if True:\n        print(len(db.internal_state.buffer))\n')
            stage.DefinePrim(probe.NODE).CreateAttribute(
                'inputs:script', Sdf.ValueTypeNames.String).Set(script)
            clock = Sdf.Path(probe.SIM_TIME)
            stage.DefinePrim(clock.GetPrimPath()).CreateAttribute(
                clock.name, Sdf.ValueTypeNames.Double).Set(0.)
            stage.GetRootLayer().Save()
            before = source.read_bytes()
            probe.prepare(source, output)
            diagnostic = Usd.Stage.Open(str(output))
            for key in ('metersPerUnit', 'kilogramsPerUnit', 'upAxis',
                        'timeCodesPerSecond', 'framesPerSecond', 'startTimeCode',
                        'endTimeCode', 'defaultPrim'):
                self.assertEqual(diagnostic.GetMetadata(key), stage.GetMetadata(key), key)
            self.assertEqual(diagnostic.GetDefaultPrim().GetPath(), '/World')
            self.assertEqual(diagnostic.GetRootLayer().customLayerData['fixture'], 'preserve')
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(ValueError):
                probe.prepare(source, output)


if __name__ == '__main__':
    unittest.main()
