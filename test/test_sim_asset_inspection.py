"""Run with the existing USD Python environment; no simulator required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from pxr import Usd, UsdPhysics
except ImportError:
    Usd = UsdPhysics = None

spec = importlib.util.spec_from_file_location(
    'asset_inspection', Path(__file__).parents[1] / 'scripts/inspect_sim_asset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipIf(Usd is None, 'Requires the existing USD Python environment')
class AssetInspectionTests(unittest.TestCase):
    def test_mass_units_missing_mass_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'asset.usda'
            stage = Usd.Stage.CreateNew(str(path))
            UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
            prim = stage.DefinePrim('/body', 'Xform')
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(2000)
            other = stage.DefinePrim('/inferred', 'Xform')
            UsdPhysics.RigidBodyAPI.Apply(other)
            stage.GetRootLayer().Save()
            first = module.inspect(path)
            self.assertEqual(first['explicit_mass_sum_kg'], 2)
            self.assertEqual(first['bodies_without_explicit_mass'], ['/inferred'])
            self.assertEqual(len(first['layers']), 1)
            self.assertEqual(first, module.inspect(path))
            UsdPhysics.MassAPI(prim).GetMassAttr().Set(3000)
            stage.GetRootLayer().Save()
            second = module.inspect(path)
            self.assertEqual(second['explicit_mass_sum_kg'], 3)
            self.assertNotEqual(first['layers'][0]['sha256'], second['layers'][0]['sha256'])

    def test_missing_asset_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                module.inspect(Path(directory) / 'missing.usda')


if __name__ == '__main__':
    unittest.main()
