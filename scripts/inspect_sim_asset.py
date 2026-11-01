#!/usr/bin/env python3
"""Report current composed USD masses and layer hashes without running physics."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(path):
    from pxr import Usd, UsdGeom, UsdPhysics

    path = Path(path).resolve(strict=True)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f'Cannot open USD stage: {path}')
    layers = []
    for layer in sorted(stage.GetUsedLayers(), key=lambda item: item.identifier):
        if layer.anonymous:
            # The empty session layer is not a source file.
            if not layer.empty:
                raise ValueError('Unexpected nonempty anonymous layer')
            continue
        source = Path(layer.realPath)
        if not source.is_file():
            raise ValueError(f'Layer is not a local file: {layer.identifier}')
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        layers.append(dict(path=str(source), sha256=digest.hexdigest(),
                           bytes=source.stat().st_size))
    kg_per_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    bodies = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.MassAPI(prim)
        attr = api.GetMassAttr()
        value = attr.Get()
        explicit = attr.HasAuthoredValueOpinion() and value is not None and value > 0
        bodies.append(dict(path=str(prim.GetPath()),
                           enabled=UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get(),
                           explicit_mass_kg=float(value * kg_per_unit) if explicit else None))
    missing = [body['path'] for body in bodies if body['explicit_mass_kg'] is None]
    return dict(schema='sim-asset-inspection.v1', root=str(path), layers=layers,
                meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                kilograms_per_unit=kg_per_unit, bodies=bodies,
                explicit_mass_sum_kg=sum(body['explicit_mass_kg'] or 0 for body in bodies),
                bodies_without_explicit_mass=missing,
                limits='Current default-variant USD composition, not historical training evidence. '
                       'Mass sum includes disabled rigid bodies and excludes inferred density mass. '
                       'No PhysX runtime, randomization or importer overrides are evaluated. '
                       'Layer hashes do not include external mesh/texture files or snapshot their bytes. '
                       'Review USD warnings for unresolved composition dependencies. '
                       'Keep source files unchanged during inspection.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.asset), indent=2, allow_nan=False))
