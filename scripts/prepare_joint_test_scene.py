#!/usr/bin/env python3
"""
Create an isolated ROS routing layer; preserve source physics and commands.

Run Isaac with ROS_DOMAIN_ID=231 and ROS_LOCALHOST_ONLY=1 as well.
This layer does not activate a controller or provide network authentication.
"""

import argparse
import hashlib
from pathlib import Path


def prepare(scene, output):
    from pxr import Sdf
    from pxr import Usd

    scene = Path(scene).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; preserve existing evidence')
    original = Usd.Stage.Open(str(scene))
    if not original:
        raise ValueError('Cannot open source scene')
    contexts = []
    endpoints = []
    for prim in original.Traverse():
        kind = prim.GetAttribute('node:type').Get()
        if kind == 'isaacsim.ros2.bridge.ROS2Context':
            contexts.append(prim)
        elif kind == 'isaacsim.ros2.bridge.ROS2QoSProfile':
            continue  # Configuration helper, not a communicating endpoint.
        elif kind and kind.startswith('isaacsim.ros2.bridge.ROS2'):
            endpoints.append(prim)
    if not contexts or not endpoints:
        raise ValueError('No ROS contexts or endpoints found')
    context_outputs = {str(p.GetPath()) + '.outputs:context' for p in contexts}
    topics = []
    for prim in endpoints:
        connections = prim.GetAttribute('inputs:context').GetConnections()
        if len(connections) != 1 or str(connections[0]) not in context_outputs:
            raise ValueError(f'Unverified ROS context: {prim.GetPath()}')
        topic = prim.GetAttribute('inputs:topicName').Get()
        if prim.GetAttribute('node:type').Get().endswith('.ROS2PublishClock'):
            topic = '/clock'
        elif not topic or not str(topic).strip('/'):
            raise ValueError(f'Unspecified ROS topic: {prim.GetPath()}')
        else:
            topic = '/joint_test/' + str(topic).strip('/')
        topics.append((prim.GetPath(), topic))
    names = [topic for _, topic in topics]
    for required in ('/joint_test/isaac_joint_states', '/joint_test/isaac_joint_commands'):
        if names.count(required) != 1:
            raise ValueError(f'Expected exactly one {required} endpoint')
    # A connected input would override the authored value at runtime.
    for prim in contexts:
        for name in ('inputs:domain_id', 'inputs:useDomainIDEnvVar'):
            if prim.GetAttribute(name).GetConnections():
                raise ValueError(f'Connected routing input: {prim.GetPath()}.{name}')
    for prim in endpoints:
        for name in ('inputs:topicName', 'inputs:nodeNamespace'):
            if prim.GetAttribute(name).GetConnections():
                raise ValueError(f'Connected routing input: {prim.GetPath()}.{name}')
    layer = Sdf.Layer.CreateNew(str(output))
    for key in original.GetRootLayer().pseudoRoot.ListInfoKeys():
        if key not in ('subLayers', 'subLayerOffsets', 'customLayerData'):
            layer.pseudoRoot.SetInfo(key, original.GetRootLayer().pseudoRoot.GetInfo(key))
    layer.subLayerPaths = [str(scene)]
    layer.customLayerData = dict(
        original.GetRootLayer().customLayerData,
        reference_scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        diagnostic='isolated joint test routing only; requires localhost-only process',
    )
    stage = Usd.Stage.Open(layer)
    for prim in contexts:
        target = stage.OverridePrim(prim.GetPath())
        # Match the installed ROS2Context schema (an unsigned byte, not uint32).
        target.CreateAttribute('inputs:domain_id', Sdf.ValueTypeNames.UChar).Set(231)
        target.CreateAttribute('inputs:useDomainIDEnvVar', Sdf.ValueTypeNames.Bool).Set(False)
    for path, topic in topics:
        target = stage.OverridePrim(path)
        target.CreateAttribute('inputs:topicName', Sdf.ValueTypeNames.String).Set(topic)
        target.CreateAttribute('inputs:nodeNamespace', Sdf.ValueTypeNames.String).Set('')
    layer.Save()
    return str(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(prepare(args.scene, args.output))
