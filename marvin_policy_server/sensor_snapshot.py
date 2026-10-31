"""Immutable sensor values for policy computation without a ROS dependency.

The field names match the numeric builders' existing message interface. Transport
code validates freshness and ownership separately. A snapshot is not permission
to publish an action.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Stamp:
    sec: int
    nanosec: int

    @classmethod
    def from_ns(cls, value: int):
        if type(value) is not int or value < 0:
            raise ValueError('Sensor stamp must be nonnegative integer nanoseconds')
        return cls(*divmod(value, 1_000_000_000))


@dataclass(frozen=True)
class Header:
    stamp: Stamp


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class JointSample:
    header: Header
    name: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...]


@dataclass(frozen=True)
class ImuSample:
    header: Header
    orientation: Quaternion
    angular_velocity: Vector3
    linear_acceleration: Vector3


@dataclass(frozen=True)
class SensorSnapshot:
    joint: JointSample
    imu: ImuSample

    @classmethod
    def from_values(cls, names, joint_stamp_ns, imu_stamp_ns, values):
        """Copy N positions, N velocities, XYZW orientation, gyro and acceleration."""
        names = tuple(names)
        if not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('Joint names must be nonempty strings')
        if len(set(names)) != len(names):
            raise ValueError('Joint names must be unique')
        data = tuple(float(value) for value in values)
        count = len(names)
        if len(data) != 2 * count + 10 or not all(math.isfinite(v) for v in data):
            raise ValueError('Sensor values have an invalid size or non-finite value')
        offset = 2 * count
        return cls(
            JointSample(Header(Stamp.from_ns(joint_stamp_ns)), names,
                        data[:count], data[count:offset]),
            ImuSample(Header(Stamp.from_ns(imu_stamp_ns)),
                      Quaternion(*data[offset:offset + 4]),
                      Vector3(*data[offset + 4:offset + 7]),
                      Vector3(*data[offset + 7:offset + 10])),
        )

    @classmethod
    def from_messages(cls, names, joint, imu):
        """Normalize message order before crossing the computation boundary."""
        names = tuple(names)
        if len(set(joint.name)) != len(joint.name):
            raise ValueError('Joint message contains duplicate names')
        indices = {name: i for i, name in enumerate(joint.name)}
        try:
            positions = [joint.position[indices[name]] for name in names]
            velocities = [joint.velocity[indices[name]] for name in names]
        except (KeyError, IndexError) as exc:
            raise ValueError('Joint message is missing required positions or velocities') from exc
        q, w, a = imu.orientation, imu.angular_velocity, imu.linear_acceleration
        stamp = lambda m: int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
        return cls.from_values(names, stamp(joint), stamp(imu),
                               positions + velocities +
                               [q.x, q.y, q.z, q.w, w.x, w.y, w.z, a.x, a.y, a.z])
