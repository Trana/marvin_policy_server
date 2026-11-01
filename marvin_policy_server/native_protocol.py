"""Versioned local transport envelope. No ROS, model or network dependency.

Use only an inherited AF_UNIX SOCK_SEQPACKET socket. Each send is one complete
packet. Lifecycle code must separately validate ownership, sequence and expiry.
The header and floating-point payload use little-endian wire order.
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import math
import socket
import struct


MAGIC = 0x4F49504D  # MPIO
VERSION = 2
MAX_VALUES = 96
MAX_TOKEN_BYTES = 256
HEADER = struct.Struct('<5I8Q')
MAX_PACKET = HEADER.size + MAX_VALUES * 8 + MAX_TOKEN_BYTES


class Kind(IntEnum):
    CONFIGURE = 1
    SNAPSHOT = 2
    RESULT = 3
    ACTIVATE = 4
    DEACTIVATE = 5
    INVALIDATE = 6
    ACK = 7
    FAULT = 8
    STATUS = 9
    SHUTDOWN = 10
    COMMIT = 11
    PREPARE = 12


class Flags(IntFlag):
    PERMITTED = 1
    ACTIVE = 2
    SENSORS_VALID = 4
    PAUSED = 8
    FAULTED = 16
    OUTPUT_VALID = 32
    ENCODER_TIMING = 64
    SUCCESS = 128
    OUTPUT_LOCKED = 256
    DIRECT_OUTPUT = 512
    OUTPUT_RELEASED = 1024


@dataclass(frozen=True)
class Packet:
    kind: Kind
    sequence: int = 0
    generation: int = 0
    requested_ns: int = 0
    ros_ns: int = 0
    joint_ns: int = 0
    imu_ns: int = 0
    flags: int = 0
    trace_id: int = 0
    values: tuple[float, ...] = ()
    token: str = ''

    def encode(self):
        kind = Kind(self.kind)
        if type(self.flags) is not int and not isinstance(self.flags, Flags):
            raise ValueError('Packet flags require an integer bit mask')
        integers = (self.sequence, self.generation, self.requested_ns, self.ros_ns,
                    self.joint_ns, self.imu_ns, int(self.flags), self.trace_id)
        if any(type(v) is not int or not 0 <= v < 2**64 for v in integers):
            raise ValueError('Packet header requires unsigned 64-bit integers')
        values = tuple(float(v) for v in self.values)
        if len(values) > MAX_VALUES or not all(math.isfinite(v) for v in values):
            raise ValueError('Packet payload is oversized or non-finite')
        token = self.token.encode('utf-8')
        if len(token) > MAX_TOKEN_BYTES or (token and kind != Kind.PREPARE):
            raise ValueError('Invalid preparation token')
        return HEADER.pack(MAGIC, VERSION, int(kind), len(values), len(token), *integers) + \
            struct.pack(f'<{len(values)}d', *values) + token

    @classmethod
    def decode(cls, data):
        if not HEADER.size <= len(data) <= MAX_PACKET:
            raise ValueError('Invalid packet size')
        magic, version, kind, count, token_size, *integers = HEADER.unpack_from(data)
        if magic != MAGIC or version != VERSION:
            raise ValueError('Unsupported native policy protocol')
        if count > MAX_VALUES or token_size > MAX_TOKEN_BYTES or len(data) != HEADER.size + count * 8 + token_size:
            raise ValueError('Packet payload length mismatch')
        values = struct.unpack_from(f'<{count}d', data, HEADER.size)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Packet payload is non-finite')
        token = data[HEADER.size + count * 8:].decode('utf-8')
        if token and Kind(kind) != Kind.PREPARE:
            raise ValueError('Unexpected preparation token')
        return cls(Kind(kind), *integers, values=values, token=token)


def validate_socket(connection):
    if connection.family != socket.AF_UNIX or \
            connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET:
        raise ValueError('Native policy transport requires a local sequence-packet socket')


def receive(connection):
    """Receive one packet. Reject truncation; do not accept a valid prefix."""
    data, _, flags, _ = connection.recvmsg(MAX_PACKET)
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        raise ValueError('Native policy packet was truncated')
    if not data:
        raise EOFError('Native policy peer closed the connection')
    return Packet.decode(data)


def send(connection, packet):
    """Never block a control callback on a stalled peer."""
    data = packet.encode()
    count = connection.send(data, socket.MSG_DONTWAIT | socket.MSG_NOSIGNAL)
    if count != len(data):
        raise OSError('Incomplete native policy packet send')
