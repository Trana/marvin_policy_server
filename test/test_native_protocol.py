"""Native and Python transport implementations must agree byte for byte."""

from dataclasses import replace
from pathlib import Path
import shutil
import socket
import struct
import subprocess

from marvin_policy_server.native_protocol import HEADER
from marvin_policy_server.native_protocol import Kind
from marvin_policy_server.native_protocol import MAX_PACKET
from marvin_policy_server.native_protocol import MAX_TOKEN_BYTES
from marvin_policy_server.native_protocol import MAX_VALUES
from marvin_policy_server.native_protocol import Packet
from marvin_policy_server.native_protocol import receive
from marvin_policy_server.native_protocol import send
from marvin_policy_server.native_protocol import validate_socket
import pytest


def example():
    return Packet(Kind.SNAPSHOT, 1, 2, 3, 4, 5, 6, 7, 8, (0.0, -1.5, 9.81))


@pytest.mark.parametrize('kind', list(Kind))
def test_roundtrip(kind):
    packet = replace(example(), kind=kind)
    assert Packet.decode(packet.encode()) == packet


def test_all_truncated_lengths_and_extra_bytes_are_rejected():
    data = example().encode()
    for length in range(len(data)):
        with pytest.raises(ValueError):
            Packet.decode(data[:length])
    with pytest.raises(ValueError):
        Packet.decode(data + b'\0')


@pytest.mark.parametrize('offset', [0, 4, 8, 12])
def test_header_corruption_is_rejected(offset):
    data = bytearray(example().encode())
    data[offset] = 255
    with pytest.raises(ValueError):
        Packet.decode(data)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_wire_and_local_values_are_rejected(value):
    data = bytearray(example().encode())
    struct.pack_into('<d', data, HEADER.size, value)
    with pytest.raises(ValueError):
        Packet.decode(data)
    with pytest.raises(ValueError):
        replace(example(), values=(value,)).encode()


@pytest.mark.parametrize('value', [-1, 2**64, 0.1, True])
def test_invalid_integer(value):
    with pytest.raises(ValueError):
        replace(example(), generation=value).encode()


def test_payload_bounds():
    largest = replace(example(), values=(1.0,) * MAX_VALUES)
    assert len(largest.encode()) == MAX_PACKET - MAX_TOKEN_BYTES
    assert Packet.decode(largest.encode()) == largest
    with pytest.raises(ValueError):
        replace(largest, values=largest.values + (1.0,)).encode()


def test_local_socket_roundtrip_truncation_and_peer_exit():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        validate_socket(parent)
        send(parent, example())
        assert receive(child) == example()
        parent.send(example().encode() + b'\0' * MAX_PACKET)
        with pytest.raises(ValueError, match='truncated'):
            receive(child)
        parent.close()
        with pytest.raises(EOFError):
            receive(child)
    finally:
        parent.close()
        child.close()


def test_stream_socket_is_rejected():
    parent, child = socket.socketpair()
    try:
        with pytest.raises(ValueError, match='sequence-packet'):
            validate_socket(parent)
    finally:
        parent.close()
        child.close()


def test_cpp_golden_packet_and_native_checks(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = Path(__file__).parents[1] / 'native_io' / 'protocol_test.cpp'
    binary = tmp_path / 'protocol_test'
    subprocess.run(
        [
            compiler,
            '-std=c++17',
            '-Wall',
            '-Wextra',
            '-Wpedantic',
            '-Werror',
            '-O2',
            str(source),
            '-o',
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)
    result = subprocess.run([str(binary), '--golden'], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == example().encode().hex()


def test_preparation_token_is_opaque_and_bounded():
    packet = Packet(Kind.PREPARE, token='å' * 64, values=(0.2,) * 12)
    assert Packet.decode(packet.encode()) == packet
    for changed in (replace(packet, token='a' * 257), replace(packet, kind=Kind.RESULT)):
        with pytest.raises(ValueError):
            changed.encode()
    data = bytearray(packet.encode())
    data[4] = 1  # Old protocol binaries must not silently accept the new contract.
    with pytest.raises(ValueError):
        Packet.decode(data)
