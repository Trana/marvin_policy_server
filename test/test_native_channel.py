"""Exercise transport failure and ACK handling without a ROS graph."""

import json
import socket
from threading import Event
from threading import Thread

from marvin_policy_server.native_channel import NativeChannel
from marvin_policy_server.native_protocol import Kind
from marvin_policy_server.native_protocol import Packet
from marvin_policy_server.native_protocol import receive
from marvin_policy_server.native_protocol import send
import pytest


@pytest.fixture
def peers():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    wake = Event()
    channel = NativeChannel(parent, wake.set, queue_limit=2)
    try:
        yield channel, child, wake
    finally:
        assert channel.close()
        child.close()


def test_packets_are_delivered_only_when_executor_drains(peers):
    channel, child, wake = peers
    packet = Packet(Kind.SNAPSHOT, sequence=1, values=(2.0,))
    send(child, packet)
    assert wake.wait(1.0)
    assert channel.drain() == (packet,)
    assert channel.drain() == ()


def test_ack_does_not_require_executor_dispatch(peers):
    channel, child, _ = peers
    observed = []

    def peer():
        operation = receive(child)
        observed.append(operation)
        send(child, Packet(Kind.ACK, sequence=operation.sequence + 1, generation=123))
        send(child, Packet(Kind.ACK, sequence=operation.sequence, generation=7))

    thread = Thread(target=peer)
    thread.start()
    ack = channel.control(Packet(Kind.DEACTIVATE), timeout=1.0)
    thread.join(1.0)
    assert not thread.is_alive()
    assert observed[0].kind == Kind.DEACTIVATE
    assert ack.generation == 7
    assert channel.drain() == ()


def test_missing_ack_fails_closed(peers):
    channel, child, _ = peers
    with pytest.raises(RuntimeError, match='acknowledgement expired'):
        channel.control(Packet(Kind.DEACTIVATE), timeout=0.02)
    assert receive(child).kind == Kind.DEACTIVATE
    with pytest.raises(EOFError):
        receive(child)
    with pytest.raises(RuntimeError, match='transport failed'):
        channel.send(Packet(Kind.RESULT))


def test_peer_death_wakes_executor_and_blocks_results(peers):
    channel, child, wake = peers
    child.close()
    assert wake.wait(1.0)
    with pytest.raises(RuntimeError, match='closed the connection'):
        channel.drain()
    with pytest.raises(RuntimeError):
        channel.send(Packet(Kind.RESULT))


def test_queue_overflow_does_not_silently_discard_old_work(peers):
    channel, child, _ = peers
    for sequence in range(3):
        send(child, Packet(Kind.SNAPSHOT, sequence=sequence))
    child.settimeout(1.0)
    with pytest.raises(EOFError):
        receive(child)
    with pytest.raises(RuntimeError, match='bounded capacity') as error:
        channel.drain()
    detail = json.loads(str(error.value).split('bounded capacity: ', 1)[1])
    assert detail['limit'] == 2
    assert [item['sequence'] for item in detail['queued']] == [0, 1]
    assert detail['incoming']['kind'] == 'SNAPSHOT'
    assert detail['incoming']['sequence'] == 2
    assert detail['incoming']['source_age_ms'] is None


def test_status_burst_keeps_latest_without_executor_dispatch(peers):
    channel, child, _ = peers
    for sequence in range(20):
        send(child, Packet(Kind.STATUS, sequence=sequence, generation=8))
    flush_reader(channel, child)
    assert channel.drain() == (Packet(Kind.STATUS, sequence=19, generation=8),)


def flush_reader(channel, child):
    """Fence previous packets with an ACK without executor dispatch."""

    def peer():
        request = receive(child)
        send(child, Packet(Kind.ACK, sequence=request.sequence))

    thread = Thread(target=peer)
    thread.start()
    channel.control(Packet(Kind.DEACTIVATE), timeout=1.0)
    thread.join(1.0)
    assert not thread.is_alive()


def test_status_does_not_consume_work_capacity_or_reorder_commits(peers):
    channel, child, _ = peers
    commit = Packet(Kind.COMMIT, sequence=1, generation=7)
    snapshot = Packet(Kind.SNAPSHOT, sequence=2, generation=8)
    latest = Packet(Kind.STATUS, generation=8)
    for packet in [Packet(Kind.STATUS, generation=7), commit, snapshot, latest]:
        send(child, packet)
    flush_reader(channel, child)
    assert channel.drain() == (commit, snapshot, latest)


def test_malformed_packet_fails_transport(peers):
    channel, child, wake = peers
    child.send(b'bad')
    assert wake.wait(1.0)
    with pytest.raises(RuntimeError, match='Invalid packet size'):
        channel.drain()


def test_close_wakes_pending_control(peers):
    channel, child, _ = peers
    errors = []

    def operation():
        try:
            channel.control(Packet(Kind.DEACTIVATE), timeout=1.0)
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = Thread(target=operation)
    thread.start()
    assert receive(child).kind == Kind.DEACTIVATE
    assert channel.close()
    thread.join(1.0)
    assert not thread.is_alive()
    assert errors == ['Native policy transport is closed']


def test_non_control_packet_rejected(peers):
    channel, _, _ = peers
    with pytest.raises(ValueError, match='not a native control'):
        channel.control(Packet(Kind.RESULT))
