# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that cuju.agents.protocol and client.py speak the same frames."""

import io
import socket
import threading

import numpy as np

from cuju import agents
from cuju.agents import client, protocol


def value():
    return {
        'data': {
            'a': np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            'b': np.ones((2, 2), dtype=bool),
        },
        'first': np.array([True, False]),
        'meta': {'overage': 60},
        'list': [1, 2.5, 'x', None],
    }


def test_both_codecs_write_the_same_bytes_and_read_each_other():
    ours = protocol.encode('act', value(), field=3)
    theirs = client.encode('act', value(), field=3)
    assert ours == theirs
    for reader in (protocol.read, client.read):
        found = reader(io.BytesIO(ours))
        assert found['kind'] == 'act' and found['field'] == 3
        assert (found['value']['data']['a'] == value()['data']['a']).all()
        assert found['value']['list'] == [1, 2.5, 'x', None]


def test_the_message_constructors_decode():
    assert protocol.read(io.BytesIO(protocol.hello({'entry': 'random'})))['value'] == {
        'inner': {'entry': 'random'}
    }
    spec = agents.Spec(environments=2)
    assert protocol.read(io.BytesIO(protocol.start(spec)))['value']['environments'] == 2
    assert protocol.read(io.BytesIO(protocol.close()))['kind'] == 'close'
    answer = protocol.read(io.BytesIO(protocol.actions(np.zeros((2, 1), dtype=np.int64))))
    assert answer['value']['actions'].shape == (2, 1)
    failure = protocol.read(io.BytesIO(protocol.error('crash', 'boom', 'trace')))
    assert failure['reason'] == 'crash' and failure['traceback'] == 'trace'


def test_a_connection_serves_a_child_over_a_socket():
    near, far = socket.socketpair()
    child = threading.Thread(
        target=lambda: client.serve(far.makefile('rb'), far.makefile('wb', buffering=0)),
        daemon=True,
    )
    child.start()
    connection = protocol.Connection(near.makefile('rwb', buffering=0))
    connection.send(protocol.hello({'entry': 'cuju.agents.standard:Random'}))
    assert connection.receive()['value']['representation'] == 'raw'
    connection.send(protocol.start(agents.Spec(environments=2, agents=1, seed=1)))
    assert connection.receive()['kind'] == 'ready'
    observation = agents.Observation(
        data={},
        first=np.ones(2, bool),
        active=np.ones((2, 1), bool),
        left_players=np.zeros(2),
        right_players=np.zeros(2),
        previous=None,
        step=np.zeros(2),
        side=np.zeros(2),
    )
    connection.send(protocol.act(observation))
    assert connection.receive()['value']['actions'].shape == (2, 1)
    connection.send(protocol.close())
    child.join(5)
    near.close()
    far.close()
