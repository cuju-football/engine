# The controller protocol

This page specifies version 1 of the messages that `cuju.agents` exchanges with a controller that runs in another process. The reference implementation of the child is the file `cuju/agents/client.py`, and `cuju agents check --command` checks any other implementation against this page. A child can be written in any language that reads and writes bytes on two pipes.

## Transport

The host starts the child and talks to it on its standard input and standard output. The child keeps its first descriptor, the one that its standard output names, for answers, and points that descriptor at its standard error before it loads any user code. Whatever user code prints then goes to standard error and can never forge an answer. The child ends when its standard input ends.

Every message is a frame.

| Bytes | Content |
| --- | --- |
| 4 | The length of the header, an unsigned little-endian integer. |
| 4 | The length of the body, an unsigned little-endian integer. |
| header length | The header, a JSON object in UTF-8. |
| body length | The body, the raw bytes of every array, one after the other. |

The header may be at most 16 MiB and the body at most 1 GiB. A reader refuses a larger frame before it reads it. Nothing is ever pickled.

The header holds these keys.

| Key | Meaning |
| --- | --- |
| `protocol` | The number 1. A frame with another number is refused. |
| `kind` | The message, one of the kinds below. |
| `value` | The content of the message as JSON, with each array replaced by `{"$array": index}`. |
| `arrays` | One entry for each array, `[dtype, shape]`, in the order of the body. |
| other keys | Fields of the message, named below. |

The dtypes are `bool`, `uint8`, `int8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`, `float16`, `float32` and `float64`, written as numpy writes them in little-endian order, such as `<f4`. An array is stored in C order, and its body bytes are little-endian. A key of an object is text, and the key `$array` is reserved.

## Starting the client

A host that runs Python starts `python -X utf8 -c STUB`, where `STUB` is the text of `cuju.agents.client.STUB`. The stub reads eight bytes, the length of the source as an unsigned little-endian integer, then that many bytes of the source of `client.py`, and runs it. The frames follow on the same input. A command that holds the text `{stub}` in its `command` list gets the stub there. A command without it runs as it is, so a child written in another language reads frames from its first byte. `cuju agents client` prints the source.

## Messages

The host sends `hello`, `start`, `act` and `close`. The child answers with `ready`, `actions` and `error`.

**`hello`.** The host sends it first. The value is `{"inner": NODE}`, the unbuilt tree that the child builds, where a node is an object with the key `entry`, the options of the entry, an optional `inner` node, and the optional objects `control` and `options`. The child answers `ready`, with the value `{"representation": TEXT, "control": OBJECT or null, "options": OBJECT or null}` and the fields `python` and `numpy` with their versions. A child that cannot build the tree answers `error` with the reason `load`.

**`start`.** The value is the spec of the session, an object with the keys `scenario`, `length`, `environments`, `agents`, `action_set`, `representation`, `players`, `assignment`, `selection`, `goalkeeper`, `roles`, `seed`, `side`, `mirrored` and `options`. The child answers `ready` with a null value. A host sends `start` once for each setup of a batch.

**`act`.** The value is an object with the keys `data`, `first`, `active`, `left_players`, `right_players`, `previous`, `step`, `side`, `meta` and `environments`. The key `data` holds the arrays of the representation, either one array or an object of arrays, each shaped `[environments, agents, ...]`. The other arrays have one entry for each environment, except `active` and `previous`, which are `[environments, agents]`. The key `meta` is an object of plain values, and `environments` is null or an array of the indices of the environments that the arrays hold. The child answers `actions`.

**`actions`.** The value is `{"actions": ARRAY, "extras": OBJECT}`. The actions are an integer array of the shape `[environments, agents]`, and the extras are numeric arrays that the host never interprets.

**`close`.** The child closes its controller and exits with code 0. It sends nothing.

**`error`.** The child sends it in place of an answer, and exits with code 1. The fields are `reason`, one of `load`, `crash`, `protocol` and `contract`, `message`, a line of text, and `traceback`, the text of the traceback. The child also writes the traceback to its standard error.

## Time, ending and trees

The host waits up to its timeout, 300 seconds by default, for `ready` and for each `actions`. A missing answer, a closed pipe, a frame that does not read and a version that differs are breaches of the contract, and the host raises `ControllerError` with the reason `timeout`, `closed`, `protocol` or `shape`.

When the host closes a child it sends `close`, closes the input, waits a few seconds, and then ends the whole tree. On Linux and macOS the child starts in a session of its own, and the host kills the session. On Windows the child starts suspended, joins a job object that ends every process in it when the job closes, and then runs, so the tree ends even if the host crashes. A process that leaves its session or its job is not tracked. `process` isolates crashes and environments. It is not a security boundary, and untrusted code belongs in a container, as [Agents](agents.md) shows.

## Conformance

`cuju agents check --command ARG...` starts the command, sends `hello`, `start` and a few `act` messages with a synthetic observation, and checks every answer against this page. It exits with status 0 when the child conforms.

## Writing a transport

`cuju.agents.protocol` holds the frame encoder and decoder, the constructors of the messages and a `Connection` over any readable and writable binary stream, a pipe or the file of a socket. It is the code that `process` uses on the host side. The file `client.py` keeps its own copy, so a child needs no cuju, and a test checks that the two write the same bytes and read each other's frames.

A transport is then a user-side controller wrapper. This one reaches a child that listens on a socket, where the child runs `cuju.agents.client.serve` on the file of an accepted connection.

```python
import socket

from cuju.agents import Actions, Controller, protocol


class Socket(Controller):
    asynchronous = False

    def __init__(self, address, inner):
        self.connection = protocol.Connection(socket.create_connection(address).makefile('rwb'))
        self.connection.send(protocol.hello(inner))
        ready = self.connection.receive()['value']
        self.representation = ready['representation']

    def start(self, spec):
        self.connection.send(protocol.start(spec))
        self.connection.receive()

    def act(self, observation):
        self.connection.send(protocol.act(observation))
        value = self.connection.receive()['value']
        return Actions(value['actions'], value['extras'])

    def close(self):
        self.connection.send(protocol.close())
```
