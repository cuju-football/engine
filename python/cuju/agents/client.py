# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The child side of `cuju.agents`: one file, the standard library and numpy, Python 3.8.

A host starts it with `python -c STUB`, where the stub reads this file's source from standard
input, so a Python environment that has numpy and no cuju can run it. It keeps the process's
first descriptor for its answers, sends everything that user code prints to standard error, and
speaks the protocol of `docs/protocol.md` until its standard input ends. The host's classes
`Spec`, `Observation` and `Actions` live here too, so one definition serves both sides.
"""

import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import struct
import sys
import traceback
import types
from typing import Any, Dict, List, Optional

import numpy as np

PROTOCOL = 1
# The stub that starts this file: it reads eight bytes of length and then the source.
STUB = (
    'import sys,struct;s=sys.stdin.buffer;n=struct.unpack("<Q",s.read(8))[0];'
    'exec(compile(s.read(n),"client.py","exec"),{"__name__":"__main__"})'
)
ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
MAXIMUM_HEADER = 16 * 1024 * 1024
MAXIMUM_BODY = 1024 * 1024 * 1024
OVERAGE = 60
KAGGLE_MAIN = 'main.py'

_LENGTHS = struct.Struct('<II')
_DTYPES = (
    'bool',
    'uint8',
    'int8',
    'int16',
    'uint16',
    'int32',
    'uint32',
    'int64',
    'uint64',
    'float16',
    'float32',
    'float64',
)
_CODES = dict((np.dtype(name).newbyteorder('<').str, np.dtype(name)) for name in _DTYPES)
_CODE_OF = dict((dtype, code) for code, dtype in _CODES.items())
_ARRAY = '$array'
_NATIVE = sys.byteorder == 'little'


class FrameError(ValueError):
    """A frame that does not read, or a value that no frame can carry."""


class ConfigurationError(ValueError):
    """A tree of controllers that does not load."""


def _array_code(dtype):
    code = _CODE_OF.get(dtype.newbyteorder('=') if not dtype.isnative else dtype)
    if code is None:
        raise FrameError('an array of %s cannot cross, only %s' % (dtype, ', '.join(_DTYPES)))
    return code


def _tree(value, arrays):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        arrays.append(value)
        return {_ARRAY: len(arrays) - 1}
    if isinstance(value, np.generic):
        return _tree(value.item(), arrays)
    if isinstance(value, dict):
        found = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FrameError('a key of a mapping must be text, found %r' % (key,))
            if key == _ARRAY:
                raise FrameError('the key %r is the frames own' % key)
            found[key] = _tree(item, arrays)
        return found
    if isinstance(value, (list, tuple)):
        return [_tree(item, arrays) for item in value]
    raise FrameError('a value of type %s cannot cross' % type(value).__name__)


def encode(kind, value=None, **fields):
    """Returns one frame: two lengths, a JSON header and the raw bytes of every array."""
    arrays = []
    tree = _tree(value, arrays)
    shapes = []
    parts = []
    for array in arrays:
        code = _array_code(array.dtype)
        if not (_NATIVE and array.dtype.isnative) and array.dtype.byteorder != '|':
            array = array.astype(array.dtype.newbyteorder('<'))
        parts.append(np.ascontiguousarray(array).tobytes())
        shapes.append([code, list(array.shape)])
    header = {'protocol': PROTOCOL, 'kind': kind}
    header.update(fields)
    header['value'] = tree
    header['arrays'] = shapes
    head = json.dumps(header, separators=(',', ':')).encode('utf-8')
    body = b''.join(parts)
    if len(head) > MAXIMUM_HEADER or len(body) > MAXIMUM_BODY:
        raise FrameError('a frame of %d bytes is too large to cross' % (len(head) + len(body)))
    return _LENGTHS.pack(len(head), len(body)) + head + body


def _value(tree, arrays):
    if isinstance(tree, dict):
        if len(tree) == 1 and _ARRAY in tree:
            index = tree[_ARRAY]
            if not isinstance(index, int) or not 0 <= index < len(arrays):
                raise FrameError('no array %r in the frame' % (index,))
            return arrays[index]
        return dict((key, _value(item, arrays)) for key, item in tree.items())
    if isinstance(tree, list):
        return [_value(item, arrays) for item in tree]
    return tree


def decode(head, body):
    """Returns the header of a frame with its value under `value`, the arrays in their places."""
    try:
        header = json.loads(head.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as error:
        raise FrameError('a frame header does not read, %s' % error)
    if not isinstance(header, dict) or not isinstance(header.get('kind'), str):
        raise FrameError('a frame header names no kind')
    if header.get('protocol') != PROTOCOL:
        raise FrameError('a frame of protocol %r, expected %d' % (header.get('protocol'), PROTOCOL))
    arrays = []
    offset = 0
    view = memoryview(body)
    for entry in header.pop('arrays', None) or ():
        try:
            code, shape = entry
            dtype = _CODES[code]
            count = math.prod(shape)
            size = count * dtype.itemsize
        except (KeyError, TypeError, ValueError) as error:
            raise FrameError('an array of the frame does not read, %r' % (error,))
        if count < 0 or offset + size > len(body):
            raise FrameError('a frame holds fewer bytes than its arrays')
        flat = np.frombuffer(view[offset : offset + size], dtype=dtype, count=count)
        arrays.append((flat if _NATIVE else flat.astype(dtype.newbyteorder('='))).reshape(shape))
        offset += size
    if offset != len(body):
        raise FrameError('a frame holds more bytes than its arrays')
    header['value'] = _value(header.get('value'), arrays)
    return header


def _read_exactly(stream, size):
    buffer = bytearray(size)
    view = memoryview(buffer)
    done = 0
    while done < size:
        count = stream.readinto(view[done:])
        if not count:
            raise EOFError('the stream ended inside a frame')
        done += count
    return buffer


def read(stream):
    """Reads one frame from a binary stream and returns its header with its value."""
    lengths = stream.read(_LENGTHS.size)
    if not lengths:
        raise EOFError('the stream ended')
    if len(lengths) < _LENGTHS.size:
        lengths = bytes(lengths) + bytes(_read_exactly(stream, _LENGTHS.size - len(lengths)))
    head_size, body_size = _LENGTHS.unpack(lengths)
    if head_size > MAXIMUM_HEADER or body_size > MAXIMUM_BODY:
        raise FrameError('a frame of %d bytes is too large to read' % (head_size + body_size))
    head = _read_exactly(stream, head_size)
    body = _read_exactly(stream, body_size)
    return decode(bytes(head), body)


def write(stream, kind, value=None, **fields):
    """Writes one frame to a binary stream and flushes it."""
    stream.write(encode(kind, value, **fields))
    stream.flush()


@dataclasses.dataclass
class Spec:
    """What a controller learns when its session starts, as plain data."""

    scenario: Optional[str] = None
    length: Optional[int] = None
    environments: int = 1
    agents: int = 1
    action_set: str = 'default'
    representation: str = 'raw'
    players: int = 1
    assignment: str = 'dynamic'
    selection: List[Any] = dataclasses.field(default_factory=list)
    goalkeeper: str = 'scenario'
    roles: List[int] = dataclasses.field(default_factory=list)
    seed: int = 0
    side: int = 0
    mirrored: bool = True
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self):
        """Returns the spec as a dictionary of plain values."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values):
        """Returns the spec that a dictionary of `to_dict` holds."""
        names = set(field.name for field in dataclasses.fields(cls))
        return cls(**dict((name, value) for name, value in values.items() if name in names))


def _take(value, index):
    if isinstance(value, dict):
        return dict((name, _take(item, index)) for name, item in value.items())
    if isinstance(value, np.ndarray):
        return value[np.asarray(index)]
    if hasattr(value, 'device') and hasattr(value, 'shape'):
        import torch

        return value[torch.as_tensor(index, device=value.device)]
    return value


class Observation:
    """What a controller sees at a step, with the arrays `[environments, agents, ...]`.

    The arrays are the host's own buffers, which the host reuses, so a controller never writes
    to them and copies what it keeps.
    """

    FIELDS = (
        'data',
        'first',
        'active',
        'left_players',
        'right_players',
        'previous',
        'step',
        'side',
    )

    def __init__(
        self,
        data=None,
        first=None,
        active=None,
        left_players=None,
        right_players=None,
        previous=None,
        step=None,
        side=None,
        meta=None,
        environments=None,
    ):
        """Holds the arrays of a step and the flags of its environments."""
        self.data = data
        self.first = first
        self.active = active
        self.left_players = left_players
        self.right_players = right_players
        self.previous = previous
        self.step = step
        self.side = side
        self.meta = {} if meta is None else meta
        self.environments = environments

    def select(self, index):
        """Returns the observation of some environments, which `environments` still names."""
        names = index if self.environments is None else _take(self.environments, index)
        found = dict((name, _take(getattr(self, name), index)) for name in self.FIELDS)
        return Observation(meta=self.meta, environments=names, **found)

    @property
    def count(self):
        """The environments that the observation holds."""
        return len(self.first)


class Actions:
    """The integer actions `[environments, agents]` of an answer, and any named extras."""

    def __init__(self, actions, extras=None):
        """Holds the actions, a tensor or an array on any device, and the extras beside them."""
        self.actions = actions
        self.extras = {} if extras is None else extras


def as_actions(value):
    """Returns an answer of a controller as `Actions`, from the forms that it may take."""
    if isinstance(value, Actions) or (
        hasattr(value, 'actions') and hasattr(value, 'extras') and not isinstance(value, dict)
    ):
        return Actions(value.actions, value.extras)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        return Actions(value[0], value[1])
    return Actions(value)


# Kaggle's side of the protocol.


class Struct(dict):
    """A dictionary whose keys are also attributes, as Kaggle's `Struct` that agents receive."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


_SCALARS = frozenset(
    ('active', 'ball_owned_player', 'ball_owned_team', 'designated', 'game_mode', 'steps_left')
)


def _numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def grf_observation(raw, environment, agent, own_players, other_players):
    """Returns the raw observation of one agent as GRF's dictionary with lists, as Kaggle gives."""
    result = {}
    for name, array in raw.items():
        value = array[environment, agent]
        if name in _SCALARS:
            result[name] = int(value)
        elif name == 'score':
            result[name] = [int(goals) for goals in value]
        elif name == 'sticky_actions':
            result[name] = value.astype(np.uint8).tolist()
        else:
            if name.startswith('left_'):
                value = value[:own_players]
            elif name.startswith('right_'):
                value = value[:other_players]
            if value.dtype == np.float32:
                value = value.astype(np.float64)
            elif value.dtype == np.int32:
                value = value.astype(np.int64)
            result[name] = value.tolist()
    return result


def accepts_configuration(function):
    """Returns whether an agent function takes a configuration, as Kaggle decides it."""
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2 or any(
        parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
    )


def _index(value, count):
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if not isinstance(value, (int, np.integer)) and isinstance(getattr(value, 'value', None), int):
        value = value.value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError('expected an integer action from the agent, found %r' % (value,))
    # GRF: an action indexes a Python list, so it may count from the end.
    if not -count <= int(value) < count:
        raise IndexError('expected an action from %d to %d, found %d' % (-count, count - 1, value))
    return int(value)


def action_index(value, count):
    """Returns the action index of a Kaggle agent's answer for one controlled player."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                'expected one action for the one controlled player, found %r' % (value,)
            )
        value = value[0]
    return _index(value, count)


def action_indices(value, count, players):
    """Returns the action indices of an answer for `players` controlled players, one each."""
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) != players:
        raise ValueError(
            'expected one action for each of the %d controlled players, found %r' % (players, value)
        )
    return [_index(item, count) for item in value]


def kaggle_observation(players_raw, step=0, overage=OVERAGE):
    """Returns Kaggle's observation of the players of a step."""
    return Struct(
        players_raw=players_raw,
        controlled_players=len(players_raw),
        step=step,
        remainingOverageTime=overage if overage != int(overage) else int(overage),
    )


def _inside(folder, path):
    folder = os.path.realpath(folder)
    path = os.path.realpath(path)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


@contextlib.contextmanager
def from_folder(folder):
    """Runs a block from a folder, first on `sys.path`, as Kaggle runs an agent."""
    folder = str(folder)
    previous = os.getcwd()
    sys.path.insert(0, folder)
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous)
        with contextlib.suppress(ValueError):
            sys.path.remove(folder)


def _forget(folder, before):
    for name in set(sys.modules) - before:
        origin = getattr(sys.modules.get(name), '__file__', None)
        if origin and _inside(folder, origin):
            del sys.modules[name]


def kaggle_agent(folder, file=KAGGLE_MAIN):
    """Returns the agent of a folder as Kaggle finds it, the last callable that `main.py` defines.

    The file runs from its bytes, so its encoding line decides how it reads, with the folder first
    on `sys.path` and as the working folder, and every call of the agent runs from the folder
    again. The modules that the file imports from its folder leave `sys.modules` afterwards.
    """
    path = os.path.join(folder, file)
    if not _inside(folder, path) or not os.path.isfile(path):
        raise ConfigurationError('expected %s in the folder %s, found it missing' % (file, folder))
    digest = hashlib.sha256(os.path.realpath(folder).encode()).hexdigest()[:12]
    stem = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(file)[0])
    module = types.ModuleType('cuju_agent_%s_%s' % (digest, stem))
    module.__file__ = path
    with open(path, 'rb') as handle:
        source = handle.read()
    with from_folder(folder):
        before = set(sys.modules)
        exec(compile(source, path, 'exec'), module.__dict__)
        _forget(folder, before)
    callables = [value for value in module.__dict__.values() if callable(value)]
    if not callables:
        raise ConfigurationError('expected main.py to define a callable agent, found none')
    agent = callables[-1]

    def run(*arguments):
        with from_folder(folder):
            return agent(*arguments)

    run.accepts_configuration = accepts_configuration(agent)
    return run


def _seed_globals(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.manual_seed(seed)


class Kaggle:
    """A Kaggle `main.py`, one fresh module instance for each environment, as Kaggle plays it."""

    representation = 'raw'
    control = None
    options = None
    asynchronous = False

    def __init__(self, path=KAGGLE_MAIN, configuration=None):
        """Binds a folder with a `main.py`, or a Python file, as a Kaggle agent."""
        path = os.path.abspath(path)
        if os.path.isdir(path):
            self.folder, self.file = path, KAGGLE_MAIN
        else:
            self.folder, self.file = os.path.dirname(path), os.path.basename(path)
        if not os.path.isfile(os.path.join(self.folder, self.file)):
            raise ConfigurationError(
                'expected a Kaggle agent at %s, found no %s' % (path, self.file)
            )
        self.configuration = Struct(configuration or {})
        self._agents = []
        self._steps = []
        self._count = ACTION_COUNTS['default']

    def start(self, spec):
        """Seeds the generators of the process and forgets every environment's agent."""
        self._agents = [None] * spec.environments
        self._steps = [0] * spec.environments
        self._count = ACTION_COUNTS[spec.action_set]
        _seed_globals(spec.seed)

    def act(self, observation):
        """Calls the agent of each environment with Kaggle's observation and returns its actions."""
        raw = dict((name, _numpy(value)) for name, value in observation.data.items())
        count = observation.count
        agents = raw['active'].shape[1]
        answers = np.zeros((count, agents), dtype=np.int64)
        active = _numpy(observation.active)
        first = _numpy(observation.first)
        own = _numpy(observation.left_players)
        other = _numpy(observation.right_players)
        names = None if observation.environments is None else _numpy(observation.environments)
        overage = observation.meta.get('overage', OVERAGE)
        for row in range(count):
            environment = row if names is None else int(names[row])
            if first[row] or self._agents[environment] is None:
                self._agents[environment] = kaggle_agent(self.folder, self.file)
                self._steps[environment] = 0
            if not active[row].any():
                continue
            entries = [
                grf_observation(raw, row, agent, int(own[row]), int(other[row]))
                for agent in range(agents)
            ]
            obs = kaggle_observation(entries, self._steps[environment], overage)
            function = self._agents[environment]
            answer = (
                function(obs, self.configuration)
                if function.accepts_configuration
                else function(obs)
            )
            self._steps[environment] += 1
            if agents == 1:
                answers[row] = [action_index(answer, self._count)]
            else:
                answers[row] = action_indices(answer, self._count, agents)
        return Actions(answers)

    def close(self):
        """Forgets the agents."""
        self._agents = []


# The loading of a tree of controllers, shared by the host and the child.


def resolve(entry, names):
    """Returns the object that an entry names, a short name of `names` or `module:object`."""
    if not isinstance(entry, str) or not entry:
        raise ConfigurationError('expected an entry as text, found %r' % (entry,))
    if ':' not in entry:
        if entry in names:
            return names[entry]
        raise ConfigurationError(
            "expected a name from %s or 'module:object', found %r"
            % (', '.join(sorted(names)), entry)
        )
    module_name, _, attributes = entry.partition(':')
    try:
        found = importlib.import_module(module_name)
    except ImportError as error:
        raise ConfigurationError('expected a module to import for %r, found %s' % (entry, error))
    for attribute in filter(None, attributes.split('.')):
        try:
            found = getattr(found, attribute)
        except AttributeError:
            raise ConfigurationError(
                'expected the object %r in the module %s, found none' % (attributes, module_name)
            )
    return found


_NODE_KEYS = ('entry', 'control', 'options', 'agents')


def build(node, names, path=''):
    """Builds the controller of a node, and the controllers below it through `load`."""
    if not isinstance(node, dict):
        raise ConfigurationError('expected a table for %r, found %r' % (path or 'the tree', node))
    entry = node.get('entry')
    if entry is None:
        raise ConfigurationError('expected an entry for %r, found none' % (path or 'the tree'))
    factory = resolve(entry, names)
    options = dict((key, value) for key, value in node.items() if key not in _NODE_KEYS)
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None and not any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        for key in options:
            if key not in parameters:
                raise ConfigurationError(
                    'expected a key that %s takes, found %s%s'
                    % (entry, path + '.' if path else '', key)
                )
    if parameters is not None and 'load' in parameters:

        def load(child, name='inner'):
            return build(child, names, (path + '.' if path else '') + name)

        options['load'] = load
    try:
        controller = factory(**options)
    except TypeError as error:
        if 'argument' in str(error):
            raise ConfigurationError(
                'expected the keys of %s at %s, found %s' % (entry, path or 'the tree', error)
            )
        raise
    for key in ('control', 'options'):
        if node.get(key) is not None:
            setattr(controller, key, node[key])
    return controller


class Random:
    """Draws uniform actions of the action set, which a conformance check plays against."""

    representation = 'raw'
    control = None
    options = None
    asynchronous = False

    def start(self, spec):
        """Seeds the draws."""
        self._count = ACTION_COUNTS[spec.action_set]
        self._rng = np.random.default_rng(spec.seed)

    def act(self, observation):
        """Returns one action for each agent of each environment."""
        shape = tuple(np.asarray(observation.active).shape)
        return Actions(self._rng.integers(0, self._count, size=shape))

    def close(self):
        """Has nothing to free."""


def child_names():
    """Returns the names that a child resolves without cuju, the loaders built into this file."""
    return {'kaggle': Kaggle, 'random': Random}


# The loop of the child.


def _report(answers, reason, error):
    text = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    sys.stderr.write(text)
    sys.stderr.flush()
    write(
        answers,
        'error',
        None,
        reason=reason,
        message='%s: %s' % (type(error).__name__, error),
        traceback=text,
    )


def _actions_value(answer):
    answer = as_actions(answer)
    return {
        'actions': _numpy(answer.actions),
        'extras': dict((name, _numpy(value)) for name, value in answer.extras.items()),
    }


def serve(stdin, answers, names=None):
    """Answers the frames of a host until its input ends, and returns the exit code."""
    controller = None
    names = child_names() if names is None else names
    while True:
        try:
            frame = read(stdin)
        except EOFError:
            break
        except FrameError as error:
            _report(answers, 'protocol', error)
            return 1
        kind = frame['kind']
        value = frame['value']
        try:
            if kind == 'hello':
                if frame.get('protocol') != PROTOCOL:
                    raise FrameError('expected protocol %d' % PROTOCOL)
                controller = build(value['inner'], names)
                control = getattr(controller, 'control', None)
                write(
                    answers,
                    'ready',
                    {
                        'representation': getattr(controller, 'representation', 'raw'),
                        'control': control,
                        'options': getattr(controller, 'options', None),
                    },
                    python=sys.version.split()[0],
                    numpy=np.__version__,
                )
            elif kind == 'start':
                controller.start(Spec.from_dict(value))
                write(answers, 'ready', None)
            elif kind == 'act':
                observation = Observation(**value)
                write(answers, 'actions', _actions_value(controller.act(observation)))
            elif kind == 'close':
                if controller is not None:
                    controller.close()
                    controller = None
                return 0
            else:
                raise FrameError('expected a message of the protocol, found %r' % kind)
        except Exception as error:
            reason = {'hello': 'load', 'start': 'crash', 'act': 'crash'}.get(kind, 'protocol')
            if isinstance(error, (ConfigurationError, ImportError)) and kind == 'hello':
                reason = 'load'
            _report(answers, reason, error)
            return 1
    if controller is not None:
        with contextlib.suppress(Exception):
            controller.close()
    return 0


def main():
    """Takes the first descriptor for answers, points it at standard error, and serves."""
    stdin = sys.stdin.buffer
    channel = os.dup(1)
    os.set_inheritable(channel, False)
    answers = os.fdopen(channel, 'wb', buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    code = serve(stdin, answers)
    with contextlib.suppress(OSError):
        answers.close()
    # Whatever the controller left running does not hold the process open.
    sys.stderr.flush()
    os._exit(code)


if __name__ == '__main__':
    main()
