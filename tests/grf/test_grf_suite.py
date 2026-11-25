# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Runs GRF's own tests of `gfootball.env`, ported with only their imports changed, on cuju."""

import importlib
import importlib.util
import os
import re
import sys
import unittest

import pytest

pytest.importorskip('absl')
pytest.importorskip('psutil')

MODULES = (
    'football_action_set_test',
    'football_env_test',
    'gym_test',
    'observation_rotation_test',
    'script_helpers_test',
    'wrappers_test',
)
FULL = os.environ.get('CUJU_GRF_SUITE_FULL', '') == '1'
_VIDEO_TESTS = (
    'ScriptHelpersTest.test__replay',
    'ScriptHelpersTest.test_dump_to_video',
)
_TEXT_TEST = 'ScriptHelpersTest.test_dump_to_txt'
_VIDEO_SKIP_REASON = 'GRF replay and video export need OpenCV, which provides cv2'

# GRF's tests of `gym.make` that fail in GRF itself under a later gym that runs GRF.
_CONSISTENT = (
    'SingleAgentWrapperTest.test_consistent_observation_and_action',
    'SingleAgentWrapperTest.test_consistent_observation_and_action_floats',
    'SingleAgentWrapperTest.test_consistent_observation_and_action_stacked',
)
_GYM_FAILURES = {
    (0, 22): (*_CONSISTENT, 'GymTest.test_environment0', 'GymTest.test_environment1'),
    (0, 23): (*_CONSISTENT, 'GymTest.test_environment0', 'GymTest.test_environment1'),
    (0, 24): _CONSISTENT,
}


def _gym_failures():
    from cuju.gfootball import _gym

    if _gym.VENDORED:
        return ()
    found = re.match(r'(\d+)\.(\d+)', _gym.__version__)
    return _GYM_FAILURES.get((int(found[1]), int(found[2])), ())


def _module(name):
    os.environ['UNITTEST_IN_DOCKER'] = '1'
    module = importlib.import_module(f'cuju.gfootball.env.{name}')
    if hasattr(module, 'fast_run'):
        module.fast_run = not FULL
    return module


def _cases():
    cases = []
    for name in MODULES:
        try:
            module = _module(name)
        except ImportError as error:
            cases.append(pytest.param(name, None, marks=pytest.mark.skip(reason=str(error))))
            continue
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        stack = [suite]
        while stack:
            item = stack.pop()
            if isinstance(item, unittest.TestSuite):
                stack.extend(reversed(list(item)))
            else:
                test = f'{type(item).__name__}.{item._testMethodName}'
                cases.append(pytest.param(name, test, id=test))
    return cases


def _run_case(case, test):
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
    result = unittest.TestResult()
    if test in _VIDEO_TESTS and importlib.util.find_spec('cv2') is None:
        result.addSkip(case, _VIDEO_SKIP_REASON)
        return result
    if test == _TEXT_TEST and not case.current_dumps():
        # GRF's text export uses the dump that its replay case usually creates.
        case.generate_replay()
    case.run(result)
    return result


@pytest.mark.parametrize(('module_name', 'test'), _cases())
def test_grf(module_name, test):
    module = _module(module_name)
    kind, method = test.split('.')
    case = getattr(module, kind)(method)
    result = _run_case(case, test)
    for _, reason in result.skipped:
        pytest.skip(reason)
    failures = result.errors + result.failures
    if test in _gym_failures():
        assert failures, f'{test} passes, where GRF fails it under this gym'
        pytest.xfail(f'GRF fails {test} under this gym as well')
    assert not failures, '\n'.join(text for _, text in failures)


def test_script_helpers_without_opencv(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'cv2', None)
    module = _module('script_helpers_test')
    monkeypatch.setattr(module, 'test_tmpdir', str(tmp_path))
    for test in _VIDEO_TESTS:
        case = module.ScriptHelpersTest(test.split('.')[1])
        result = _run_case(case, test)
        assert result.skipped == [(case, _VIDEO_SKIP_REASON)]
        assert result.testsRun == 0
    case = module.ScriptHelpersTest('test_dump_to_txt')
    result = _run_case(case, _TEXT_TEST)
    assert result.testsRun == 1
    assert result.wasSuccessful() and not result.skipped
