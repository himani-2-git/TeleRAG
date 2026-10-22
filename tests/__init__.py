"""Test package initialization."""

import atexit
import os
import sys
import unittest

# Ensure PyTorch DLLs initialize cleanly before other native libraries on Windows
try:
    import torch  # noqa: F401
except ImportError:
    pass

_DATA_DIR = os.path.abspath("data")


def _snapshot_data_dir() -> dict[str, float]:
    """Snapshot modification times of all paths under ./data."""
    if not os.path.isdir(_DATA_DIR):
        return {}
    snapshot: dict[str, float] = {}
    for root, dirs, files in os.walk(_DATA_DIR):
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                snapshot[dir_path] = os.path.getmtime(dir_path)
            except OSError:
                pass
        for f in files:
            path = os.path.join(root, f)
            try:
                snapshot[path] = os.path.getmtime(path)
            except OSError:
                pass
    return snapshot


_initial_data_snapshot = _snapshot_data_dir()


def assert_data_dir_unchanged() -> None:
    """Assert that paths under ./data were not modified, created, or deleted."""
    current = _snapshot_data_dir()
    changed: list[str] = []

    for path, initial_mtime in _initial_data_snapshot.items():
        if path not in current:
            changed.append(f"{path} (deleted)")
        elif current[path] != initial_mtime:
            changed.append(f"{path} (modified)")

    for path in current:
        if path not in _initial_data_snapshot:
            changed.append(f"{path} (created)")

    if changed:
        rel_changed = [
            f"  - {os.path.relpath(p.split(' ')[0], os.getcwd())} {' '.join(p.split(' ')[1:])}"
            for p in changed
        ]
        msg = (
            "Tests must never touch the real ./data directory!\n"
            "The following paths under ./data were modified, created, or deleted during test execution:\n"
            + "\n".join(rel_changed)
        )
        raise AssertionError(msg)


_orig_testcase_run = unittest.TestCase.run


def _guarded_testcase_run(self, result=None):
    res = _orig_testcase_run(self, result)
    assert_data_dir_unchanged()
    return res


unittest.TestCase.run = _guarded_testcase_run


def _on_test_process_exit():
    try:
        assert_data_dir_unchanged()
    except AssertionError as e:
        sys.stderr.write(f"\n{e}\n")
        os._exit(1)


atexit.register(_on_test_process_exit)
