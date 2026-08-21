"""Verify a BUILT distribution, in a throwaway environment, from outside the repository.

WHY THIS EXISTS. Verifying 0.10.2 by hand reported `AGREE: True` for a wheel that had not been
built. A shell `&&` chain broke on a glob that matched nothing, so the build never ran; the check
then imported `gradecore` from the current working directory (the repo) and read its version from
the source tree's egg-info. Both numbers agreed and both came from the working copy. Nothing in
the check could tell "verified the built distribution" from "verified the directory I was
standing in".

So this refuses on the three ways that illusion is produced, before comparing any version:

  1. no distribution file matches the expected version
  2. the import resolves outside the temporary environment's site-packages
  3. the working directory is inside the repository

A green version comparison is not evidence that a distribution was tested. Usage:

    python tools/verify_release.py 0.10.2
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


def _fail(msg: str) -> None:
    raise SystemExit(f"release verification FAILED: {msg}")


def verify(expected: str, repo: pathlib.Path | None = None) -> None:
    repo = (repo or pathlib.Path(__file__).resolve().parent.parent).resolve()
    # Refuse a repo root we cannot identify. `__file__.parent.parent` is only the repository when
    # this script sits in its tools/ directory; a copy elsewhere resolves to something like "/",
    # which is a parent of every path and makes the cwd check below refuse for a false reason.
    # Found by a mutation that relocated this file and got "cwd /private/tmp is inside the
    # repository". A guard that fires for the wrong reason is not a working guard.
    if not (repo / "pyproject.toml").is_file():
        _fail(f"{repo} has no pyproject.toml, so it is not the repository root. This script must "
              f"run from the repo's tools/ directory, or be given an explicit repo path.")

    cwd = pathlib.Path.cwd().resolve()
    if cwd == repo or repo in cwd.parents:
        _fail(f"cwd {cwd} is inside the repository. Run this from elsewhere, or an import can "
              f"resolve to the working copy and the check proves nothing about the distribution.")

    wheels = sorted((repo / "dist").glob(f"*-{expected}-*.whl"))
    if not wheels:
        _fail(f"no wheel in {repo / 'dist'} matches version {expected}. Build first. An absent "
              f"artifact must never read as a pass.")
    wheel = wheels[-1]

    with tempfile.TemporaryDirectory() as td:
        env = pathlib.Path(td) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True, capture_output=True)
        py = env / "bin" / "python"
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", str(wheel)],
                       check=True, capture_output=True)
        probe = (
            "import gradecore, json\n"
            "from importlib.metadata import version\n"
            "print(json.dumps({'file': gradecore.__file__,\n"
            "                  'module': gradecore.__version__,\n"
            "                  'dist': version('gradecore')}))\n"
        )
        got = json.loads(subprocess.run([str(py), "-c", probe], cwd=td, check=True,
                                        capture_output=True, text=True).stdout)
        resolved = pathlib.Path(got["file"]).resolve()
        if str(env.resolve()) not in str(resolved):
            _fail(f"import resolved to {resolved}, outside the temporary environment. The source "
                  f"tree answered, not the installed distribution.")

    if got["module"] != expected or got["dist"] != expected:
        _fail(f"version mismatch: module {got['module']!r}, distribution {got['dist']!r}, "
              f"expected {expected!r}")

    print(f"release {expected} verified")
    print(f"  wheel     {wheel.name}")
    print(f"  imported  {resolved}")
    print(f"  module    {got['module']}   distribution {got['dist']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/verify_release.py VERSION")
    verify(sys.argv[1])
