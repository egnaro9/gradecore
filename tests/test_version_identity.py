"""The module's version must identify the distribution that supplied the code.

THE DEFECT THIS PINS. gradecore 0.10.1 shipped with `__version__ = "0.10.0"`. The wheel's
METADATA said Version: 0.10.1 and `gradecore.__version__` said 0.10.0, because the release bumped
pyproject.toml and left a second, hand-maintained literal behind. `pip show` and the runtime
attribute disagreed about which release was installed.

That is a provenance failure in a provenance library. Anything recording "graded by gradecore
X.Y.Z" from the module attribute recorded the wrong release, and nothing noticed, because no test
compared the two sources.

__version__ is now read from installed metadata, so the literal cannot drift. This test is the
guard that keeps it that way.
"""
import importlib.metadata

import pytest

import gradecore


def test_module_version_matches_the_installed_distribution():
    try:
        dist = importlib.metadata.version("gradecore")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("gradecore is not installed; the source-tree fallback is in use")
    assert gradecore.__version__ == dist, (
        f"gradecore.__version__ is {gradecore.__version__!r} but the installed distribution is "
        f"{dist!r}. These identify different releases, which is how 0.10.1 shipped reporting "
        f"itself as 0.10.0."
    )


def test_the_version_is_not_a_bare_hand_maintained_literal():
    """The literal is a source-tree fallback only. If someone reverts to a plain assignment the
    drift becomes possible again, so the shape is checked and not only the value."""
    import pathlib
    src = pathlib.Path(gradecore.__file__).read_text()
    assert "_dist_version(" in src, (
        "__version__ is no longer derived from distribution metadata, so it can drift from "
        "pyproject.toml again, which is exactly what happened in 0.10.1"
    )
