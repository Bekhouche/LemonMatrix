"""Guards against the repo-root schema/ files silently drifting from the
packaged copy under src/lemonmatrix/schema/ that setuptools actually ships.

Real bug found and fixed: schema/*.json used to be real, independently
maintained files, while pyproject.toml's package-data only ever pulls from
src/lemonmatrix/schema/*.json. Only result.schema.json had ever been copied
there by hand -- the five newer schemas (classify/tts/stt/imagegen/audiogen)
were silently missing from every built wheel. Confirmed by actually building
a wheel and installing it into a fresh venv. Fixed by making the repo-root
copies symlinks into the package instead of independent files, so there is
exactly one file on disk and drift is structurally impossible -- this test
exists to catch a regression (e.g. someone overwriting a symlink with a real
file again) before it ships silently.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SCHEMA_DIR = REPO_ROOT / "schema"
PACKAGE_SCHEMA_DIR = REPO_ROOT / "src" / "lemonmatrix" / "schema"


def test_every_repo_root_schema_file_is_a_symlink_into_the_package():
    schema_files = sorted(REPO_SCHEMA_DIR.glob("*.json"))
    assert schema_files, "expected at least one schema file at the repo root"

    for path in schema_files:
        assert path.is_symlink(), (
            f"{path} is a real file, not a symlink -- it will silently drift from "
            f"{PACKAGE_SCHEMA_DIR / path.name}, which is what actually ships in the wheel"
        )
        target = (path.parent / path.readlink()).resolve()
        assert target == (PACKAGE_SCHEMA_DIR / path.name).resolve(), (
            f"{path} does not point into {PACKAGE_SCHEMA_DIR}"
        )


def test_every_package_schema_file_is_shipped_and_exposed_at_the_repo_root():
    # The inverse check: every schema setuptools will package must also be
    # reachable from the repo root (so validate.py's dev-mode lookup and the
    # installed-package lookup can never see different content).
    package_files = sorted(PACKAGE_SCHEMA_DIR.glob("*.json"))
    assert package_files, "expected at least one schema file in the packaged directory"

    repo_root_names = {p.name for p in REPO_SCHEMA_DIR.glob("*.json")}
    for path in package_files:
        assert path.name in repo_root_names, f"{path.name} is packaged but missing from {REPO_SCHEMA_DIR}"
