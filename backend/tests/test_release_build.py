import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_release.py"
SPEC = importlib.util.spec_from_file_location("build_release", SCRIPT)
assert SPEC and SPEC.loader
build_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_release)


def test_production_release_contains_runtime_but_not_evaluation_tools(tmp_path) -> None:
    destination = tmp_path / "production.zip"
    build_release.write_zip(destination, build_release.production_files())

    with ZipFile(destination) as archive:
        names = set(archive.namelist())

    assert "backend/calibration.json" in names
    assert "backend/app/main.py" in names
    assert "extension/manifest.json" in names
    assert "setup.cmd" in names
    assert "start.cmd" in names
    assert "backend/evaluation/compare_models.py" not in names
    assert "backend/tests/test_calibration.py" not in names
    assert "backend/requirements-dev.txt" not in names
    assert not any(name.endswith(".ses") for name in names)


def test_development_release_contains_evaluation_tools() -> None:
    names = build_release.development_files()

    assert "backend/evaluation/compare_models.py" in names
    assert "backend/tests/test_model_comparison.py" in names
    assert "backend/requirements-dev.txt" in names


def test_backend_deployment_release_contains_runtime_and_public_docs() -> None:
    names = build_release.backend_deployment_files()

    assert "backend/Dockerfile" in names
    assert "backend/app/run.py" in names
    assert "backend/calibration.json" in names
    assert "DEPLOYMENT.md" in names
    assert "PRIVACY.md" in names
    assert "extension/manifest.json" not in names
    assert "backend/evaluation/evaluate_web.py" not in names


def test_store_extension_has_fixed_https_api_and_minimal_host_access() -> None:
    files = build_release.store_extension_files("https://api.example.com/v1/")
    manifest = json.loads(files["manifest.json"])
    background = files["background.js"].decode("utf-8")

    assert "manifest.json" in files
    assert "extension/manifest.json" not in files
    assert "icons/icon16.png" in files
    assert "icons/icon32.png" in files
    assert "icons/icon48.png" in files
    assert "icons/icon128.png" in files
    assert manifest["host_permissions"] == ["https://api.example.com/*"]
    assert manifest["permissions"] == ["activeTab", "scripting", "storage"]
    assert "optional_host_permissions" not in manifest
    assert 'const BUILD_MODE = "public";' in background
    assert 'apiBase: "https://api.example.com/v1",' in background
    assert "127.0.0.1" not in background
    packaged_code = "\n".join(
        files[name].decode("utf-8")
        for name in ("background.js", "content.js", "popup.js")
    )
    assert "eval(" not in packaged_code
    assert "new Function(" not in packaged_code
    assert "Start the local backend" not in packaged_code
    assert "Run setup.cmd" not in packaged_code
    assert 'type === "PING_BACKEND"' not in packaged_code


def test_submission_guide_is_in_source_but_not_store_package() -> None:
    assert "CWS_SUBMISSION.md" in build_release.development_files()
    files = build_release.store_extension_files("https://api.example.com")
    assert "CWS_SUBMISSION.md" not in files


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com",
        "https://user:password@api.example.com",
        "https://api.example.com?debug=true",
        "not-a-url",
    ],
)
def test_store_extension_rejects_unsafe_api_url(url: str) -> None:
    with pytest.raises(ValueError):
        build_release.store_extension_files(url)
