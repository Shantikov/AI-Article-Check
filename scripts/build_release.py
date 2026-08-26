from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
FIXED_ZIP_DATE = (2026, 8, 26, 0, 0, 0)

ROOT_FILES = {
    "CWS_SUBMISSION.md",
    "DEPLOYMENT.md",
    "LICENSE",
    "PRIVACY.md",
    "README.md",
    "README.ru.md",
    "STORE_LISTING.md",
    "THIRD_PARTY_NOTICES.md",
    "docker-compose.yml",
    "setup.cmd",
    "setup.ps1",
    "start.cmd",
    "start.ps1",
}
DEPLOYMENT_ROOT_FILES = {
    "DEPLOYMENT.md",
    "LICENSE",
    "PRIVACY.md",
    "THIRD_PARTY_NOTICES.md",
}
BACKEND_RUNTIME_FILES = {
    "backend/.dockerignore",
    "backend/.env.example",
    "backend/Dockerfile",
    "backend/calibration.json",
    "backend/requirements.txt",
}
DEVELOPMENT_FILES = {
    ".gitignore",
    "backend/pytest.ini",
    "backend/requirements-dev.txt",
    "scripts/build_release.py",
}


def extension_version() -> str:
    manifest = json.loads((ROOT / "extension/manifest.json").read_text(encoding="utf-8"))
    return str(manifest["version"])


def files_under(relative_directory: str) -> set[str]:
    directory = ROOT / relative_directory
    return {
        path.relative_to(ROOT).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix not in {".pyc", ".ses"}
    }


def production_files() -> set[str]:
    return (
        ROOT_FILES
        | BACKEND_RUNTIME_FILES
        | files_under("backend/app")
        | files_under("extension")
        | files_under("store-assets")
    )


def development_files() -> set[str]:
    return (
        production_files()
        | DEVELOPMENT_FILES
        | files_under("backend/evaluation")
        | files_under("backend/tests")
    )


def backend_deployment_files() -> set[str]:
    return DEPLOYMENT_ROOT_FILES | BACKEND_RUNTIME_FILES | files_under("backend/app")


def write_zip(destination: Path, relative_paths: set[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_path in sorted(relative_paths):
            data = (ROOT / relative_path).read_bytes()
            info = ZipInfo(relative_path, FIXED_ZIP_DATE)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def public_api_base(value: str) -> tuple[str, str]:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "The public API base must be an HTTPS URL without credentials, "
            "a query, or a fragment"
        )
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return normalized, origin


def store_extension_files(api_base: str) -> dict[str, bytes]:
    normalized, origin = public_api_base(api_base)
    files: dict[str, bytes] = {}
    for source in sorted(files_under("extension")):
        archive_name = source.removeprefix("extension/")
        files[archive_name] = (ROOT / source).read_bytes()

    manifest = json.loads(files["manifest.json"].decode("utf-8"))
    manifest["host_permissions"] = [f"{origin}/*"]
    manifest.pop("optional_host_permissions", None)
    files["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )

    background = files["background.js"].decode("utf-8")
    local_mode = 'const BUILD_MODE = "local";'
    local_base = 'apiBase: "http://127.0.0.1:8787",'
    if background.count(local_mode) != 1 or background.count(local_base) != 1:
        raise RuntimeError("Public-build markers in extension/background.js changed")
    background = background.replace(local_mode, 'const BUILD_MODE = "public";')
    background = background.replace(local_base, f"apiBase: {json.dumps(normalized)},")
    files["background.js"] = background.encode("utf-8")
    return files


def write_data_zip(destination: Path, files: dict[str, bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_name in sorted(files):
            info = ZipInfo(archive_name, FIXED_ZIP_DATE)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, files[archive_name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AI Article Check archives.")
    parser.add_argument(
        "--public-api-base",
        help="Build the Chrome Web Store ZIP against this HTTPS API base.",
    )
    args = parser.parse_args()
    version = extension_version()
    production_path = DIST / f"AI-Article-Check-{version}.zip"
    development_path = DIST / f"AI-Article-Check-{version}-dev.zip"
    source_path = DIST / f"AI-Article-Check-{version}-source.zip"
    github_update_path = DIST / f"AI-Article-Check-{version}-GitHub-update.zip"
    backend_path = DIST / f"AI-Article-Check-{version}-backend.zip"
    write_zip(production_path, production_files())
    write_zip(development_path, development_files())
    write_zip(source_path, development_files())
    write_zip(github_update_path, development_files())
    write_zip(backend_path, backend_deployment_files())
    print(f"Production archive: {production_path}")
    print(f"Development archive: {development_path}")
    print(f"Source archive: {source_path}")
    print(f"GitHub update archive: {github_update_path}")
    print(f"Backend deployment archive: {backend_path}")
    if args.public_api_base:
        store_path = DIST / f"AI-Article-Check-{version}-store.zip"
        write_data_zip(store_path, store_extension_files(args.public_api_base))
        print(f"Chrome Web Store archive: {store_path}")
    else:
        print(
            "Chrome Web Store archive: not built "
            "(pass --public-api-base https://your-api.example)"
        )


if __name__ == "__main__":
    main()
