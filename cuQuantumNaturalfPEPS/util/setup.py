#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import mmap
import os
import re
import shutil
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent.parent
VERSION_FILE = PACKAGE_DIR / "c_api_version.txt"
DEFAULT_LIBRARY = PACKAGE_DIR / "build" / "cuda" / "qnpeps.so"
STAMP_FILE = PACKAGE_DIR / "build" / "setup.signature"
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
COMPILED_VERSION_PATTERN = re.compile(
    rb"cuQuantumNaturalfPEPS ([0-9]+\.[0-9]+\.[0-9]+)"
    rb"(?: \([0-9]{4}-[0-9]{2}-[0-9]{2}\))?"
)


class VersionState(Enum):
    MISSING = "missing"
    MATCH = "match"
    MISMATCH = "mismatch"
    INVALID = "invalid"


def version_tuple(version: str) -> Tuple[int, int, int]:
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"expected major.minor.patch, got {version!r}")
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def expected_version() -> str:
    try:
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"failed to read {VERSION_FILE}: {error}") from error
    try:
        version_tuple(version)
    except ValueError as error:
        raise RuntimeError(f"invalid C API version in {VERSION_FILE}: {error}") from error
    return version


def selected_library() -> Path:
    override = os.environ.get("QNPEPS_LIB", "")
    return Path(override).expanduser() if override else DEFAULT_LIBRARY


def embedded_versions(library: Path) -> Tuple[str, ...]:
    try:
        with library.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as contents:
                versions = {
                    match.group(1).decode("ascii")
                    for match in COMPILED_VERSION_PATTERN.finditer(contents)
                }
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to inspect {library}: {error}") from error
    return tuple(sorted(versions, key=version_tuple))


def inspect_version(library: Optional[Path] = None) -> Tuple[VersionState, Optional[str], str]:
    library = selected_library() if library is None else library
    wanted = expected_version()

    try:
        if not library.is_file() or library.stat().st_size == 0:
            return VersionState.MISSING, None, wanted
    except OSError as error:
        raise RuntimeError(f"failed to inspect {library}: {error}") from error

    versions = embedded_versions(library)
    if len(versions) != 1:
        detail = None if not versions else ", ".join(versions)
        return VersionState.INVALID, detail, wanted

    compiled = versions[0]
    state = VersionState.MATCH if compiled == wanted else VersionState.MISMATCH
    return state, compiled, wanted


def check_version(library: Optional[Path] = None) -> VersionState:
    try:
        state, compiled, wanted = inspect_version(library)
    except RuntimeError as error:
        print(f"[setup.py] {error}", file=sys.stderr)
        return VersionState.INVALID

    if state is VersionState.MISSING:
        print("No compiled .so found.")
    elif state is VersionState.MATCH:
        print(f'.so was compiled for version "{compiled}", which is correct.')
    elif state is VersionState.MISMATCH:
        print(f'.so was compiled for version "{compiled}" (expected "{wanted}").')
    else:
        detail = "no version" if compiled is None else f'versions "{compiled}"'
        print(
            f".so contains {detail} (expected version is \"{wanted}\").",
            file=sys.stderr,
        )
    return state


def source_files() -> Iterable[Path]:
    files = set()
    for directory_name in ("cuda", "src", "util"):
        directory = PACKAGE_DIR / directory_name
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(PACKAGE_DIR)
            if directory_name == "cuda" and len(relative.parts) > 1:
                if relative.parts[1].startswith("build"):
                    continue
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            files.add(path)

    files.update((PACKAGE_DIR / "setup.sh", VERSION_FILE))
    for root in (PACKAGE_DIR, PACKAGE_DIR.parent):
        for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
            path = root / name
            if path.is_file():
                files.add(path)
    return sorted(files, key=lambda path: path.as_posix())


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def executable_path(name: str) -> str:
    executable = shutil.which(name)
    return str(Path(executable).resolve()) if executable else "<not-found>"


def setup_signature() -> str:
    digest = hashlib.sha256()
    cuda_home = os.environ.get("CUDA_HOME", "")
    resolved_cuda_home = str(Path(cuda_home).resolve()) if cuda_home else ""
    fields = (
        ("setup-signature", "2"),
        ("root", str(PACKAGE_DIR)),
        ("julia", executable_path("julia")),
        ("nvcc", executable_path("nvcc")),
        ("cuda-home", resolved_cuda_home),
        ("cuda-version", os.environ.get("QNPEPS_CUDA_VERSION", "")),
        ("julia-depot", os.environ.get("JULIA_DEPOT_PATH", "")),
    )
    for name, value in fields:
        digest.update(f"{name}={value}\n".encode("utf-8"))
    for path in source_files():
        relative = path.relative_to(PACKAGE_DIR.parent)
        digest.update(f"file={relative.as_posix()}\0{hash_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def active_for_this_checkout() -> bool:
    active = os.environ.get("QNPEPS_ACTIVE_ROOT", "")
    if not active:
        return False
    try:
        return Path(active).expanduser().resolve() == PACKAGE_DIR
    except OSError:
        return False


def run(command: Sequence[str]) -> bool:
    return subprocess.run(command, cwd=str(PACKAGE_DIR.parent), check=False).returncode == 0


def visible_gpu_exists() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    return (
        subprocess.run(
            (nvidia_smi, "-L"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def write_stamp(signature: str) -> None:
    STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".setup.signature.",
            dir=str(STAMP_FILE.parent),
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(signature + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, STAMP_FILE)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def setup(force: bool, show_path: bool) -> int:
    if not active_for_this_checkout():
        print("[setup.py] cuQuantumNaturalfPEPS environment activation failed", file=sys.stderr)
        return 1
    try:
        wanted = expected_version()
        signature = setup_signature()
    except (OSError, RuntimeError) as error:
        print(f"[setup.py] {error}", file=sys.stderr)
        return 1

    library_exists = DEFAULT_LIBRARY.is_file() and DEFAULT_LIBRARY.stat().st_size > 0
    stamp_matches = False
    if STAMP_FILE.is_file():
        try:
            stamp_matches = STAMP_FILE.read_text(encoding="utf-8").strip() == signature
        except OSError:
            stamp_matches = False

    try:
        state, compiled, _ = inspect_version(DEFAULT_LIBRARY)
    except RuntimeError as error:
        print(f"[setup.py] {error}", file=sys.stderr)
        return 1

    if not force and state is VersionState.MATCH and stamp_matches:
        print("cuQuantumNaturalfPEPS ready")
        return 0

    remove_library = False
    if force:
        print("Forcing rebuild of qnpeps.so.")
        remove_library = library_exists
    elif state is VersionState.MISMATCH:
        print(f'Updating qnpeps.so from "{compiled}" to "{wanted}".')
        remove_library = True
    elif state is VersionState.INVALID:
        print(f'Replacing qnpeps.so because the version "{compiled}" is invalid.')
        remove_library = library_exists
    elif state is VersionState.MATCH:
        print("Rebuilding qnpeps.so")

    if remove_library:
        try:
            DEFAULT_LIBRARY.unlink()
        except OSError as error:
            print(f"[setup.py] failed to remove {DEFAULT_LIBRARY}: {error}", file=sys.stderr)
            return 1

    command_args = ["--show-path"] if show_path else []
    print("Configuring Julia")
    if not run(
        (
            "julia",
            "--startup-file=no",
            str(PACKAGE_DIR / "util" / "configure_julia.jl"),
        )
    ):
        return 1

    print("Building qnpeps.so")
    if not run((str(PACKAGE_DIR / "util" / "build_cuda.sh"), *command_args)):
        return 1

    if visible_gpu_exists():
        print("Checking Julia and CUDA runtime")
        if not run(
            (
                "julia",
                "--startup-file=no",
                f"--project={PACKAGE_DIR}",
                str(PACKAGE_DIR / "util" / "runtime_info.jl"),
            )
        ):
            return 1

    try:
        write_stamp(setup_signature())
    except OSError as error:
        print(f"[setup.py] failed to write {STAMP_FILE}: {error}", file=sys.stderr)
        return 1
    print("cuQuantumNaturalfPEPS setup complete")
    return 0


def parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="source cuQuantumNaturalfPEPS/setup.sh",
        description="Activate cuQuantumNaturalfPEPS and build its CUDA library.",
    )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "-c", "--check", action="store_true", help="check whether this shell is ready to use the library"
    )
    checks.add_argument(
        "-cv", "--check-version", action="store_true", help="check the compiled version of the qnpeps.so file"
    )
    checks.add_argument(
        "--api-version", action="store_true", help="print the expected version of the qnpeps.so file"
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="recompile the qnpeps.so file"
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="print PATH during activation and compilation",
    )
    parsed = parser.parse_args(arguments)
    if (parsed.check or parsed.check_version or parsed.api_version) and (
        parsed.force or parsed.show_path
    ):
        parser.error("check modes cannot be combined with build options")
    return parsed


def main(arguments: Sequence[str]) -> int:
    if sys.version_info < (3, 9):
        print("[setup.py] Python 3.9 or newer is required", file=sys.stderr)
        return 1
    args = parse_args(arguments)
    if args.api_version:
        try:
            print(expected_version())
        except RuntimeError as error:
            print(f"[setup.py] {error}", file=sys.stderr)
            return 1
        return 0
    if args.check:
        ready = active_for_this_checkout() and all(
            shutil.which(command) is not None for command in ("julia", "nvcc")
        )
        if ready:
            print("Shell is ready to use cuQuantumNaturalfPEPS CUDA code.")
            return 0
        print("Shell is not ready to use cuQuantumNaturalfPEPS CUDA code.", file=sys.stderr)
        return 1
    if args.check_version:
        return 0 if check_version() is VersionState.MATCH else 1
    return setup(force=args.force, show_path=args.show_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
