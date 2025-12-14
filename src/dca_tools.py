#!/usr/bin/env python3
"""
DCA utility/validation toolbox (single-file entrypoint).

This script intentionally uses ONLY the Python standard library so it can run
before dependencies are installed.

Commands:
  - check-python
  - check-longpaths
  - validate-requirements <file>
  - write-checksums
  - verify-checksums
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _is_windows() -> bool:
    return os.name == "nt"


def cmd_check_python(_: argparse.Namespace) -> int:
    v = sys.version_info
    if v.major == 3 and 8 <= v.minor <= 11:
        print(f"[OK] Python {v.major}.{v.minor}.{v.micro} is supported (3.8–3.11).")
        return 0
    _eprint(f"[FAIL] Python {v.major}.{v.minor}.{v.micro} is not supported.")
    _eprint("       Please use Python 3.8–3.11 (PyCaret compatibility).")
    return 1


def _read_longpaths_enabled_windows() -> Optional[bool]:
    """
    Returns:
      True/False if registry value can be read, otherwise None.
    """
    if not _is_windows():
        return None
    try:
        import winreg  # type: ignore

        key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return bool(int(value))
    except Exception:
        return None


def cmd_check_longpaths(_: argparse.Namespace) -> int:
    if not _is_windows():
        print("[OK] Non-Windows OS detected; Windows Long Path setting is not applicable.")
        return 0

    enabled = _read_longpaths_enabled_windows()
    if enabled is True:
        print("[OK] Windows Long Paths: ENABLED (LongPathsEnabled=1).")
        return 0
    if enabled is False:
        _eprint("[WARN] Windows Long Paths: DISABLED (LongPathsEnabled=0).")
        _eprint("       This can cause pip install failures with deep package paths.")
        _eprint("       Recommended fixes:")
        _eprint(r"         - Enable Long Paths (requires admin) and reboot:")
        _eprint(r"           HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1")
        _eprint(r"         - Or use a short project path (e.g., C:\DCA_App) and a short venv location.")
        return 1

    _eprint("[INFO] Could not determine Windows Long Paths setting (no access to registry?).")
    _eprint("       If pip fails with very deep paths, enable Long Paths or use a shorter path.")
    return 0


_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(.*)$")


def _iter_requirement_lines(text: str) -> Iterable[Tuple[int, str]]:
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield i, line


def validate_requirements_text(text: str) -> Tuple[bool, List[str]]:
    """
    Very lightweight validation:
    - UTF-8 decodable is checked by the caller
    - disallow NUL bytes
    - parse out a plausible package name
    - basic sanity warnings for suspicious names
    """
    issues: List[str] = []

    if "\x00" in text:
        issues.append("ERROR: File contains NUL (\\x00) bytes (likely corrupted).")
        return False, issues

    for line_num, line in _iter_requirement_lines(text):
        if "\t" in line:
            issues.append(f"WARNING: Line {line_num}: contains tabs; use spaces instead.")

        if line.startswith(("-", "--")):
            # Allow pip directives like -r, --extra-index-url, etc.
            continue

        m = _REQ_NAME_RE.match(line)
        if not m:
            issues.append(f"ERROR: Line {line_num}: could not parse requirement: {line!r}")
            continue

        name = m.group(1)

        # Package names should be ASCII-ish. If we see non-ascii in name, treat as error.
        if any(ord(ch) > 127 for ch in name):
            issues.append(f"ERROR: Line {line_num}: non-ASCII characters in package name: {name!r}")
            continue

        # Suspicion heuristics (warnings only)
        if re.fullmatch(r"[a-z]{15,}", name):
            issues.append(
                f"WARNING: Line {line_num}: unusually long lowercase-only package name: {name!r}"
            )
        if re.search(r"(.)\1{5,}", name):
            issues.append(
                f"WARNING: Line {line_num}: excessive repeated characters in package name: {name!r}"
            )

    ok = not any(s.startswith("ERROR:") for s in issues)
    return ok, issues


def cmd_validate_requirements(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        _eprint(f"[FAIL] Requirements file not found: {path}")
        return 1

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        _eprint("[FAIL] Requirements file encoding issue. Ensure it is UTF-8 encoded.")
        _eprint(f"       File: {path}")
        return 1
    except Exception as e:
        _eprint(f"[FAIL] Could not read requirements file: {e}")
        _eprint(f"       File: {path}")
        return 1

    ok, issues = validate_requirements_text(text)
    print("=" * 60)
    print("Requirements File Validator")
    print("=" * 60)
    print(f"Validating: {path.name}\n")

    for msg in issues:
        print(msg)

    if ok:
        print("\n[OK] Requirements file looks parseable (basic checks passed).")
        return 0
    print("\n[FAIL] Requirements file validation failed.")
    return 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_checksums_manifest(manifest_path: Path, rows: List[Tuple[str, str]]) -> None:
    manifest_path.write_text(
        "# File Integrity Checksums\n"
        "# Generated by src/dca_tools.py write-checksums\n"
        "# Format: SHA256_HASH  RELATIVE_PATH\n\n"
        + "".join(f"{checksum}  {relpath}\n" for checksum, relpath in rows),
        encoding="utf-8",
    )


def cmd_write_checksums(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    manifest = root / args.output

    files: List[str]
    if args.files:
        files = args.files
    else:
        files = [
            "requirements.txt",
            "launch_app.bat",
            "config/config.json",
            "streamlit_app/app.py",
            "README.md",
        ]

    rows: List[Tuple[str, str]] = []
    missing: List[str] = []
    for rel in files:
        p = (root / rel).resolve()
        if not p.exists() or not p.is_file():
            missing.append(rel)
            continue
        rows.append((_sha256_file(p), rel.replace("\\", "/")))

    _write_checksums_manifest(manifest, rows)
    print(f"[OK] Wrote {len(rows)} checksums to: {manifest}")
    if missing:
        _eprint("[WARN] Missing files (not included in manifest):")
        for rel in missing:
            _eprint(f"  - {rel}")
    return 0


def _read_manifest(manifest_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        checksum, relpath = parts
        out[relpath.strip()] = checksum.strip()
    return out


def cmd_verify_checksums(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    manifest = root / args.manifest
    if not manifest.exists():
        _eprint(f"[FAIL] Manifest not found: {manifest}")
        return 1

    expected = _read_manifest(manifest)
    if not expected:
        _eprint(f"[FAIL] Manifest is empty or invalid: {manifest}")
        return 1

    ok_count = 0
    fail_count = 0
    missing_count = 0

    for relpath, exp in sorted(expected.items()):
        target = (root / relpath).resolve()
        if not target.exists():
            print(f"[MISSING] {relpath}")
            missing_count += 1
            continue
        got = _sha256_file(target)
        if got.lower() == exp.lower():
            print(f"[OK] {relpath}")
            ok_count += 1
        else:
            print(f"[FAIL] {relpath}")
            print(f"       Expected: {exp}")
            print(f"       Actual:   {got}")
            fail_count += 1

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"Verified (OK): {ok_count}")
    print(f"Failed:        {fail_count}")
    print(f"Missing:       {missing_count}")

    return 0 if (fail_count == 0) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="src/dca_tools.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("check-python", help="Verify Python version is 3.8–3.11")
    sp.set_defaults(func=cmd_check_python)

    sp = sub.add_parser("check-longpaths", help="Check Windows LongPathsEnabled setting")
    sp.set_defaults(func=cmd_check_longpaths)

    sp = sub.add_parser("validate-requirements", help="Validate a requirements file (basic checks)")
    sp.add_argument("file", help="Path to requirements file (e.g., requirements.txt)")
    sp.set_defaults(func=cmd_validate_requirements)

    sp = sub.add_parser("write-checksums", help="Write file_checksums.txt for key project files")
    sp.add_argument("--root", default=str(PROJECT_ROOT), help="Project root directory")
    sp.add_argument("--output", default="file_checksums.txt", help="Output manifest filename")
    sp.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional list of relative file paths to include (defaults to a curated list)",
    )
    sp.set_defaults(func=cmd_write_checksums)

    sp = sub.add_parser("verify-checksums", help="Verify files against a checksum manifest")
    sp.add_argument("--root", default=str(PROJECT_ROOT), help="Project root directory")
    sp.add_argument("--manifest", default="file_checksums.txt", help="Manifest filename")
    sp.set_defaults(func=cmd_verify_checksums)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


