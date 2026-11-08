#!/usr/bin/env python3
"""Build and verify a local review bundle; external robot assets stay external."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from omnicalib import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/omnicalib_v030_reproduction.zip")
    )
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    parser.add_argument("--include", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = set()

    def add(path):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Bundle inputs must be within the reproduction repository")
        if (
            path.is_file()
            and path != args.output.resolve()
            and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".pyo")
        ):
            files.add(path)

    for name in ("README.md", ".gitignore", "pyproject.toml", "requirements-lock.txt"):
        add(root / name)
    for name in ("omnicalib", "configs", "scripts", "tests", "docs"):
        for path in (root / name).rglob("*"):
            add(path)
    for path in args.results:
        if not path.is_dir():
            raise ValueError(f"Missing results directory: {path}")
        for candidate in path.rglob("*"):
            add(candidate)
    for path in args.include:
        if not path.is_file():
            raise ValueError(f"Missing artifact: {path}")
        add(path)
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    manifest = {
        "version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "role": "Local independent algorithm/synthetic reproduction; not hardware validation",
        "external_robot_assets_bundled": False,
        "paper_pdf_bundled": False,
        "note": "Install dependencies from requirements-lock.txt. Supply the original robot assets separately; use --asset-root or the documented sibling directory.",
        "sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=args.output.parent, suffix=".zip", delete=False
        ) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            for path in sorted(files):
                bundle.write(path, "OmniCalibration/" + str(path.relative_to(root)))
            bundle.writestr(
                "OmniCalibration/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n"
            )
        with zipfile.ZipFile(temporary) as bundle:
            assert bundle.testzip() is None
            for name, digest in hashes.items():
                assert hashlib.sha256(bundle.read("OmniCalibration/" + name)).hexdigest() == digest
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{digest}  {args.output.name}\n")
    print(
        f"Verified {len(files)} files; {args.output.stat().st_size / 1024**2:.2f} MiB; {args.output}"
    )


if __name__ == "__main__":
    main()
