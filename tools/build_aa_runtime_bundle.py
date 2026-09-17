"""Export/validate a portable private AA8 reader resource bundle; no decoding."""

import argparse
import json
from pathlib import Path

from poker_engine.desktop.aa_bundle import export_bundle, validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--profile", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--profile", type=Path, required=True)
    validate.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    result = export_bundle(args.profile, args.output) if args.command == "export" else (
        validate_bundle(args.profile, args.manifest_sha256))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
