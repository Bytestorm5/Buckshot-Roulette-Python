from __future__ import annotations

import argparse
from pathlib import Path

from s3cmd_utils import sync_bucket_to_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all model artifacts from the remote bucket.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Local directory to sync the bucket into (default: ./models).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose s3cmd output while syncing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.output.expanduser().resolve()
    try:
        sync_bucket_to_path(destination, quiet=args.quiet, require=True)
    except RuntimeError as exc:
        print(f"[S3] Failed to download bucket: {exc}")
        raise SystemExit(1) from exc
    print(f"[S3] Downloaded bucket contents into {destination}")


if __name__ == "__main__":
    main()
