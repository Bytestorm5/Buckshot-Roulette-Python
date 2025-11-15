from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import string
from pathlib import Path

from s3cmd_utils import upload_file_to_s3


def _random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small file inside a folder and upload it via s3cmd to verify write access."
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models"),
        help="Local root that mirrors the bucket structure (default: ./models).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="s3_write_test",
        help="Folder (relative to --model-root) where the test file will be created.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the generated file on disk after uploading (default deletes it).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.model_root.expanduser().resolve()
    target_folder = root / args.folder
    target_folder.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"write_test_{timestamp}_{_random_suffix()}.txt"
    file_path = target_folder / filename
    content = [
        "Buckshot Roulette RL write test",
        f"UTC timestamp: {timestamp}",
        f"Host: {os.uname().nodename if hasattr(os, 'uname') else os.getenv('COMPUTERNAME', 'unknown')}",
        f"Folder: {args.folder}",
    ]
    file_path.write_text("\n".join(content))
    print(f"[S3-TEST] Created {file_path}")

    success = upload_file_to_s3(file_path, relative_to=root, quiet=False)
    if success:
        print(f"[S3-TEST] Upload succeeded. Key mirrors {args.folder}/{filename} under the bucket.")
    else:
        print("[S3-TEST] Upload skipped (missing credentials?).")

    if not args.keep:
        try:
            file_path.unlink(missing_ok=True)
            if not any(target_folder.iterdir()):
                target_folder.rmdir()
            print(f"[S3-TEST] Cleaned up {file_path}")
        except OSError as exc:
            print(f"[S3-TEST] Warning: failed to delete local test file ({exc}).")


if __name__ == "__main__":
    main()
