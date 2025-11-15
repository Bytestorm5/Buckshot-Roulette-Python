from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence
import sys
import shutil
DEFAULT_ORIGIN_ENDPOINT = "https://brrl-models.nyc3.digitaloceanspaces.com"

ENV_ENDPOINT_KEYS: Sequence[str] = (
    "BRRL_MODELS_ORIGIN",
    "S3_ORIGIN_ENDPOINT",
    "S3_ENDPOINT",
    "SPACES_ORIGIN_ENDPOINT",
)
ENV_ACCESS_KEYS: Sequence[str] = (
    "BRRL_MODELS_ACCESS_KEY",
    "S3_ACCESS_KEY",
    "SPACES_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
)
ENV_SECRET_KEYS: Sequence[str] = (
    "BRRL_MODELS_SECRET_KEY",
    "S3_SECRET_KEY",
    "SPACES_SECRET_KEY",
    "AWS_SECRET_ACCESS_KEY",
)
ENV_BUCKET_KEYS: Sequence[str] = ("BRRL_MODELS_BUCKET", "S3_BUCKET", "SPACES_BUCKET")

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT_DIR / ".env"


@dataclass
class S3Config:
    bucket: str
    origin_endpoint: str
    access_key: str
    secret_key: str
    host_base: str

    def base_arguments(self) -> list[str]:
        return [
            "s3cmd",
            f"--access_key={self.access_key}",
            f"--secret_key={self.secret_key}",
            f"--host={self.host_base}",
            f"--host-bucket=%(bucket)s.{self.host_base}",
        ]

    def remote_uri(self, key: Optional[str] = None) -> str:
        key = (key or "").strip("/")
        return f"s3://{self.bucket}/{key}" if key else f"s3://{self.bucket}"


_ENV_CACHE: Optional[Dict[str, str]] = None
_S3_CONFIG: Optional[S3Config] = None
_WARNED_MISSING_CONFIG = False


def _load_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _env_values() -> Dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _load_env_file(DEFAULT_ENV_PATH)
    return _ENV_CACHE


def _lookup_env(keys: Sequence[str]) -> Optional[str]:
    env_map = _env_values()
    for key in keys:
        value = env_map.get(key)
        if value:
            return value
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _parse_origin(origin: str) -> tuple[Optional[str], str]:
    trimmed = origin.strip()
    if "://" in trimmed:
        trimmed = trimmed.split("://", 1)[1]
    trimmed = trimmed.strip("/")
    if not trimmed:
        return None, ""
    if "." not in trimmed:
        return None, trimmed
    bucket_candidate, host_base = trimmed.split(".", 1)
    return (bucket_candidate or None), host_base


def get_s3_config(*, optional: bool = True, quiet: bool = False) -> Optional[S3Config]:
    global _S3_CONFIG, _WARNED_MISSING_CONFIG
    if _S3_CONFIG is not None:
        return _S3_CONFIG

    origin = _lookup_env(ENV_ENDPOINT_KEYS) or DEFAULT_ORIGIN_ENDPOINT
    access = _lookup_env(ENV_ACCESS_KEYS)
    secret = _lookup_env(ENV_SECRET_KEYS)
    bucket = _lookup_env(ENV_BUCKET_KEYS)
    bucket_guess, host_base = _parse_origin(origin)

    if not bucket:
        bucket = bucket_guess
    if not host_base:
        # Fall back to the default DigitalOcean Spaces host if we only had a bucket name.
        _, host_base = _parse_origin(DEFAULT_ORIGIN_ENDPOINT)

    missing: list[str] = []
    if not access:
        missing.append("access key")
    if not secret:
        missing.append("secret key")
    if not bucket:
        missing.append("bucket name or origin endpoint")
    if not host_base:
        missing.append("host base")

    if missing:
        if optional:
            if not quiet and not _WARNED_MISSING_CONFIG:
                print(
                    "[S3] Missing credentials in .env (expected keys like "
                    "S3_ORIGIN_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY); skipping S3 integration."
                )
                _WARNED_MISSING_CONFIG = True
            return None
        raise RuntimeError(f"S3 configuration incomplete; please set {_describe_missing(missing)} in .env.")

    _S3_CONFIG = S3Config(
        bucket=bucket,
        origin_endpoint=origin,
        access_key=access,
        secret_key=secret,
        host_base=host_base,
    )
    return _S3_CONFIG


def _describe_missing(parts: Sequence[str]) -> str:
    if not parts:
        return "the required keys"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def require_s3_config(quiet: bool = False) -> S3Config:
    config = get_s3_config(optional=False, quiet=quiet)
    if config is None:  # pragma: no cover - defensive
        raise RuntimeError("S3 configuration is required but missing.")
    return config


def _format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_s3cmd(args: Sequence[str], config: S3Config, *, quiet: bool = False) -> None:
    """
    Run s3cmd with the given arguments and configuration.

    We try a few strategies so this works both when:
      - `s3cmd` is on PATH (e.g. global install), and
      - `s3cmd` is installed via pip in the current venv (as a script in venv/Scripts).
    """
    base = config.base_arguments()  # e.g. ["s3cmd", "--access_key=...", ...]
    s3cmd_name = base[0]
    s3cmd_exe: Optional[str] = shutil.which(s3cmd_name)

    cmd: list[str]

    if s3cmd_exe:
        # Normal case: s3cmd is an executable on PATH
        cmd = [s3cmd_exe] + base[1:] + list(args)
    else:
        # Fallback: look for an s3cmd script next to the current Python executable
        scripts_dir = Path(sys.executable).parent  # typically .../venv/Scripts on Windows
        candidates = [
            scripts_dir / "s3cmd.exe",
            scripts_dir / "s3cmd.py",
            scripts_dir / "s3cmd",
        ]
        script = next((c for c in candidates if c.exists()), None)

        if script is not None:
            # Invoke the script via the same Python interpreter that runs this code
            cmd = [sys.executable, str(script)] + base[1:] + list(args)
        else:
            # Last resort: keep original behaviour and let FileNotFoundError surface
            cmd = base + list(args)

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - external dependency
        raise RuntimeError(
            "s3cmd executable not found. Make sure 'pip install s3cmd' was run in this "
            "environment or install a system-wide s3cmd."
        ) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        raise RuntimeError(
            f"s3cmd command failed (exit {exc.returncode}): {_format_command(cmd)}"
        ) from exc


def _normalize_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _relative_key(path: Path, base: Optional[Path]) -> Optional[str]:
    if base is None:
        return None
    target = _normalize_path(path)
    root = _normalize_path(base)
    try:
        relative = target.relative_to(root)
    except ValueError:
        return None
    rel_str = relative.as_posix().lstrip("./")
    return rel_str or None


def upload_file_to_s3(path: Path, *, relative_to: Optional[Path], quiet: bool = False) -> bool:
    config = get_s3_config(optional=True, quiet=quiet)
    if config is None:
        return False
    key = _relative_key(path, relative_to) or Path(path).name
    remote_uri = config.remote_uri(key)
    _run_s3cmd(["put", str(path), remote_uri], config, quiet=quiet)
    if not quiet:
        print(f"[S3] Uploaded {path} -> {remote_uri}")
    return True


def ensure_local_file(path: Path, *, relative_to: Optional[Path], quiet: bool = False) -> bool:
    if Path(path).exists():
        return False
    return download_file_from_s3(path, relative_to=relative_to, quiet=quiet)


def download_file_from_s3(
    path: Path,
    *,
    relative_to: Optional[Path],
    quiet: bool = False,
    force: bool = False,
) -> bool:
    config = get_s3_config(optional=True, quiet=quiet)
    if config is None:
        return False
    key = _relative_key(path, relative_to)
    if key is None:
        if not quiet:
            print(f"[S3] Cannot download {path}; it is outside the synced root.")
        return False
    remote_uri = config.remote_uri(key)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["get", remote_uri, str(dest)]
    if force:
        args.insert(0, "--force")
    _run_s3cmd(args, config, quiet=quiet)
    if not quiet:
        print(f"[S3] Downloaded {remote_uri} -> {dest}")
    return True


def sync_prefix_to_path(
    local_path: Path,
    prefix: str = "",
    *,
    quiet: bool = False,
    require: bool = False,
) -> bool:
    config = get_s3_config(optional=not require, quiet=quiet)
    if config is None:
        return False
    trimmed = prefix.strip().strip("/")
    remote = config.remote_uri(trimmed)
    if not remote.endswith("/"):
        remote = f"{remote}/"
    destination = Path(local_path)
    destination.mkdir(parents=True, exist_ok=True)
    _run_s3cmd(["sync", remote, str(destination)], config, quiet=quiet)
    if not quiet:
        pretty = trimmed or "."
        print(f"[S3] Synced {remote} -> {destination} (prefix '{pretty}')")
    return True


def sync_folder_from_s3(folder: Path, *, relative_to: Optional[Path], quiet: bool = False) -> bool:
    prefix = _relative_key(folder, relative_to)
    if prefix is None:
        return False
    return sync_prefix_to_path(folder, prefix, quiet=quiet)


def sync_bucket_to_path(local_path: Path, *, quiet: bool = False, require: bool = False) -> bool:
    return sync_prefix_to_path(local_path, prefix="", quiet=quiet, require=require)
