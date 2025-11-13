from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm

try:
    import trueskill
except ImportError as exc:  # pragma: no cover - guidance for users
    trueskill = None
    TRUESKILL_IMPORT_ERROR = exc
else:
    TRUESKILL_IMPORT_ERROR = None

from buckshot_roulette.multiplayer.game import Items, RoundConfig, SequenceConfig

import rl
from rl import MOVE_TYPES, PolicyNet, RLEngine, play_one_round


# --------------------------- Manual Configuration --------------------------- #

# Edit this list to pin exact architectures you want to explore.
MANUAL_ARCHITECTURES: List[Dict[str, Any]] = [
    {"hidden_layers": 2, "hidden_dim": 128, "skip_connection": True, "tag": "baseline"},
    {"hidden_layers": 3, "hidden_dim": 192, "skip_connection": False, "tag": "deep_skip"},
]

# Default kwargs that will be forwarded into RoundConfig(**kwargs).
# Provide a JSON file via --round-config to override without editing this file.
DEFAULT_ROUND_CONFIG: Dict[str, Any] = {
    # Example:
    # "start_charges": 4,
    # "enabled_items": {"saw": 1, "magnifying_glass": 1, "jammer": 1, "cigarettes": 1,
    #                   "beer": 1, "burner_phone": 1, "adrenaline": 1, "inverter": 1, "remote": 1},
}

# ------------------------------ Data Structures ----------------------------- #


@dataclass(frozen=True)
class ArchitectureSpec:
    hidden_layers: int
    hidden_dim: int
    skip_connection: bool = False
    tag: Optional[str] = None

    def descriptor(self) -> str:
        base = f"{self.hidden_layers}x{self.hidden_dim}"
        if self.skip_connection:
            base += "_skip"
        if self.tag:
            base = f"{self.tag}:{base}"
        return base


@dataclass(eq=False)
class ModelCheckpoint:
    path: Path
    spec: ArchitectureSpec
    opponent: str
    steps: int
    param_count: int

    @property
    def name(self) -> str:
        return self.path.stem


SKIP_PATCH_NOTIFIED: Set[Path] = set()


# ----------------------------- Helper Functions ----------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate RL ensembles for Buckshot Roulette.")
    parser.add_argument("--seed", type=int, default=1337, help="random seed for architecture sampling and evaluation order")
    parser.add_argument("--players", type=int, default=4, choices=[-1, 2, 3, 4], help="number of players in a round")
    parser.add_argument("--model-root", type=Path, default=Path("models"), help="directory holding checkpoint subfolders")
    parser.add_argument("--random-episodes", type=int, default=100_000, help="games to train vs Random engine")
    parser.add_argument("--mirror-episodes", type=int, default=900_000, help="games to train vs mirror self-play")
    parser.add_argument("--mirror-mode", type=str, default="frozen", choices=["shared", "frozen"], help="mirror opponent variant to use")
    parser.add_argument("--target-update", type=int, default=200, help="frozen mirror sync interval")
    parser.add_argument("--log-every", type=int, default=20_000, help="logging cadence during training")
    parser.add_argument("--lr", type=float, default=3e-4, help="optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--entropy", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--auto-architectures", type=int, default=4, help="how many random architectures to sample on top of MANUAL_ARCHITECTURES")
    parser.add_argument("--round-config", type=str, default=None, help="path to JSON (or inline JSON string) with RoundConfig kwargs")
    parser.add_argument("--output-csv", type=Path, default=Path("models/ensemble_results.csv"), help="where to write evaluation summary CSV")
    parser.add_argument("--plot-path", type=Path, default=Path("models/ensemble_param_vs_elo.png"), help="path for param-count-vs-ELO plot")
    parser.add_argument("--eval-games-per-match", type=int, default=7, help="rounds per evaluation matchup")
    parser.add_argument("--eval-iterations", type=int, default=15, help="how many reshuffled passes to run during evaluation")
    parser.add_argument("--manual-only", action="store_true", help="disable automatic architecture sampling")
    parser.add_argument("--eval-scope", type=str, nargs="*", default=None,
                        help="limit evaluation scopes (subfolder/global); default runs both")
    parser.add_argument("--max-workers", type=int, default=16, help="maximum concurrent workers for async tasks")
    return parser.parse_args()


def ensure_trueskill_available():
    if trueskill is None:
        raise ImportError(
            "The 'trueskill' package is required for rl_ensemble.py. "
            "Install it with `pip install trueskill`."
        ) from TRUESKILL_IMPORT_ERROR


def load_round_config_overrides(value: Optional[str]) -> Dict[str, Any]:
    if value is None:
        return dict(DEFAULT_ROUND_CONFIG)
    candidate = Path(value)
    if candidate.exists():
        data = json.loads(candidate.read_text())
    else:
        data = json.loads(value)
    merged = dict(DEFAULT_ROUND_CONFIG)
    merged.update(data)
    return merged


def convert_round_kwargs(raw_kwargs: Dict[str, Any], player_count: int) -> Dict[str, Any]:
    """Convert JSON-friendly values into RoundConfig-friendly objects."""
    kwargs = dict(raw_kwargs)
    seq_config = kwargs.pop("sequences", None)
    if seq_config is not None:
        sequences: List[Optional[SequenceConfig]] = []
        for entry in seq_config:
            if entry is None:
                sequences.append(None)
                continue
            counts = entry.get("counts")
            counts_tuple = tuple(counts) if counts is not None else None
            sequences.append(SequenceConfig(counts=counts_tuple, item_count=entry.get("item_count"),
                                            player_count=player_count))
        kwargs["sequences"] = sequences
    for item_field in ("item_caps", "global_item_caps", "enabled_items"):
        if item_field in kwargs:
            kwargs[item_field] = Items(**kwargs[item_field])
    return kwargs


def build_round_config(player_count: int, raw_kwargs: Dict[str, Any]) -> RoundConfig:
    kwargs = convert_round_kwargs(raw_kwargs, player_count)
    return RoundConfig(player_count=player_count, **kwargs)


@contextmanager
def override_round_factory(raw_kwargs: Dict[str, Any], player_count: int):
    """Temporarily override rl.make_default_round to respect custom kwargs."""
    if not raw_kwargs:
        yield
        return
    original = rl.make_default_round

    def _custom(player_count_param: int) -> RoundConfig:
        return build_round_config(player_count_param, raw_kwargs)

    rl.make_default_round = _custom
    try:
        yield
    finally:
        rl.make_default_round = original


def generate_architectures(rng: random.Random, auto_count: int, manual_only: bool) -> List[ArchitectureSpec]:
    specs = [ArchitectureSpec(**entry) for entry in MANUAL_ARCHITECTURES]
    if manual_only:
        return specs
    layer_choices = [2, 3, 4, 5]
    width_choices = [96, 128, 160, 192, 224, 256]
    sampled: set[ArchitectureSpec] = set(specs)
    while len(specs) < len(MANUAL_ARCHITECTURES) + max(0, auto_count):
        spec = ArchitectureSpec(
            hidden_layers=rng.choice(layer_choices),
            hidden_dim=rng.choice(width_choices),
            skip_connection=rng.random() < 0.35,
            tag=f"auto_{len(specs)}",
        )
        if spec not in sampled:
            specs.append(spec)
            sampled.add(spec)
    return specs


def _spawn_rng_seed(rng: random.Random) -> int:
    """Return an independent seed derived from the shared RNG."""
    return rng.randrange(0, 2**63)


def infer_dims(players: int) -> Tuple[int, int, int]:
    max_players = players if players > 0 else 4
    state_dim = 97
    move_dim = len(MOVE_TYPES) + (max_players + 1) + 2
    return max_players, state_dim, move_dim


def count_parameters(players: int, spec: ArchitectureSpec) -> int:
    _, state_dim, move_dim = infer_dims(players)
    policy = PolicyNet(
        state_dim=state_dim,
        move_dim=move_dim,
        hidden_dim=spec.hidden_dim,
        hidden_layers=spec.hidden_layers,
        skip_connection=spec.skip_connection,
    )
    return sum(p.numel() for p in policy.parameters())


def checkpoint_dir(root: Path, spec: ArchitectureSpec, opponent: str) -> Path:
    return root / f"model_{spec.hidden_layers}_{spec.hidden_dim}_{opponent}"


def snapshot_checkpoints(folder: Path) -> set[str]:
    if not folder.exists():
        return set()
    return {p.name for p in folder.glob("model_*.pt")}


def parse_training_steps(filename: str) -> int:
    try:
        return int(Path(filename).stem.split("_")[-1])
    except (ValueError, IndexError):
        return -1


def latest_checkpoint_info(folder: Path) -> Tuple[Optional[Path], int]:
    if not folder.exists():
        return None, 0
    best_step = -1
    best_paths: List[Path] = []
    for path in sorted(folder.glob("model_*.pt")):
        step = parse_training_steps(path.name)
        if step < 0:
            continue
        if step > best_step:
            best_step = step
            best_paths = [path]
        elif step == best_step:
            best_paths.append(path)
    if best_step < 0 or not best_paths:
        return None, 0
    # Deterministically prefer the lexicographically last path for stability.
    return best_paths[-1], best_step


def ensure_metadata(folder: Path, spec: ArchitectureSpec, players: int) -> Dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "metadata.json"
    existing: Dict[str, Any] = {}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
    merged = {
        "hidden_layers": spec.hidden_layers,
        "hidden_dim": spec.hidden_dim,
        "skip_connection": spec.skip_connection,
        "tag": spec.tag,
        "players": players,
        "param_count": count_parameters(players, spec),
    }
    existing.update({k: v for k, v in merged.items() if v is not None})
    meta_path.write_text(json.dumps(existing, indent=2))
    return existing


def train_stage(
    spec: ArchitectureSpec,
    *,
    opponent: str,
    episodes: int,
    load_path: Optional[Path],
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
) -> List[ModelCheckpoint]:
    if episodes <= 0:
        return []
    folder = checkpoint_dir(args.model_root, spec, opponent)
    ensure_metadata(folder, spec, args.players)
    existing_checkpoint, completed_episodes = latest_checkpoint_info(folder)
    remaining = max(0, episodes - completed_episodes)
    before = snapshot_checkpoints(folder)

    records: List[ModelCheckpoint] = []
    new_file_entries: List[Tuple[str, int]] = []

    if remaining <= 0:
        print(
            f"[TRAIN] {spec.descriptor()} vs {opponent} already reached "
            f"{completed_episodes}/{episodes} episodes; skipping."
        )
        after = before
    else:
        resume_from = existing_checkpoint or load_path
        resume_note = ""
        if existing_checkpoint:
            resume_note = " (resuming from latest checkpoint)"
        elif load_path:
            resume_note = " (warm start)"
        print(
            f"[TRAIN] {spec.descriptor()} vs {opponent}: {completed_episodes}/{episodes} complete."
            f" Training {remaining} more episodes{resume_note}."
        )
        with override_round_factory(round_kwargs, args.players):
            rl.train(
                episodes=remaining,
                players=args.players,
                opponent_type=opponent,
                mirror_mode=args.mirror_mode,
                target_update=args.target_update,
                log_every=args.log_every,
                lr=args.lr,
                gamma=args.gamma,
                entropy=args.entropy,
                hidden_dim=spec.hidden_dim,
                hidden_layers=spec.hidden_layers,
                skip_connection=spec.skip_connection,
                model_root=args.model_root,
                load_path=str(resume_from) if resume_from else None,
            )
        after = snapshot_checkpoints(folder)
        new_files = sorted(after - before)
        for name in new_files:
            rel_step = parse_training_steps(name)
            if rel_step < 0:
                rel_step = remaining
            absolute_step = completed_episodes + rel_step
            new_name = f"model_{spec.hidden_layers}_{spec.hidden_dim}_{opponent}_{absolute_step}.pt"
            src = folder / name
            dst = folder / new_name
            if dst != src:
                src.rename(dst)
            new_file_entries.append((new_name, absolute_step))
    metadata = json.loads((folder / "metadata.json").read_text())
    for name, steps in new_file_entries:
        path = folder / name
        record = ModelCheckpoint(
            path=path,
            spec=spec,
            opponent=opponent,
            steps=steps,
            param_count=metadata.get("param_count", count_parameters(args.players, spec)),
        )
        records.append(record)
    if not records:
        # Fallback: no new checkpoints (likely rerunning). Still include latest.
        for path in sorted(folder.glob("model_*.pt")):
            records.append(
                ModelCheckpoint(
                    path=path,
                    spec=spec,
                    opponent=opponent,
                    steps=parse_training_steps(path.name),
                    param_count=metadata.get("param_count", count_parameters(args.players, spec)),
                )
            )
        print(f"[WARN] No new checkpoints detected for {spec.descriptor()} vs {opponent}; using existing files.")
    return records


def latest_checkpoint_path(folder: Path, rng: random.Random) -> Optional[Path]:
    checkpoint, _ = latest_checkpoint_info(folder)
    return checkpoint


def train_architecture(
    spec: ArchitectureSpec,
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
    rng: random.Random,
) -> List[ModelCheckpoint]:
    records = []
    records.extend(
        train_stage(
            spec,
            opponent="random",
            episodes=args.random_episodes,
            load_path=None,
            args=args,
            round_kwargs=round_kwargs,
        )
    )
    random_folder = checkpoint_dir(args.model_root, spec, "random")
    warm_start = latest_checkpoint_path(random_folder, rng)
    records.extend(
        train_stage(
            spec,
            opponent="mirror",
            episodes=args.mirror_episodes,
            load_path=warm_start,
            args=args,
            round_kwargs=round_kwargs,
        )
    )
    return records


def _train_architecture_entry(
    spec: ArchitectureSpec,
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
    seed: int,
) -> List[ModelCheckpoint]:
    local_rng = random.Random(seed)
    return train_architecture(spec, args, round_kwargs, local_rng)


async def train_specs_async(
    specs: List[ArchitectureSpec],
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
    rng: random.Random,
) -> None:
    if not specs:
        return
    worker_count = max(1, min(args.max_workers, len(specs)))
    seeds = [_spawn_rng_seed(rng) for _ in specs]
    loop = asyncio.get_running_loop()

    if worker_count == 1:
        with tqdm(total=len(specs), desc="Spec Training...") as progress:
            for spec, seed in zip(specs, seeds):
                await loop.run_in_executor(None, _train_architecture_entry, spec, args, round_kwargs, seed)
                progress.update(1)
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        tasks = [
            loop.run_in_executor(executor, _train_architecture_entry, spec, args, round_kwargs, seed)
            for spec, seed in zip(specs, seeds)
        ]
        with tqdm(total=len(tasks), desc="Spec Training...") as progress:
            for future in asyncio.as_completed(tasks):
                await future
                progress.update(1)


def build_policy(spec: ArchitectureSpec, players: int) -> PolicyNet:
    _, state_dim, move_dim = infer_dims(players)
    policy = PolicyNet(
        state_dim=state_dim,
        move_dim=move_dim,
        hidden_dim=spec.hidden_dim,
        hidden_layers=spec.hidden_layers,
        skip_connection=spec.skip_connection,
    ).to("cpu")
    policy.eval()
    return policy


def load_policy_state(record: ModelCheckpoint) -> Dict[str, torch.Tensor]:
    return torch.load(record.path, map_location="cpu")


def state_dict_has_skip_layers(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(key.startswith(("state_merge", "move_merge")) for key in state_dict)


def build_policy_from_state_dict(
    record: ModelCheckpoint,
    player_count: int,
    state_dict: Dict[str, torch.Tensor],
) -> PolicyNet:
    policy = build_policy(record.spec, player_count)
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError as exc:
        policy = _resolve_state_dict_mismatch(record, player_count, state_dict, exc)
    return policy


def _resolve_state_dict_mismatch(
    record: ModelCheckpoint,
    player_count: int,
    state_dict: Dict[str, torch.Tensor],
    error: RuntimeError,
) -> PolicyNet:
    has_skip_weights = state_dict_has_skip_layers(state_dict)

    def _patch_spec(skip_connection: bool) -> ArchitectureSpec:
        return ArchitectureSpec(
            hidden_layers=record.spec.hidden_layers,
            hidden_dim=record.spec.hidden_dim,
            skip_connection=skip_connection,
            tag=record.spec.tag,
        )

    if record.spec.skip_connection and not has_skip_weights:
        patched_spec = _patch_spec(False)
        policy = build_policy(patched_spec, player_count)
        policy.load_state_dict(state_dict)
        record.spec = patched_spec
        record.param_count = count_parameters(player_count, patched_spec)
        folder = record.path.parent
        if folder not in SKIP_PATCH_NOTIFIED:
            print(
                f"[WARN] {record.path} metadata expects skip connections but checkpoint lacks "
                "skip weights; evaluating with skip_connection=False."
            )
            SKIP_PATCH_NOTIFIED.add(folder)
        return policy

    if not record.spec.skip_connection and has_skip_weights:
        patched_spec = _patch_spec(True)
        policy = build_policy(patched_spec, player_count)
        try:
            policy.load_state_dict(state_dict)
        except RuntimeError:
            pass
        else:
            record.spec = patched_spec
            record.param_count = count_parameters(player_count, patched_spec)
            folder = record.path.parent
            if folder not in SKIP_PATCH_NOTIFIED:
                print(
                    f"[WARN] {record.path} checkpoint includes skip weights; evaluating with "
                    "skip_connection=True."
                )
                SKIP_PATCH_NOTIFIED.add(folder)
            return policy

    raise error


def simulate_match(
    participants: Sequence[ModelCheckpoint],
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
    state_cache: Dict[Path, Dict[str, torch.Tensor]],
) -> List[int]:
    """Return ranks for the participants (0 is best) based on wins across games."""
    group_size = len(participants)
    wins = Counter({idx: 0 for idx in range(group_size)})
    player_count = args.players if args.players > 0 else 4

    for _ in range(args.eval_games_per_match):
        engines: List[RLEngine] = []
        for seat, record in enumerate(participants):
            state_dict = state_cache.setdefault(record.path, load_policy_state(record))
            policy = build_policy_from_state_dict(record, player_count, state_dict)
            optimizer = optim.Adam(policy.parameters(), lr=1e-5)
            engines.append(
                RLEngine(
                    playing_as=seat,
                    policy=policy,
                    optimizer=optimizer,
                    gamma=args.gamma,
                    entropy_coef=args.entropy,
                    train_mode=False,
                )
            )
        cfg = build_round_config(player_count, round_kwargs)
        winner = play_one_round(list(engines), trainer_idx=0, cfg=cfg)
        wins[winner] += 1

    win_values = [wins[idx] for idx in range(group_size)]
    unique_wins = sorted(set(win_values), reverse=True)
    ranks = [unique_wins.index(value) for value in win_values]
    return ranks


def run_trueskill_matches(
    records: List[ModelCheckpoint],
    *,
    scope: str,
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
    rng: random.Random,
) -> Dict[ModelCheckpoint, Tuple[trueskill.Rating, int]]:
    ensure_trueskill_available()
    group_size = args.players if args.players > 0 else 4
    if len(records) < group_size:
        return {}
    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings = {record: env.create_rating() for record in records}
    participation = Counter()
    state_cache: Dict[Path, Dict[str, torch.Tensor]] = {}

    for iteration in range(args.eval_iterations):
        order = list(records)
        rng.shuffle(order)
        for anchor in order:
            pool = [r for r in order if r is not anchor]
            if len(pool) < group_size - 1:
                continue
            opponents = rng.sample(pool, group_size - 1)
            participants = [anchor] + opponents
            ranks = simulate_match(participants, args, round_kwargs, state_cache)
            rating_groups = [(ratings[p],) for p in participants]
            new_ratings = env.rate(rating_groups, ranks=ranks)
            for participant, new_rating in zip(participants, new_ratings):
                ratings[participant] = new_rating[0]
                participation[participant] += 1
        print(f"[EVAL:{scope}] iteration {iteration + 1}/{args.eval_iterations} complete.")

    return {record: (ratings[record], participation[record]) for record in records}


def _run_trueskill_entry(
    records: List[ModelCheckpoint],
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
    seed: int,
    scope: str,
) -> Dict[ModelCheckpoint, Tuple[trueskill.Rating, int]]:
    local_rng = random.Random(seed)
    return run_trueskill_matches(records, scope=scope, args=args, round_kwargs=round_kwargs, rng=local_rng)


def gather_checkpoints(model_root: Path, default_players: int) -> List[ModelCheckpoint]:
    records: List[ModelCheckpoint] = []
    for folder in sorted(model_root.glob("model_*_*_*")):
        if not folder.is_dir():
            continue
        _, layers_str, hidden_str, opponent = folder.name.split("_", 3)
        spec = ArchitectureSpec(hidden_layers=int(layers_str), hidden_dim=int(hidden_str), skip_connection=False)
        meta_path = folder / "metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            spec = ArchitectureSpec(
                hidden_layers=int(metadata.get("hidden_layers", spec.hidden_layers)),
                hidden_dim=int(metadata.get("hidden_dim", spec.hidden_dim)),
                skip_connection=bool(metadata.get("skip_connection", spec.skip_connection)),
                tag=metadata.get("tag"),
            )
        param_count = metadata.get("param_count")
        if param_count is None:
            param_count = count_parameters(default_players, spec)
        for ckpt in folder.glob("model_*.pt"):
            records.append(
                ModelCheckpoint(
                    path=ckpt,
                    spec=spec,
                    opponent=opponent,
                    steps=parse_training_steps(ckpt.name),
                    param_count=param_count,
                )
            )
    return records


async def evaluation_pipeline(
    records: List[ModelCheckpoint],
    args: argparse.Namespace,
    round_kwargs: Dict[str, Any],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    scopes = args.eval_scope or ["subfolder", "global"]
    task_specs: List[Tuple[str, str, List[ModelCheckpoint], str]] = []

    if "subfolder" in scopes:
        grouped: Dict[str, List[ModelCheckpoint]] = defaultdict(list)
        for record in records:
            grouped[record.path.parent.name].append(record)
        for folder_name, subset in grouped.items():
            task_specs.append(("subfolder", folder_name, subset, f"folder:{folder_name}"))

    if "global" in scopes and records:
        task_specs.append(("global", "all", records, "global"))

    if not task_specs:
        return results

    worker_count = max(1, min(args.max_workers, len(task_specs)))
    loop = asyncio.get_running_loop()
    executor: Optional[ThreadPoolExecutor] = None
    if len(task_specs) > 1:
        executor = ThreadPoolExecutor(max_workers=worker_count)

    try:
        futures = []
        for scope_name, group_name, subset, scope_label in task_specs:
            seed = _spawn_rng_seed(rng)
            future = loop.run_in_executor(
                executor,
                _run_trueskill_entry,
                subset,
                args,
                round_kwargs,
                seed,
                scope_label,
            )
            futures.append((scope_name, group_name, future))

        for scope_name, group_name, future in futures:
            eval_out = await future
            for record, (rating, matches) in eval_out.items():
                results.append(
                    build_result_row(record, rating, matches, scope=scope_name, group=group_name)
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return results


def build_result_row(
    record: ModelCheckpoint,
    rating: trueskill.Rating,
    matches: int,
    *,
    scope: str,
    group: str,
) -> Dict[str, Any]:
    return {
        "model_name": record.name,
        "path": str(record.path),
        "hidden_layers": record.spec.hidden_layers,
        "hidden_dim": record.spec.hidden_dim,
        "skip_connection": record.spec.skip_connection,
        "tag": record.spec.tag,
        "opponent": record.opponent,
        "training_steps": record.steps,
        "parameter_count": record.param_count,
        "rating_mu": rating.mu,
        "rating_sigma": rating.sigma,
        "scope": scope,
        "group": group,
        "matches": matches,
    }


def write_results_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[RESULTS] Wrote {len(rows)} rows to {path}")


def plot_param_vs_elo(rows: List[Dict[str, Any]], plot_path: Path) -> None:
    global_rows = [row for row in rows if row["scope"] == "global"]
    if not global_rows:
        print("[PLOT] Skipping plot because no global ratings were produced.")
        return
    xs = [row["parameter_count"] for row in global_rows]
    ys = [row["rating_mu"] for row in global_rows]
    colors = ["tab:blue" if row["opponent"] == "mirror" else "tab:orange" for row in global_rows]
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, c=colors, alpha=0.8)
    plt.xlabel("Parameter count")
    plt.ylabel("TrueSkill μ (higher is better)")
    plt.title("Model size vs performance (global TrueSkill)")
    plt.grid(True, alpha=0.3)
    for row in global_rows:
        plt.annotate(row["model_name"], (row["parameter_count"], row["rating_mu"]), fontsize=7, alpha=0.7)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT] Saved parameter-count plot to {plot_path}")


async def async_main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    round_kwargs = load_round_config_overrides(args.round_config)
    specs = generate_architectures(rng, args.auto_architectures, args.manual_only)
    print(f"[SETUP] Training {len(specs)} architectures with seed {args.seed}")

    await train_specs_async(specs, args, round_kwargs, rng)

    checkpoints = gather_checkpoints(args.model_root, args.players)
    if not checkpoints:
        print("[WARN] No checkpoints found; nothing to evaluate.")
        return

    eval_rows = await evaluation_pipeline(checkpoints, args, round_kwargs, rng)
    write_results_csv(eval_rows, args.output_csv)
    plot_param_vs_elo(eval_rows, args.plot_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
