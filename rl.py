from __future__ import annotations
import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import copy

from tqdm import trange

# Use the import paths the user specified in their project.
from buckshot_roulette.multiplayer.game import (
    BuckshotRoulette,
    BuckshotGame,
    GameStatus,
    RoundConfig,
    GameConfig,
)
from buckshot_roulette.multiplayer.ai import AbstractEngine, Dealer, Random as RandomEngine
from rl_metrics_overlay import SeatWRTracker, LogitDriftProbe, ValueLossEMA, format_seat_wr

seat_wr = SeatWRTracker(num_seats=4)
logit_probe = LogitDriftProbe(buffer_size=2048, device="cpu")  # or "cuda" if you prefer
value_ema = ValueLossEMA(beta=0.98)

prev_policy = None  # will hold a frozen copy to compare against

# ----------------------------- Feature Encoding ----------------------------- #

ITEMS = [
    "saw",
    "magnifying_glass",
    "jammer",
    "cigarettes",
    "beer",
    "burner_phone",
    "adrenaline",
    "inverter",
    "remote",
]

MOVE_TYPES = [
    "shoot",           # parameterized by offset (0..N-1, offset from current turn)
    "saw",
    "magnifying_glass",
    "jammer",          # parameterized by target offset (1..N-1 for jam)
    "cigarettes",
    "beer",
    "burner_phone",
    "adrenaline",
    "inverter",
    "remote",
]


def _one_hot(idx: int, size: int) -> List[float]:
    v = [0.0] * size
    if 0 <= idx < size:
        v[idx] = 1.0
    return v

def encode_state(game: BuckshotRoulette, me: int, known_shells: List[Optional[bool]]) -> torch.Tensor:
    """Encode *public* game state + my own info into a flat feature vector.
    """
    pc = game.player_count

    # Charges (normalized to [0,1] by start_charges), plus alive flag.
    start_charges = float(game.config.start_charges)
    charges = []
    alive = []
    item_feats = []
    # Order by *offset* relative to me for translational invariance.
    for off in range(pc):
        idx = (me + off) % pc
        c = game.charges[idx]
        charges.append(c / max(1.0, start_charges))
        alive.append(1.0 if c > 0 else 0.0)
        
        op_items = game.items[idx]
        for key in ITEMS:
            item_feats.append(min(8, getattr(op_items, key)) / 8.0)

    # Shell counts & probabilities.
    total = float(max(1, game.total))
    live = float(game.live)
    p_live = live / total
    shell_feats = [live / 8.0, total / 8.0, p_live]
    only_one = 1.0 if game.total == 1 else 0.0
    shell_feats.append(only_one)

    # Status flags relative to current turn.
    stat_feats = [0.0] * 7
    if GameStatus.JAMMED_0 in game.statuses: stat_feats[0] = 1.0
    if GameStatus.JAMMED_1 in game.statuses: stat_feats[1] = 1.0
    if GameStatus.JAMMED_2 in game.statuses: stat_feats[2] = 1.0
    if GameStatus.JAMMED_3 in game.statuses: stat_feats[3] = 1.0
    if GameStatus.ADRENALINE_ACTIVE in game.statuses: stat_feats[4] = 1.0
    if GameStatus.INVERTER_UNCERTAINTY in game.statuses: stat_feats[5] = 1.0
    if GameStatus.SAWED_OFF in game.statuses: stat_feats[6] = 1.0

    # Turn direction (remote flips this).
    direction = [1.0 if game.turn_inc > 0 else 0.0, 1.0 if game.turn_inc < 0 else 0.0]

    # Agent memory of the very next shell (optional info from magnifying glass/beer)
    ks = []
    for i in range(10):
        # For each shell:
        # - 0: Is live?
        # - 1: Is blank?
        # - 2: Is unknown?
        # - 3: Exists?
        if i >= len(known_shells):
            # Shell is not in range
            ks.extend([0.0, 0.0, 0.0, 0.0])
            continue
        shell_status = known_shells[i]
        if shell_status is True:
            # Shell is live
            ks.extend([1.0, 0.0, 0.0, 1.0])
            continue
        if shell_status is False:
            # Shell is blank
            ks.extend([0.0, 1.0, 0.0, 1.0])
            continue
        if shell_status is None:
            # Shell is unknown
            ks.extend([0.0, 0.0, 1.0, 1.0])
            continue

    features = (
        charges + alive + item_feats + shell_feats + stat_feats + direction + ks
    )
    return torch.tensor(features, dtype=torch.float32)


def parse_move_tuple(move_tup: Tuple[int, str]) -> Tuple[str, int]:
    """Convert environment move tuple -> (move_type, offset/param)
    - ('shoot_2')  -> ('shoot', 2)
    - ('jammer_1') -> ('jammer', 1)
    - ('saw')      -> ('saw', -1)
    """
    _, move = move_tup
    if move.startswith("shoot_"):
        return "shoot", int(move.split("_")[1])
    if move.startswith("jammer_"):
        return "jammer", int(move.split("_")[1])
    return move, -1


def encode_move(game: BuckshotRoulette, me: int, move_tup: Tuple[int, str]) -> torch.Tensor:
    """Encode a single *candidate* move into features.
    - Type one-hot across MOVE_TYPES
    - Parameter (offset for shoot/jammer) as one-hot among up to 4 players
    - Whether we're in adrenaline steal phase (special action context)
    """
    pc = game.player_count
    move_type, param = parse_move_tuple(move_tup)

    # Type one-hot
    t_hot = _one_hot(MOVE_TYPES.index(move_type), len(MOVE_TYPES))

    # Offset param one-hot (for shoot/jammer); 0..pc-1 offsets
    off_hot = _one_hot(param if 0 <= param < pc else pc, pc + 1)  # +1 for "no param"

    # Is this the forced adrenaline item-theft turn?
    adrenaline_turn = 1.0 if (GameStatus.ADRENALINE_ACTIVE in game.statuses) else 0.0

    # Self-target flag if applicable (e.g., shoot_0)
    self_flag = 0.0
    if move_type == "shoot" and param == 0:
        self_flag = 1.0

    return torch.tensor(t_hot + off_hot + [adrenaline_turn, self_flag], dtype=torch.float32)


# ----------------------------- Policy Network ------------------------------- #


def _build_mlp(input_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    modules: List[nn.Module] = []
    prev = input_dim
    for _ in range(layers):
        modules.append(nn.Linear(prev, hidden_dim))
        modules.append(nn.ReLU())
        prev = hidden_dim
    return nn.Sequential(*modules)


class PolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        move_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.state = _build_mlp(state_dim, hidden_dim, hidden_layers)
        self.move = _build_mlp(move_dim, hidden_dim, hidden_layers)
        if skip_connection:
            self.state_merge = nn.Linear(hidden_dim + state_dim, hidden_dim)
            self.move_merge = nn.Linear(hidden_dim + move_dim, hidden_dim)
        else:
            self.state_merge = None
            self.move_merge = None
        self.val_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, move_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score each candidate move and return (logits, value).
        - state: [S]
        - move_batch: [K, M]
        Returns:
            logits: [K]
            value:  [1]
        """
        logit_probe.maybe_add(state_feats=state, move_feats=move_batch)
        s = self.state(state)               # [H]
        mv = self.move(move_batch)          # [K, H]
        if self.skip_connection:
            assert self.state_merge is not None and self.move_merge is not None
            s = torch.cat([s, state], dim=-1)
            s = self.state_merge(s)
            mv = torch.cat([mv, move_batch], dim=-1)
            mv = self.move_merge(mv)
        # Dot-product scorer between state embedding and each move embedding
        logits = (mv * s).sum(dim=1)        # [K]
        value = self.val_head(s).squeeze(-1)  # []
        value_ema.update(value)      # call this each train/eval step
        return logits, value


# --------------------------- Replay / Trajectories -------------------------- #

@dataclass
class StepRecord:
    state: torch.Tensor
    moves: torch.Tensor      # [K, M]
    act_idx: int
    logp: torch.Tensor
    value: torch.Tensor
    reward: float


class Trajectory:
    def __init__(self):
        self.steps: List[StepRecord] = []

    def add(self, rec: StepRecord):
        self.steps.append(rec)

    def finalize(self, gamma: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (logps, advantages, value_targets) for the whole trajectory."""
        returns = []
        G = 0.0
        for step in reversed(self.steps):
            G = float(step.reward) + gamma * G
            returns.append(G)
        returns.reverse()

        values = torch.stack([s.value for s in self.steps])
        targets = torch.tensor(returns, dtype=torch.float32)
        adv = targets - values.detach()
        logps = torch.stack([s.logp for s in self.steps])
        return logps, adv, targets


# ------------------------------- RL Engine --------------------------------- #

class RLEngine(AbstractEngine):
    """Policy-gradient agent that plugs into the existing multiplayer framework.

    During training, set `train_mode=True` to accumulate trajectory data on each
    of this agent's *own* turns; call `finish_round(win_bonus)` after the round.
    """
    def __init__(self, playing_as: int, policy: PolicyNet, optimizer: optim.Optimizer,
                 gamma: float = 0.99, entropy_coef: float = 0.01, train_mode: bool = True):
        self.me = playing_as
        self.policy = policy
        self.opt = optimizer
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.train_mode = train_mode
        self.traj = Trajectory()
        # Agent memory of the next few shells (first one used as a feature)
        self.known_shells: List[Optional[bool]] = []

    # -------------------- AbstractEngine interface methods ----------------- #

    def choice(self, game: BuckshotRoulette) -> Tuple[str, Optional[int]]:
        # Prepare state + legal moves.
        if not self.known_shells:
            self.known_shells = [None] * game.total
            if game.total == 1:
                self.known_shells[0] = game.live > 0
        if len(self.known_shells) > game.total:
            self.known_shells = self.known_shells[-game.total:]
        elif len(self.known_shells) < game.total:
            self.known_shells += [None] * (game.total - len(self.known_shells))

        legal: List[Tuple[int, str]] = game.moves()  # list[(target_idx, move_str)]               
        state = encode_state(game, self.me, self.known_shells)
        move_feats = torch.stack([encode_move(game, self.me, m) for m in legal])

        # Score and sample.
        logits, value = self.policy(state, move_feats)
        dist = torch.distributions.Categorical(logits=logits)
        a_idx = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(a_idx))

        # Record the step with a provisional reward of 0 (filled in at on_own_move).
        if self.train_mode:
            self.traj.add(StepRecord(state=state, moves=move_feats, act_idx=a_idx,
                                     logp=logp.detach(), value=value.detach(), reward=0.0))

        # Convert back to environment API.
        target, move = legal[a_idx]
        ad_target = None
        if GameStatus.ADRENALINE_ACTIVE in game.statuses:
            # When adrenaline is active, env expects the target player separately
            ad_target = target
        return move, ad_target

    def on_own_move(self, last_move: str, res):
        """Fill in immediate shaping reward for the *last* recorded step.
        `res` is the agent-private result from env.make_move.
        - Shoot live at an opponent: +damage
        - Shoot live at self: -damage
        - Magnifying glass / beer information is incorporated into memory only.
        """
        # Update shell knowledge for future feature encoding.
        self._update_knowledge_after_move(last_move, res)
        
        if not self.train_mode or not self.traj.steps:
            return

        # Reward shaping from private result structure.
        r = 0.0
        if isinstance(res, tuple) or isinstance(res, list):
            # Different moves return different shapes; for shooting it is +/- damage.
            # For beer/magnifying glass, the *value* of information is learned implicitly.
            if last_move.startswith("shoot_") and isinstance(res, (tuple, list)):
                # env returns (private, public, shotgun_after). We already received the private part here.
                # But in play_round, the engine receives only the private scalar in `res` for on_own_move.
                pass
        if isinstance(res, (int, float)) and last_move.startswith("shoot_"):
            r += float(res)*0.05
        for i, shell in enumerate(self.known_shells):
            if shell is not None:
                r += 0.01 * (0.75 ** i)

        # Append reward to the most recent step.
        self.traj.steps[-1].reward += r

    def on_opponent_move(self, last_move, res):
        if isinstance(last_move, str) and (last_move.startswith("shoot_") or last_move == "beer"):
            if self.known_shells:
                self.known_shells = self.known_shells[1:]
        elif last_move == "inverter" and self.known_shells and self.known_shells[0] is not None:
            self.known_shells[0] = not self.known_shells[0]
        pass

    def on_reload(self, board: BuckshotRoulette):
        # Reset shell memory to the new sequence size.
        self.known_shells = [None] * board.total

    # --------------------------- Training utilities ------------------------- #

    def _update_knowledge_after_move(self, last_move: str, result):
        # Mirror the Dealer's memory rules for first-shell knowledge.
        if last_move.startswith("shoot_"):
            if self.known_shells:
                self.known_shells = self.known_shells[1:]
        elif last_move == "magnifying_glass":
            if self.known_shells:
                # result is True/False for next shell
                try:
                    self.known_shells[0] = bool(result)
                except Exception:
                    pass
        elif last_move == "burner_phone" and result is not None:
            # result is (idx, bool)
            try:
                idx, val = result
                if 0 <= idx < len(self.known_shells):
                    self.known_shells[idx] = bool(val)
            except Exception:
                pass
        elif last_move == "beer" and result is not None:
            # result for beer is the top-shell value (or None when nothing to pop)
            try:
                if isinstance(result, bool):
                    self.known_shells = self.known_shells[1:]
                    self.known_shells.insert(0, result)
            except Exception:
                pass
        elif last_move == "inverter" and self.known_shells and self.known_shells[0] is not None:
            self.known_shells[0] = not self.known_shells[0]

    def finish_round(self, did_win: bool):
        """Call after a round ends to apply policy/value updates from the trajectory."""
        if not self.train_mode or len(self.traj.steps) == 0:
            self.traj = Trajectory()
            return

        # Add terminal bonus to the *last* step only.
        terminal_bonus = 1.0 if did_win else -1.0
        self.traj.steps[-1].reward += terminal_bonus

        logps, adv, targets = self.traj.finalize(self.gamma)

        # Recompute entropy from stored logits is non-trivial; approximate by using
        # the logps we tracked and assuming near-on-policy sampling. For a more
        # exact entropy term, re-run forward passes with stored states & move sets.
        # Here we do the exact recompute for stability.
        policy_loss = torch.tensor(0.0)
        entropy = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)

        for step, A, tgt in zip(self.traj.steps, adv, targets):
            logits, value = self.policy(step.state, step.moves)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(torch.tensor(step.act_idx))
            policy_loss = policy_loss - logp * A
            entropy = entropy + dist.entropy().mean()
            value_loss = value_loss + (value - tgt).pow(2).mean()

        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        # Reset trajectory for the next round.
        self.traj = Trajectory()


# --------------------------------- Training -------------------------------- #

def play_one_round(engines: List[AbstractEngine], trainer_idx: int, cfg: RoundConfig) -> int:
    """Play a single round and return the winning player's index."""
    n = len(engines)

    # perm[new_pos] = old_idx  (mapping from shuffled position -> original index)
    perm = list(range(n))
    random.shuffle(perm)

    # Make a shuffled *copy* of engines (no mutation of the input list)
    shuffled_engines = [engines[i] for i in perm]
    for new_pos, eng in enumerate(shuffled_engines):
        if hasattr(eng, "me"):
            eng.me = new_pos
    # If you need to pass trainer_idx in the shuffled seating, map it:
    # inv[old_idx] = new_pos
    inv = [0] * n
    for new_pos, old_idx in enumerate(perm):
        inv[old_idx] = new_pos
    trainer_idx_shuffled = inv[trainer_idx]  # use if your Game needs it

    game = BuckshotGame(players=shuffled_engines, config=GameConfig([cfg] * 3))
    # If BuckshotGame needs a trainer index, pass trainer_idx_shuffled to it.

    winner_shuffled = game.play()       # index in shuffled_engines
    winner_original = perm[winner_shuffled]  # map back to original engines index
    
    seat_wr.update(trainer_idx_shuffled, did_win=(winner_shuffled == trainer_idx_shuffled))
    return winner_original


def make_default_round(player_count: int, **kwargs) -> RoundConfig:
    return RoundConfig(player_count=player_count, **kwargs)


def _checkpoint_schedule(total_episodes: int) -> Deque[int]:
    """Return sorted episode numbers representing each 5% milestone."""
    if total_episodes <= 0:
        return deque()
    fractions = range(5, 101, 5)
    checkpoints = sorted({max(1, math.ceil(total_episodes * pct / 100.0)) for pct in fractions})
    return deque(checkpoints)


def _checkpoint_directory(model_root: Path, hidden_layers: int, hidden_dim: int, opponent: str) -> Path:
    folder = model_root / f"model_{hidden_layers}_{hidden_dim}_{opponent}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_checkpoint(policy: PolicyNet, directory: Path, hidden_layers: int, hidden_dim: int, opponent: str, episode: int) -> None:
    name = f"model_{hidden_layers}_{hidden_dim}_{opponent}_{episode}.pt"
    path = directory / name
    torch.save(policy.state_dict(), path)
    print(f"Saved checkpoint to {path}")


def train(
    *,
    episodes: int,
    players: int,
    opponent_type: str,
    mirror_mode: str,
    target_update: int,
    log_every: int,
    lr: float,
    gamma: float,
    entropy: float,
    hidden_dim: int,
    hidden_layers: int,
    skip_connection: bool,
    model_root: Path,
    load_path: Optional[str] = None,
):
    global prev_policy
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer dimensions for encoders.
    # Conservative upper bounds: up to 4 players -> state/move dims below are adequate.
    max_players = players if players > 0 else 4
    state_dim = 97
    move_dim = len(MOVE_TYPES) + (max_players + 1) + 2  # type + offsetOH + (adrenaline_turn,self_flag)

    policy = PolicyNet(
        state_dim=state_dim,
        move_dim=move_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        skip_connection=skip_connection,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    if load_path:
        print(f"Loading {load_path}...")
        policy.load_state_dict(torch.load(load_path, map_location=device))
    else:
        print("No loaded model; training from scratch. Models can be loaded with --load.")
    
    # Create engines: RL vs Dealer (default) or RL mirror self-play if desired.
    rl = RLEngine(playing_as=0, policy=policy, optimizer=optimizer,
                  gamma=gamma, entropy_coef=entropy, train_mode=True)
    engines: List[AbstractEngine] = [rl]
    for i in range(1, max_players):
        if opponent_type == "dealer":
            opponent: AbstractEngine = Dealer(playing_as=i)
        elif opponent_type == "random":
            opponent = RandomEngine(playing_as=i)
        else:
            # Mirror self-play modes:
            # - shared: opponent uses the *same live* policy params (non-training)
            # - frozen: opponent is a deep-copied policy updated every --target-update episodes
            if mirror_mode == "shared":
                opponent = RLEngine(playing_as=i, policy=policy, optimizer=optimizer,
                                    gamma=gamma, entropy_coef=entropy, train_mode=False)
            else:
                opp_policy = copy.deepcopy(policy)
                opponent = RLEngine(playing_as=i, policy=opp_policy,
                                    optimizer=optim.Adam(opp_policy.parameters(), lr=1e-5),
                                    gamma=gamma, entropy_coef=entropy, train_mode=False)
        engines.append(opponent)

    
    checkpoint_dir = _checkpoint_directory(model_root, hidden_layers, hidden_dim, opponent_type)
    checkpoint_schedule = _checkpoint_schedule(episodes)

    # Training loop over rounds.
    win = 0
    for ep in range(1, episodes + 1):
        player_count = players if players > 0 else random.randint(2, 4)
        cfg = make_default_round(player_count=player_count)
        w = play_one_round(engines, trainer_idx=0, cfg=cfg)
        did_win = (w == 0)
        if did_win:
            win += 1
        rl.finish_round(did_win)

        # Periodically refresh frozen mirror with current learner weights
        if opponent_type == "mirror" and mirror_mode == "frozen" and ep % max(1, target_update) == 0:
            for i in range(1, len(engines)):
                if isinstance(engines[i], RLEngine):
                    engines[i].policy.load_state_dict(policy.state_dict())
        while checkpoint_schedule and ep >= checkpoint_schedule[0]:
            milestone = checkpoint_schedule.popleft()
            _save_checkpoint(policy, checkpoint_dir, hidden_layers, hidden_dim, opponent_type, milestone)
        if ep % max(1, log_every) == 0:
            pct = 100.0 * win / ep
            print(f"[ep {ep:5d}] win%={pct:5.1f} (last {log_every}) vs {opponent_type}")
            per_seat = seat_wr.win_rates()
            
            wr_avg = sum(x for x in per_seat if x == x) / 4.0  # mean of seats (NaNs guarded)
            print("\tWR_avg={:.3f} | seats {}".format(wr_avg, format_seat_wr(seat_wr)))
            seat_wr.reset()

            # Logit drift (vs previous snapshot)
            if prev_policy is not None:
                logit_delta = logit_probe.logit_l2_delta(prev_policy, rl.policy)
                print("\tlogitΔ(L2)={:.4f}".format(logit_delta))
            prev_policy = copy.deepcopy(rl.policy).eval()  # freeze for next interval

            # Value loss EMA + drift (update this wherever you already compute value loss scalars)
            
            v_drift = value_ema.drift_since_last_log()             # call only when you log
            print("\tvalue_loss_MA={:.4f}{}".format(
                value_ema.ma, "" if v_drift is None else " (Δ since last log={:+.4f})".format(v_drift)))


@torch.no_grad()
def evaluate(
    *,
    episodes: int,
    players: int,
    opponent: str,
    mirror_mode: str,
    gamma: float,
    entropy: float,
    hidden_dim: int,
    hidden_layers: int,
    skip_connection: bool,
    load_path: Optional[str] = None,
):
    global prev_policy
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_players = players if players > 0 else 4
    state_dim = 97
    move_dim = len(MOVE_TYPES) + (max_players + 1) + 2

    policy = PolicyNet(
        state_dim=state_dim,
        move_dim=move_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        skip_connection=skip_connection,
    ).to(device)
    if load_path:
        print(f"Loading {load_path}...")
        policy.load_state_dict(torch.load(load_path, map_location=device))
    else:
        print("Warning: No loaded model. Models can be loaded with --load.")

    eval_optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    rl = RLEngine(playing_as=0, policy=policy, optimizer=eval_optimizer,
                  gamma=gamma, entropy_coef=entropy, train_mode=False)
    engines: List[AbstractEngine] = [rl]
    for i in range(1, max_players):
        if opponent == "dealer":
            opponent: AbstractEngine = Dealer(playing_as=i)
        elif opponent == "random":
            opponent = RandomEngine(playing_as=i)
        else:
            # Mirror self-play modes:
            # - shared: opponent uses the *same live* policy params (non-training)
            # - frozen: opponent is a deep-copied policy updated every --target-update episodes
            if mirror_mode == "shared":
                opponent = RLEngine(playing_as=i, policy=policy, optimizer=eval_optimizer,
                                    gamma=gamma, entropy_coef=entropy, train_mode=False)
            else:
                opp_policy = copy.deepcopy(policy)
                opponent = RLEngine(playing_as=i, policy=opp_policy,
                                    optimizer=optim.Adam(opp_policy.parameters(), lr=1e-5),
                                    gamma=gamma, entropy_coef=entropy, train_mode=False)
        engines.append(opponent)
    

    win = 0
    for ep in range(1, episodes + 1):
        player_count = players if players > 0 else random.randint(2, 4)
        cfg = make_default_round(player_count=player_count)
        w = play_one_round(engines, trainer_idx=0, cfg=cfg)
        did_win = (w == 0)
        if did_win:
            win += 1

    pct = 100.0 * win / episodes
    print(f"Eval: {win}/{episodes} wins ({pct:.1f}%) vs {opponent}")
    
    per_seat = seat_wr.win_rates()
    wr_avg = sum(x for x in per_seat if x == x) / 4.0  # mean of seats (NaNs guarded)
    print("\tWR_avg={:.3f} | seats {}".format(wr_avg, format_seat_wr(seat_wr)))
    seat_wr.reset()

    # Logit drift (vs previous snapshot)
    if prev_policy is not None:
        logit_delta = logit_probe.logit_l2_delta(prev_policy, rl.policy)
        print("\tlogitΔ(L2)={:.4f}".format(logit_delta))
    prev_policy = copy.deepcopy(rl.policy).eval()  # freeze for next interval

    # Value loss EMA + drift (update this wherever you already compute value loss scalars)
    
    v_drift = value_ema.drift_since_last_log()             # call only when you log
    print("\tvalue_loss_MA={:.4f}{}".format(
        value_ema.ma, "" if v_drift is None else " (Δ since last log={:+.4f})".format(v_drift)))


# ----------------------------------- CLI ----------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Policy-gradient training for Buckshot Roulette")
    p.add_argument("--episodes", type=int, default=1000, help="number of rounds to train/evaluate")
    p.add_argument("--players", type=int, default=4, choices=[-1, 2,3,4], help="number of players in a round")
    p.add_argument("--hidden", "--hidden-dim", dest="hidden_dim", type=int, default=128,
                   help="hidden layer width for the policy/value encoders")
    p.add_argument("--hidden-layers", type=int, default=2, help="number of hidden layers in the MLP towers")
    p.add_argument("--skip-connection", action="store_true",
                   help="concatenate raw inputs with the final hidden layer before scoring moves")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p.add_argument("--entropy", type=float, default=0.01, help="entropy bonus coefficient")
    p.add_argument("--log-every", dest="log_every", type=int, default=200)
    p.add_argument("--opponent", type=str, default="dealer", choices=["dealer", "random", "mirror"], help="opponent type")
    p.add_argument("--model-root", type=str, default="models", help="directory to store auto-named model checkpoints")
    p.add_argument("--load", type=str, default=None, help="path to load policy weights for evaluation")
    p.add_argument("--eval", action="store_true", help="run evaluation only")
    p.add_argument("--mirror-mode", type=str, default="frozen", choices=["shared","frozen"],
                   help="mirror opponents: 'shared' uses the live learner policy; 'frozen' uses a slowly updated copy")
    p.add_argument("--target-update", type=int, default=200, help="episodes between copying learner weights to frozen mirror")
    return p.parse_args()


def main():
    args = parse_args()
    model_root = Path(args.model_root)
    shared_kwargs = dict(
        players=args.players,
        opponent_type=args.opponent,
        mirror_mode=args.mirror_mode,
        gamma=args.gamma,
        entropy=args.entropy,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        skip_connection=args.skip_connection,
        load_path=args.load,
    )
    if args.eval:
        evaluate(episodes=args.episodes, **shared_kwargs)
    else:
        train(
            episodes=args.episodes,
            target_update=args.target_update,
            log_every=args.log_every,
            lr=args.lr,
            model_root=model_root,
            **shared_kwargs,
        )


if __name__ == "__main__":
    main()
