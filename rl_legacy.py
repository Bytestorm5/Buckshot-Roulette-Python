"""
Policy-gradient (actor-critic) self-play trainer for Buckshot Roulette.

- Wraps the provided multiplayer environment (BuckshotRoulette/BuckshotGame).
- Implements an AbstractEngine-compatible RL agent that can be dropped in
  alongside the built-in Dealer() or Random() engines.
- Uses a masked policy over the environment's dynamically generated move list.
- Learns from per-move shaping rewards and round win/loss bonuses.

Requirements: PyTorch (tested with 2.x). No other external deps.

Run a quick training session (2-player vs Dealer) from the CLI:

    python buckshot_roulette_rl_policy_gradient.py \
        --episodes 2000 \
        --players 2 \
        --hidden 128 \
        --lr 3e-4 \
        --gamma 0.99 \
        --entropy 0.01

Evaluate vs Dealer after training (100 rounds):

    python buckshot_roulette_rl_policy_gradient.py --eval --episodes 100

Notes
-----
- The environment exposes partial information; we deliberately avoid peeking at
  opponents' item inventories to keep the agent honest. Public info (charges,
  statuses, live/blank counts) is used, plus the agent's own remembered shell
  knowledge from tools like magnifying glass / beer / burner phone.
- The policy scores *candidate moves* using a state encoder and a separate move
  encoder, then applies a masked softmax over legal moves only.
- The value head provides a baseline to reduce variance (actor-critic).
"""

from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def encode_state(game: BuckshotRoulette, me: int, known_shell_first: Optional[bool]) -> torch.Tensor:
    """Encode *public* game state + my own info into a flat feature vector.
    - No opponent item peeking.
    - Includes: charges per player, my items, live/total shells, statuses, direction.
    - Includes agent memory for the next shell if known: [-1 unknown, 0 blank, 1 live].
    """
    pc = game.player_count

    # Charges (normalized to [0,1] by start_charges), plus alive flag.
    start_charges = float(game.config.start_charges)
    charges = []
    alive = []
    # Order by *offset* relative to me for translational invariance.
    for off in range(pc):
        idx = (me + off) % pc
        c = game.charges[idx]
        charges.append(c / max(1.0, start_charges))
        alive.append(1.0 if c > 0 else 0.0)

    # My items (counts clamped to [0, 8] and normalized by 8).
    my_items = game.items[me]
    item_feats = []
    for key in ITEMS:
        item_feats.append(min(8, getattr(my_items, key)) / 8.0)

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
    ks = [0.0, 0.0]
    if known_shell_first is True:
        ks = [1.0, 0.0]          # known live
    elif known_shell_first is False:
        ks = [0.0, 1.0]          # known blank

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

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, move_dim: int, hidden: int = 128):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.move = nn.Sequential(
            nn.Linear(move_dim, hidden),
            nn.ReLU(),
        )
        self.val_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, move_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score each candidate move and return (logits, value).
        - state: [S]
        - move_batch: [K, M]
        Returns:
            logits: [K]
            value:  [1]
        """
        s = self.state(state)               # [H]
        mv = self.move(move_batch)          # [K, H]
        # Dot-product scorer between state embedding and each move embedding
        logits = (mv * s).sum(dim=1)        # [K]
        value = self.val_head(s).squeeze(-1)  # []
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

        legal = game.moves()  # list[(target_idx, move_str)]
        state = encode_state(game, self.me, self.known_shells[0] if self.known_shells else None)
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

    def on_own_move(self, last_move, res):
        """Fill in immediate shaping reward for the *last* recorded step.
        `res` is the agent-private result from env.make_move.
        - Shoot live at an opponent: +damage
        - Shoot live at self: -damage
        - Magnifying glass / beer information is incorporated into memory only.
        """
        if not self.train_mode or not self.traj.steps:
            # Still update internal knowledge even if not training.
            self._update_knowledge_after_move(last_move, res)
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
        if isinstance(res, (int, float)):
            r += float(res)

        # Append reward to the most recent step.
        self.traj.steps[-1].reward += r

        # Update shell knowledge for future feature encoding.
        self._update_knowledge_after_move(last_move, res)

    def on_opponent_move(self, last_move, res):
        # If opponent shot me, my charges drop; the env does not pass explicit target.
        # For simplicity we don't shape reward here; terminal win/loss will drive learning.
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
    return winner_original


def make_default_round(player_count: int) -> RoundConfig:
    return RoundConfig(player_count=player_count)


def train(args: argparse.Namespace):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer dimensions for encoders.
    # Conservative upper bounds: up to 4 players -> state/move dims below are adequate.
    max_players = args.players if args.players > 0 else 4
    state_dim = (
        2 * max_players   # charges + alive flags
        + len(ITEMS)      # my items
        + 4               # shells: live/8, total/8, p_live, one_left
        + 7               # statuses
        + 2               # direction
        + 2               # known_first_shell one-hot (live/blank, unknown=00)
    )
    move_dim = len(MOVE_TYPES) + (max_players + 1) + 2  # type + offsetOH + (adrenaline_turn,self_flag)

    policy = PolicyNet(state_dim=state_dim, move_dim=move_dim, hidden=args.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    if args.load:
        print(f"Loading {args.load}...")
        policy.load_state_dict(torch.load(args.load, map_location=device))
    else:
        print("No loaded model; training from scratch. Models can be loaded with --load.")
    
    # Create engines: RL vs Dealer (default) or RL mirror self-play if desired.
    rl = RLEngine(playing_as=0, policy=policy, optimizer=optimizer,
                  gamma=args.gamma, entropy_coef=args.entropy, train_mode=True)
    engines: List[AbstractEngine] = [rl]
    for i in range(1, args.players):
        if args.opponent == "dealer":
            opponent: AbstractEngine = Dealer(playing_as=i)
        elif args.opponent == "random":
            opponent = RandomEngine(playing_as=i)
        else:
            # Mirror self-play modes:
            # - shared: opponent uses the *same live* policy params (non-training)
            # - frozen: opponent is a deep-copied policy updated every --target-update episodes
            if args.mirror_mode == "shared":
                opponent = RLEngine(playing_as=i, policy=policy, optimizer=optimizer,
                                    gamma=args.gamma, entropy_coef=args.entropy, train_mode=False)
            else:
                opp_policy = copy.deepcopy(policy)
                opponent = RLEngine(playing_as=i, policy=opp_policy,
                                    optimizer=optim.Adam(opp_policy.parameters(), lr=1e-5),
                                    gamma=args.gamma, entropy_coef=args.entropy, train_mode=False)
        engines.append(opponent)

    

    # Training loop over rounds.
    win = 0
    for ep in range(1, args.episodes + 1):
        player_count = args.players if args.players > 0 else random.randint(2, 4)
        cfg = make_default_round(player_count=player_count)
        w = play_one_round(engines, trainer_idx=0, cfg=cfg)
        did_win = (w == 0)
        if did_win:
            win += 1
        rl.finish_round(did_win)

        # Periodically refresh frozen mirror with current learner weights
        if args.opponent == "mirror" and args.mirror_mode == "frozen" and ep % max(1, args.target_update) == 0:
            for i in range(1, len(engines)):
                if isinstance(engines[i], RLEngine):
                    engines[i].policy.load_state_dict(policy.state_dict())

        if ep % max(1, args.log_every) == 0:
            pct = 100.0 * win / ep
            print(f"[ep {ep:5d}] win%={pct:5.1f} (last {args.log_every}) vs {args.opponent}")

    # Save the trained weights if requested.
    if args.save:
        torch.save(policy.state_dict(), args.save)
        print(f"Saved policy to {args.save}")


@torch.no_grad()
def evaluate(args: argparse.Namespace):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_players = args.players if args.players > 0 else 4
    state_dim = (
        2 * max_players + len(ITEMS) + 4 + 7 + 2 + 2
    )
    move_dim = len(MOVE_TYPES) + (max_players + 1) + 2

    policy = PolicyNet(state_dim=state_dim, move_dim=move_dim, hidden=args.hidden).to(device)
    if args.load:
        print(f"Loading {args.load}...")
        policy.load_state_dict(torch.load(args.load, map_location=device))
    else:
        print("Warning: No loaded model. Models can be loaded with --load.")

    rl = RLEngine(playing_as=0, policy=policy, optimizer=optim.Adam(policy.parameters(), lr=1e-5),
                  gamma=args.gamma, entropy_coef=args.entropy, train_mode=False)
    engines: List[AbstractEngine] = [rl]
    for i in range(1, args.players):
        if args.opponent == "dealer":
            opponent: AbstractEngine = Dealer(playing_as=i)
        elif args.opponent == "random":
            opponent = RandomEngine(playing_as=i)
        else:
            # Mirror self-play modes:
            # - shared: opponent uses the *same live* policy params (non-training)
            # - frozen: opponent is a deep-copied policy updated every --target-update episodes
            if args.mirror_mode == "shared":
                opponent = RLEngine(playing_as=i, policy=policy, optimizer=optimizer,
                                    gamma=args.gamma, entropy_coef=args.entropy, train_mode=False)
            else:
                opp_policy = copy.deepcopy(policy)
                opponent = RLEngine(playing_as=i, policy=opp_policy,
                                    optimizer=optim.Adam(opp_policy.parameters(), lr=1e-5),
                                    gamma=args.gamma, entropy_coef=args.entropy, train_mode=False)
        engines.append(opponent)
    

    win = 0
    for ep in trange(1, args.episodes + 1, desc="Running Episodes", leave=False):
        player_count = args.players if args.players > 0 else random.randint(2, 4)
        cfg = make_default_round(player_count=player_count)
        w = play_one_round(engines, trainer_idx=0, cfg=cfg)
        did_win = (w == 0)
        if did_win:
            win += 1

    pct = 100.0 * win / args.episodes
    print(f"Eval: {win}/{args.episodes} wins ({pct:.1f}%) vs {args.opponent}")


# ----------------------------------- CLI ----------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Policy-gradient training for Buckshot Roulette")
    p.add_argument("--episodes", type=int, default=1000, help="number of rounds to train/evaluate")
    p.add_argument("--players", type=int, default=2, choices=[-1, 2,3,4], help="number of players in a round")
    p.add_argument("--hidden", type=int, default=128, help="hidden size for encoders")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p.add_argument("--entropy", type=float, default=0.01, help="entropy bonus coefficient")
    p.add_argument("--log-every", dest="log_every", type=int, default=50)
    p.add_argument("--opponent", type=str, default="dealer", choices=["dealer", "random", "mirror"], help="opponent type")
    p.add_argument("--save", type=str, default=None, help="path to save policy weights after training")
    p.add_argument("--load", type=str, default=None, help="path to load policy weights for evaluation")
    p.add_argument("--eval", action="store_true", help="run evaluation only")
    p.add_argument("--mirror-mode", type=str, default="frozen", choices=["shared","frozen"],
                   help="mirror opponents: 'shared' uses the live learner policy; 'frozen' uses a slowly updated copy")
    p.add_argument("--target-update", type=int, default=200, help="episodes between copying learner weights to frozen mirror")
    return p.parse_args()


def main():
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
