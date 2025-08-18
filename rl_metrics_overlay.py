
"""
rl_metrics_overlay.py

Drop-in metrics overlay for 4-player self-play training.
Implements, without changing your model:
  A.2 Seat-stratified win rate (per-seat WRs)
  B.8 Logit drift between current and previous checkpoints
  C.9 Value-loss EMA & drift (plus optional slope statistics)

Usage sketch (pseudocode):

    from rl_metrics_overlay import (
        SeatWRTracker, LogitDriftProbe, ValueLossEMA,
        slope_ci, format_seat_wr
    )

    seat_wr = SeatWRTracker(num_seats=4)
    logit_probe = LogitDriftProbe(buffer_size=2048, device="cpu")
    value_ema = ValueLossEMA(beta=0.98)  # effective window ~ 1/(1-beta)

    prev_policy = None

    for eval_step in ...:
        # --- During eval games:
        # After each game, with learner_seat in {0,1,2,3} and did_win in {0,1}:
        seat_wr.update(learner_seat, did_win)

        # While stepping or during eval, add a few probes (encoded state & legal-move features):
        logit_probe.maybe_add(state_feats, move_feats)  # tensors on CPU

        # --- After eval batch:
        per_seat_wr = seat_wr.win_rates()            # list of length 4
        avg_wr = sum(per_seat_wr)/len(per_seat_wr)   # seat-averaged WR

        logit_delta = None
        if prev_policy is not None:
            logit_delta = logit_probe.logit_l2_delta(prev_policy, curr_policy, logits_fn)
        prev_policy = deepcopy(curr_policy).eval()   # snapshot for next time

        v_ma = value_ema.update(latest_value_loss_scalar)  # pass a float each train/eval
        v_drift = value_ema.drift_since_last_log()         # None on first call

        # Optional: slope on the recent history of avg WR / v_ma (use ordinary least squares)
        slope, lo, hi = slope_ci(xs, ys)  # where xs are eval indices or steps, ys are the metric values

        print(f"WR_avg={avg_wr:.3f}  seats " + format_seat_wr(seat_wr) +
              (f" | logitΔ(L2)={logit_delta:.4f}" if logit_delta is not None else "") +
              (f" | value_loss_MA={v_ma:.4f}" if v_ma is not None else "") +
              (f" (Δ since last log={v_drift:+.4f})" if v_drift is not None else "")
        )
        seat_wr.reset()  # if you want fresh counts per eval

Notes
-----
- This overlay expects you already have *encoded* state features and move features available
  at choice-time (the same tensors you feed into your policy). It does not change your model.
- You can wire `maybe_add` right where you build `(state_feats, move_feats)` in your engine.
- Provide a `logits_fn(policy, state_feats, move_feats) -> logits_on_legal_actions` callback,
  or rely on the default heuristic: it tries to call `policy(state_feats, move_feats)`.
- All operations here are torch/no-grad and CPU by default to avoid training impact.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
import math
import random

import torch
import numpy as np


# ------------------------ A.2 Seat-stratified WR ----------------------------

@dataclass
class SeatWRTracker:
    num_seats: int = 4
    seat_wins: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    seat_games: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def update(self, seat: int, did_win: bool) -> None:
        if seat < 0 or seat >= self.num_seats:
            return
        self.seat_games[seat] += 1
        if did_win:
            self.seat_wins[seat] += 1

    def win_rates(self) -> List[float]:
        wrs = []
        for s in range(self.num_seats):
            g = self.seat_games[s]
            w = self.seat_wins[s]
            wrs.append((w / g) if g > 0 else float("nan"))
        return wrs

    def spread(self) -> float:
        wrs = [wr for wr in self.win_rates() if not math.isnan(wr)]
        return (max(wrs) - min(wrs)) if wrs else float("nan")

    def reset(self) -> None:
        self.seat_wins = [0] * self.num_seats
        self.seat_games = [0] * self.num_seats


def format_seat_wr(tracker: SeatWRTracker) -> str:
    parts = []
    for s in range(tracker.num_seats):
        g = tracker.seat_games[s]
        if g > 0:
            w = tracker.seat_wins[s]
            parts.append(f"{s}:{100.0*w/g:.1f}%({g})")
        else:
            parts.append(f"{s}:-")
    return " ".join(parts)


# ------------------------ B.8 Logit drift (L2) ------------------------------

class LogitDriftProbe:
    """
    Maintains a small reservoir of (state_feats, move_feats) pairs collected
    during normal play, and computes normalized L2 drift between logits produced
    by two policies on the same probe.
    """
    def __init__(self, buffer_size: int = 1024, device: str = "cpu", replace_prob: float = 1.0):
        self.buffer_size = int(buffer_size)
        self.device = torch.device(device)
        self.states: List[torch.Tensor] = []
        self.moves: List[torch.Tensor] = []
        self.seen: int = 0
        self.replace_prob: float = float(replace_prob)

    def maybe_add(self, state_feats: torch.Tensor, move_feats: torch.Tensor) -> None:
        """
        Add a (state, moves) pair with reservoir sampling.
        - `state_feats`: shape [S] or [1,S]
        - `move_feats`: shape [M, F] for M legal moves (features per move)
        Tensors are detached & moved to CPU to avoid autograd and GPU memory pressure.
        """
        if random.random() > self.replace_prob:
            return
        s = state_feats.detach().cpu().clone().view(-1)
        m = move_feats.detach().cpu().clone()
        self.seen += 1
        if len(self.states) < self.buffer_size:
            self.states.append(s)
            self.moves.append(m)
        else:
            j = random.randint(0, self.seen - 1)
            if j < self.buffer_size:
                self.states[j] = s
                self.moves[j] = m

    def __len__(self) -> int:
        return len(self.states)

    @staticmethod
    def _default_logits_fn(policy, state_feats: torch.Tensor, move_feats: torch.Tensor) -> torch.Tensor:
        """
        Best-effort default: try `policy(state_feats, move_feats)`.
        Must return logits over the M legal actions (shape [M]).
        """
        out = policy(state_feats.unsqueeze(0), move_feats)  # common pattern
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 2 and out.shape[0] == 1:
            out = out[0]
        return out  # shape [M]

    @torch.no_grad()
    def logit_l2_delta(
        self,
        policy_prev,
        policy_curr,
        logits_fn: Optional[Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        eps: float = 1e-6,
        batch: int = 128,
    ) -> float:
        """
        Compute mean normalized L2 distance between logits on the probe set.

        If `normalize` is True, per-state logits are z-normalized across the M legal moves
        (shift/scale invariant), and the L2 is divided by sqrt(M).
        """
        if len(self.states) == 0:
            return float("nan")

        logits_fn = logits_fn or self._default_logits_fn
        policy_prev = policy_prev.to(self.device).eval()
        policy_curr = policy_curr.to(self.device).eval()

        n = len(self.states)
        total = 0.0
        count = 0
        for i in range(0, n, batch):
            s_batch = self.states[i:i+batch]
            m_batch = self.moves[i:i+batch]
            # Stack with padding where needed (jagged M per state) by looping—simple and robust.
            for s, m in zip(s_batch, m_batch):
                s = s.to(self.device)
                m = m.to(self.device)
                z_prev = logits_fn(policy_prev, s, m).float()
                z_curr = logits_fn(policy_curr, s, m).float()
                if normalize:
                    mu_p = z_prev.mean()
                    sd_p = z_prev.std(unbiased=False).clamp_min(eps)
                    mu_c = z_curr.mean()
                    sd_c = z_curr.std(unbiased=False).clamp_min(eps)
                    z_prev = (z_prev - mu_p) / sd_p
                    z_curr = (z_curr - mu_c) / sd_c
                M = float(max(1, z_prev.numel()))
                d = torch.norm(z_curr - z_prev, p=2) / math.sqrt(M)
                total += float(d.item())
                count += 1
        return total / max(1, count)


# ------------------------ C.9 Value-loss EMA & drift ------------------------

@dataclass
class ValueLossEMA:
    beta: float = 0.98
    ma: Optional[float] = None
    _last_logged: Optional[float] = None
    history: List[Tuple[float, float]] = field(default_factory=list)  # (x, ma)

    def update(self, value_loss_scalar: float, x: Optional[float] = None) -> float:
        """Update EMA with a new scalar. Optionally record (x, ma) to history for trend analysis."""
        v = float(value_loss_scalar)
        if self.ma is None:
            self.ma = v
        else:
            self.ma = self.beta * self.ma + (1.0 - self.beta) * v
        if x is not None:
            self.history.append((float(x), float(self.ma)))
        return self.ma

    def drift_since_last_log(self) -> Optional[float]:
        """Return MA - last_logged, and update last_logged to current MA. None on first use."""
        if self.ma is None:
            return None
        if self._last_logged is None:
            self._last_logged = self.ma
            return None
        delta = self.ma - self._last_logged
        self._last_logged = self.ma
        return delta


# ------------------------ Tiny regression + CI helper -----------------------

def slope_ci(xs: Sequence[float], ys: Sequence[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Ordinary least squares slope with (1-alpha) CI.
    Returns (slope, lo, hi). Uses t-distribution CI.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = x.size
    if n < 3:
        sxx = sxy = 0.0
        slope = 0.0
        return slope, slope, slope
    xm = x.mean()
    ym = y.mean()
    sxx = ((x - xm)**2).sum()
    sxy = ((x - xm)*(y - ym)).sum()
    slope = sxy / (sxx if sxx > 0 else 1.0)
    # Residual variance
    resid = y - (slope * (x - xm) + ym)
    s2 = (resid @ resid) / max(1, n - 2)
    se = math.sqrt(s2 / max(1e-12, sxx))
    # t critical
    from scipy.stats import t as student_t  # available in many envs; if not, fall back to normal approx
    try:
        tcrit = float(student_t.ppf(1 - alpha/2, df=max(1, n - 2)))
    except Exception:
        # Normal approx fallback
        tcrit = 1.96
    lo = slope - tcrit * se
    hi = slope + tcrit * se
    return slope, lo, hi
