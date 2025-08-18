"""
Single-script Flask app for visualizing & simulating Buckshot Roulette (2–4 players).

Features
- Choose player count (2–4) and starting charge count.
- For each player slot choose: Human, Random, Dealer, or Model - <name> (hard-coded registry below).
- Live "board" view updates after each move; shows items, turns, charges, statuses, shotgun length.
- If there are no Human players selected, run batch simulations (N trials) to compare win rates.
- No external templates/static files: all HTML/CSS is inline.
- Tries to import your local project first: 'buckshot_roulette.multiplayer.game' & '...ai'.
  Falls back to local 'game.py' and 'ai.py' if those imports fail.

Integration Notes
- This script makes minimal assumptions about your environment's API.
  It was designed by reading redacted stubs of your files and the following behaviors:
  * Environment class: BuckshotRoulette
    - Must expose: .player_count, .current_turn, .turn_inc, .items (list of Items-like objects),
                   .winner() -> Optional[int or object], .moves() -> list[(ad_target, move) or ...],
                   .make_move(move, shotgun, adrenaline_target=None, allow_reload=False)
    - Also exposes per-sequence counts: .live and .total (used to (re)build the shotgun list).
  * Moves: ai.Random engine looked like it did:
        ad_target, move = random.choice(board.moves())
        return move, None if ad_target == self.me else ad_target
    So this app expects engines to return (move, ad_target_or_None).
  * Items object: dataclass-like with integer fields or a dict-like.
  * GameStatus: Enum providing flags like ADRENALINE_ACTIVE, JAMMED_0, etc (only displayed if present).

- If your APIs differ, adjust the small "Adapter" section near the top (SEARCH: ### ADAPTER AREA ###).

Run
    python buckshot_roulette_flask.py
Then open http://127.0.0.1:5000/

"""
import os
import random
import inspect
import importlib
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from flask import Flask, render_template_string, request, redirect, url_for, session, flash

# ------------------------------- Imports -------------------------------

# Try the user's package layout first, then fall back to local files.
GAME_MOD = None
AI_MOD = None
RL_MOD = None

def _soft_import(modname: str):
    try:
        return importlib.import_module(modname)
    except Exception:
        return None

# Preferred project-style imports
# GAME_MOD = _soft_import("buckshot_roulette.multiplayer.game") or _soft_import("game")
# AI_MOD = _soft_import("buckshot_roulette.multiplayer.ai") or _soft_import("ai")
# RL_MOD = _soft_import("buckshot_roulette.multiplayer.rl") or _soft_import("rl")

# if GAME_MOD is None or AI_MOD is None:
#     raise RuntimeError("Could not import game/ai modules. Ensure either your package (buckshot_roulette.multiplayer) "
#                        "is importable, or place game.py and ai.py next to this script.")

# # Pull commonly used names if they exist
# BuckshotRoulette = getattr(GAME_MOD, "BuckshotRoulette", None)
# RoundConfig = getattr(GAME_MOD, "RoundConfig", None)
# GameStatus = getattr(GAME_MOD, "GameStatus", None)

# DealerClass = getattr(AI_MOD, "Dealer", None)
# RandomClass = getattr(AI_MOD, "Random", None) or getattr(AI_MOD, "RandomEngine", None)
from buckshot_roulette.multiplayer.game import BuckshotRoulette, RoundConfig, GameStatus
from buckshot_roulette.multiplayer.ai import Dealer as DealerClass, Random as RandomClass

# --------------------------- Flask Setup --------------------------------

SECRET = os.environ.get("FLASK_SECRET_KEY", os.urandom(16))
app = Flask(__name__)
app.secret_key = SECRET

# Single-process in-memory store for game sessions
GAMES: Dict[str, "GameSession"] = {}

# --------------------------- ADAPTER AREA -------------------------------
# If your project uses slightly different names or signatures, tweak here.

def build_round_config(player_count: int, start_charges: int):
    """
    Create a RoundConfig-like object with a 'start_charges' field, if available.
    Otherwise, return None and let BuckshotRoulette's defaults apply.
    """
    if RoundConfig is None:
        return None
    try:
        # Try a flexible constructor
        try:
            rc = RoundConfig(player_count=player_count)
        except Exception:
            rc = RoundConfig()  # fallback

        # Stuff we know exists in your stub
        if hasattr(rc, "start_charges"):
            setattr(rc, "start_charges", int(start_charges))
        if hasattr(rc, "player_count"):
            setattr(rc, "player_count", int(player_count))
        return rc
    except Exception:
        return None

def build_env(player_count: int, start_charges: int):
    rc = build_round_config(player_count, start_charges)
    try:
        if rc is None:
            env = BuckshotRoulette(config=None, player_count=player_count)  # rely on defaults
        else:
            # Some versions accept (RoundConfig, player_count); others only (RoundConfig)
            try:
                env = BuckshotRoulette(rc, player_count=player_count)
            except TypeError:
                env = BuckshotRoulette(rc)
        return env
    except Exception as e:
        raise RuntimeError(f"Failed to construct BuckshotRoulette: {e}")

def get_items_dict(item_obj: Any) -> Dict[str, int]:
    """Robustly turn an Items-like object into a dict of counts."""
    if item_obj is None:
        return {}
    if is_dataclass(item_obj):
        return {k: int(v) for k, v in asdict(item_obj).items()}
    if hasattr(item_obj, "__dict__"):
        return {k: int(v) for k, v in vars(item_obj).items() if not k.startswith("_")}
    if isinstance(item_obj, dict):
        return {k: int(v) for k, v in item_obj.items()}
    # Fallback by introspection of common fields
    fields = [a for a in dir(item_obj) if not a.startswith("_")]
    out = {}
    for f in fields:
        val = getattr(item_obj, f, None)
        if isinstance(val, (int,)):
            out[f] = int(val)
    return out

def env_winner(env) -> Optional[int]:
    try:
        return env.winner()
    except Exception:
        return None

def env_current_turn(env) -> int:
    return int(getattr(env, "current_turn", 0))

def env_player_count(env) -> int:
    return int(getattr(env, "player_count", 2))

def env_live_total(env) -> Tuple[int, int]:
    """Return (live, total) for the current sequence to rebuild shotgun when empty."""
    live = int(getattr(env, "live", 0))
    total = int(getattr(env, "total", live))
    if total == 0:
        total = live
    return live, total

def env_moves(env) -> List[Tuple[Optional[int], str]]:
    """Return [(ad_target, move_str), ...] to drive engines & UI."""
    try:
        raw = env.moves()
        moves = []
        for m in raw:
            if isinstance(m, (list, tuple)) and len(m) == 2 and isinstance(m[1], str):
                moves.append((int(m[0]) if m[0] is not None else None, str(m[1])))
            elif isinstance(m, str):
                moves.append((env_current_turn(env), m))
            else:
                # unknown; stringify
                moves.append((env_current_turn(env), str(m)))
        return moves
    except Exception:
        return []

def env_make_move(env, move: str, shotgun: List[bool], ad_target: Optional[int]) -> Tuple[Any, Any, List[bool]]:
    """Call make_move with the expected signature; return (res_priv, res_pub, new_shotgun)."""
    fn = getattr(env, "make_move", None)
    if fn is None:
        raise RuntimeError("Environment does not provide make_move(...)")
    # Call with allow_reload=False so we can rebuild shotgun ourselves (like BuckshotGame)
    sig = inspect.signature(fn)
    kwargs = {}
    if "adrenaline_target" in sig.parameters:
        kwargs["adrenaline_target"] = ad_target
    if "allow_reload" in sig.parameters:
        kwargs["allow_reload"] = False
    res_priv, res_pub, new_shotgun = fn(move, shotgun, **kwargs)
    return res_priv, res_pub, new_shotgun

# ------------------------ Engines & Registry ----------------------------

class FallbackRandom:
    """Simple random engine if AI_MOD.Random isn't available."""
    def __init__(self, playing_as: int):
        self.me = playing_as
    def choice(self, board) -> Tuple[str, Optional[int]]:
        moves = env_moves(board)
        if not moves:
            return ("shoot_0", None)
        ad_target, move = random.choice(moves)
        return (move, None if ad_target == self.me else ad_target)
    def on_own_move(self, last_move, res): pass
    def on_opponent_move(self, last_move, res): pass
    def on_reload(self, board): pass

def make_dealer(idx: int):
    if DealerClass is not None:
        return DealerClass(idx)
    return FallbackRandom(idx)

def make_random(idx: int):
    if RandomClass is not None:
        return RandomClass(idx)
    return FallbackRandom(idx)

# Hard-code your models here. You can add multiple bespoke names/paths.
MODEL_REGISTRY: Dict[str, Callable[[int], Any]] = {}

# Example #1: Try to wire a PolicyGradient-like agent in rl.py if present
if RL_MOD is not None:
    # Try common class names; fallback to Random if missing
    for cls_name in ["PolicyGradientAgent", "PolicyGradientEngine", "ActorCriticAgent", "RLAgent"]:
        if hasattr(RL_MOD, cls_name):
            ModelClass = getattr(RL_MOD, cls_name)
            MODEL_REGISTRY[f"Model - {cls_name}"] = lambda idx, MC=ModelClass: MC(idx)
            break

# Always provide at least one demo "model" (alias of Random) so UI has options.
if "Model - Randomized" not in MODEL_REGISTRY:
    MODEL_REGISTRY["Model - Randomized"] = lambda idx: make_random(idx)

# Built-in controller choices
CHOICES = ["Human", "Random", "Dealer"] + list(MODEL_REGISTRY.keys())

def build_engine(kind: str, idx: int):
    if kind == "Human":
        return None
    if kind == "Random":
        return make_random(idx)
    if kind == "Dealer":
        return make_dealer(idx)
    if kind in MODEL_REGISTRY:
        return MODEL_REGISTRY[kind](idx)
    # default
    return make_random(idx)

# -------------------------- Game Session -------------------------------

class GameSession:
    def __init__(self, game_id: str, player_kinds: List[str], start_charges: int):
        self.game_id = game_id
        self.player_kinds = player_kinds[:]  # labels
        self.player_count = len(player_kinds)
        self.start_charges = int(start_charges)

        # Build env and engines
        self.env = build_env(self.player_count, self.start_charges)
        self.engines = [build_engine(kind, i) for i, kind in enumerate(player_kinds)]
        self.humans = [i for i, eng in enumerate(self.engines) if eng is None]

        # Initial shotgun for the first sequence
        live, total = env_live_total(self.env)
        self.shotgun: List[bool] = [True]*live + [False]*(total - live)
        random.shuffle(self.shotgun)

        # History for UI/debug
        self.history: List[Dict[str, Any]] = []

    # Display helpers
    def items_for(self, idx: int) -> Dict[str, int]:
        try:
            return get_items_dict(self.env.items[idx])
        except Exception:
            return {}

    def status_flags(self) -> List[str]:
        try:
            sts = getattr(self.env, "statuses", set())
            if isinstance(sts, set):
                return [str(s) for s in sts]
            return [str(sts)]
        except Exception:
            return []

    def winner(self) -> Optional[int]:
        w = env_winner(self.env)
        # BuckshotGame.play() hinted that it returns the *player index*.
        # If not, try to coerce to int if it looks like a number
        try:
            return int(w) if w is not None else None
        except Exception:
            return w if w in (0,1,2,3) else None

    def legal_moves(self) -> List[Tuple[Optional[int], str]]:
        return env_moves(self.env)

    def do_move(self, move: str, ad_target: Optional[int] = None):
        player = env_current_turn(self.env)
        res_priv, res_pub, new_shotgun = env_make_move(self.env, move, self.shotgun, ad_target)

        # Update shotgun and possibly reload to next sequence
        self.shotgun = list(new_shotgun) if isinstance(new_shotgun, list) else list(self.shotgun)
        if len(self.shotgun) == 0:
            # Move to next sequence and reshuffle based on env's new live/total
            live, total = env_live_total(self.env)
            self.shotgun = [True]*live + [False]*(total - live)
            random.shuffle(self.shotgun)
            # Notify engines of reload if they expose it
            for i, eng in enumerate(self.engines):
                if hasattr(eng, "on_reload") and callable(eng.on_reload):
                    try:
                        eng.on_reload(self.env)
                    except Exception:
                        pass

        # Callbacks for learning/heuristics (best-effort)
        for i, eng in enumerate(self.engines):
            if eng is None:
                continue
            try:
                if i == player and hasattr(eng, "on_own_move"):
                    eng.on_own_move(move, res_pub)
                elif i != player and hasattr(eng, "on_opponent_move"):
                    eng.on_opponent_move(move, res_pub)
            except Exception:
                pass

        self.history.append({
            "player": player,
            "move": move,
            "ad_target": ad_target,
            "shotgun_left": len(self.shotgun),
            "res_pub": str(res_pub),
            "res_priv": str(res_priv),
        })

    def auto_advance_until_human_or_over(self, max_steps: int = 999):
        steps = 0
        while steps < max_steps and self.winner() is None:
            cur = env_current_turn(self.env)
            eng = self.engines[cur]
            if eng is None:
                break  # stop at human
            # Engine chooses
            try:
                move, ad_target = eng.choice(self.env)
            except Exception:
                # Fallback random choice
                mv = self.legal_moves()
                if not mv:
                    break
                ad_target, move = random.choice(mv)
                ad_target = None if ad_target == cur else ad_target
            # Apply
            self.do_move(move, ad_target)
            steps += 1

# ----------------------------- HTML ------------------------------------

PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Buckshot Roulette – Visual Simulator</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: #0b0f14; color: #e6edf3; }
    header { padding: 16px 24px; border-bottom: 1px solid #1f2a35; display:flex; gap:16px; align-items:center; }
    h1 { font-size: 20px; margin: 0; }
    main { padding: 24px; max-width: 1100px; margin: 0 auto; }
    .card { background: #10161d; border: 1px solid #1f2a35; border-radius: 14px; padding: 16px; }
    .row { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }
    .col-6 { grid-column: span 6; }
    .col-12 { grid-column: span 12; }
    .col-3 { grid-column: span 3; }
    label { font-size: 12px; opacity: 0.8; display:block; margin-bottom: 6px; }
    input, select { width: 100%; padding: 10px 12px; border-radius: 10px; background:#0b0f14; color:#e6edf3; border:1px solid #233040; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #2b3b4f; background:#0b61b2; color:#fff; cursor:pointer; }
    button.secondary { background:#0b0f14; }
    .players { display:grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap:14px; }
    .player { padding: 12px; border:1px solid #253343; border-radius: 12px; background: #0e141b; }
    .tag { font-size: 11px; opacity: 0.8; padding:2px 6px; border-radius: 6px; border:1px solid #33475e; display:inline-block; }
    .muted { opacity: 0.75; }
    .status { display:flex; gap:6px; flex-wrap:wrap; }
    .shotgun { font-feature-settings: "tnum"; letter-spacing: 0.02em; }
    ul.history { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:8px; }
    ul.history li { padding:8px 10px; border:1px dashed #2c3f55; border-radius:10px; background:#0b0f14; }
    table { width:100%; border-collapse: collapse; }
    th, td { text-align:left; padding:8px 10px; border-bottom:1px solid #243445; }
    .grid-2 { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  </style>
</head>
<body>
<header>
  <h1>🧨 Buckshot Roulette – Visual Simulator</h1>
  <div class="muted">2–4 players • choose controllers • simulate or play</div>
</header>
<main>
  {% with msgs = get_flashed_messages() %}
    {% if msgs %}
      <div class="card" style="margin-bottom:16px;">
        {% for m in msgs %}<div>{{ m }}</div>{% endfor %}
      </div>
    {% endif %}
  {% endwith %}

  <div class="card" style="margin-bottom:16px;">
    <form method="post" action="{{ url_for('start') }}">
      <div class="row">
        <div class="col-3">
          <label>Player Count</label>
          <select name="player_count">
            {% for n in [2,3,4] %}
              <option value="{{n}}" {% if n == player_count %}selected{% endif %}>{{n}}</option>
            {% endfor %}
          </select>
        </div>
        <div class="col-3">
          <label>Starting Charges (per player)</label>
          <input name="charge_count" type="number" min="1" max="10" value="{{ charge_count or 4 }}">
        </div>
      </div>
      <div class="row" style="margin-top:12px;">
        {% for i in range(0, 4) %}
          <div class="col-3">
            <label>Player {{ i }}</label>
            <select name="p{{i}}">
              {% for choice in choices %}
                <option value="{{ choice }}" {% if (player_kinds|length>i and player_kinds[i]==choice) %}selected{% endif %}>{{ choice }}</option>
              {% endfor %}
            </select>
          </div>
        {% endfor %}
      </div>
      <div style="margin-top:12px; display:flex; gap:8px;">
        <button type="submit">Start / Reset</button>
        {% if game_id %}
          <a href="{{ url_for('game') }}"><button class="secondary" type="button">Refresh Board</button></a>
        {% endif %}
      </div>
    </form>
  </div>

  {% if game_id %}
  <div class="row">
    <div class="col-6">
      <div class="card">
        <h3 style="margin-top:0;">Board</h3>
        <div class="players">
          {% for i in range(0, session_obj.player_count) %}
            <div class="player">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>Player {{ i }}</strong>
                <span class="tag">{{ session_obj.player_kinds[i] }}</span>
              </div>
              <div class="muted" style="margin-top:4px;">Turn: {% if i == cur_turn %}➡️{% else %}&nbsp;{% endif %}</div>
              <div style="margin-top:8px;">
                <div class="grid-2">
                  <div>
                    <div class="muted">Items</div>
                    <div style="font-size:12px;">
                      {% for k,v in items[i].items() %}
                        {% if v>0 %}<div>{{ k }}: {{ v }}</div>{% endif %}
                      {% endfor %}
                      {% if items[i]|length == 0 %}<div class="muted">none</div>{% endif %}
                    </div>
                  </div>
                  <div>
                    <div class="muted">Status</div>
                    <div class="status">
                      {% for s in statuses %}<span class="tag">{{ s.split('.')[-1] }}</span>{% endfor %}
                      {% if statuses|length == 0 %}<div class="muted">—</div>{% endif %}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>
        <div style="margin-top:12px; display:flex; justify-content:space-between;">
          <div class="shotgun">🔩 Shotgun shells remaining (unknown order): <strong>{{ shotgun_len }}</strong></div>
          <div class="muted">Sequence live/total (internal): {{ live }}/{{ total }}</div>
        </div>
        {% if winner is not none %}
          <div style="margin-top:12px; font-size:18px;">🏆 Winner: Player {{ winner }} ({{ session_obj.player_kinds[winner] }})</div>
        {% endif %}
      </div>

      <div class="card" style="margin-top:16px;">
        <h3 style="margin-top:0;">Move History</h3>
        <ul class="history">
          {% for h in session_obj.history|reverse %}
            <li><strong>P{{h.player}}</strong> → <code>{{h.move}}</code>{% if h.ad_target is not none %} (target {{h.ad_target}}){% endif %}
            <span class="muted">• shells: {{h.shotgun_left}}</span></li>
          {% endfor %}
          {% if session_obj.history|length == 0 %}
            <li class="muted">No moves yet.</li>
          {% endif %}
        </ul>
      </div>
    </div>

    <div class="col-6">
      <div class="card">
        <h3 style="margin-top:0;">Actions</h3>
        {% if winner is none %}
          {% if cur_is_human %}
            <form method="post" action="{{ url_for('human_move') }}">
              <label>Choose your move</label>
              <select name="selection">
                {% for ad_target, move in legal_moves %}
                  <option value="{{ (ad_target if ad_target is not none else '') ~ '|' ~ move }}">
                    {{ move }}{% if ad_target is not none %} → target {{ ad_target }}{% endif %}
                  </option>
                {% endfor %}
              </select>
              <div style="margin-top:10px; display:flex; gap:8px;">
                <button type="submit">Play Move</button>
                <a href="{{ url_for('skip_to_ai') }}"><button class="secondary" type="button">Let AI play until my turn</button></a>
              </div>
            </form>
          {% else %}
            <div class="muted">It's an AI's turn. You can advance:</div>
            <div style="margin-top:8px; display:flex; gap:8px;">
              <a href="{{ url_for('step_ai') }}"><button>Step 1 move</button></a>
              <a href="{{ url_for('skip_to_human') }}"><button class="secondary">Run until human/finish</button></a>
            </div>
          {% endif %}
        {% else %}
          <div class="muted">Game over.</div>
        {% endif %}
      </div>

      {% if no_humans %}
      <div class="card" style="margin-top:16px;">
        <h3 style="margin-top:0;">Batch Simulation (no human players)</h3>
        <form method="post" action="{{ url_for('batch') }}">
          <div class="row">
            <div class="col-6">
              <label>Trials (N)</label>
              <input name="trials" type="number" min="1" max="100000" value="200">
            </div>
          </div>
          <div style="margin-top:10px;">
            <button type="submit">Run Simulation</button>
          </div>
        </form>
        {% if batch_results %}
          <div style="margin-top:12px;">
            <table>
              <thead><tr><th>Controller</th><th>Wins</th><th>Win Rate</th></tr></thead>
              <tbody>
                {% for name, wins, rate in batch_results %}
                  <tr><td>{{ name }}</td><td>{{ wins }}</td><td>{{ '%0.2f%%' % (rate*100) }}</td></tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% endif %}
      </div>
      {% endif %}
    </div>
  </div>
  {% endif %}
</main>
</body>
</html>
"""

# ----------------------------- Routes ----------------------------------

def _get_or_create_session(player_count: int, charge_count: int, player_kinds: List[str]) -> str:
    game_id = os.urandom(8).hex()
    GAMES[game_id] = GameSession(game_id, player_kinds[:player_count], charge_count)
    session["game_id"] = game_id
    return game_id

def _cur_session() -> Optional[GameSession]:
    gid = session.get("game_id")
    if not gid:
        return None
    return GAMES.get(gid)

@app.route("/", methods=["GET"])
def home():
    # Default form state
    default_kinds = ["Human", "Dealer"] + ["Random", "Random"]
    ctx = {
        "choices": CHOICES,
        "player_count": 2,
        "charge_count": 4,
        "player_kinds": default_kinds,
        "game_id": session.get("game_id"),
        "session_obj": _cur_session(),
        "batch_results": None,
    }
    return render_template_string(PAGE, **ctx)

@app.route("/start", methods=["POST"])
def start():
    player_count = int(request.form.get("player_count", 2))
    charge_count = int(request.form.get("charge_count", 4))
    player_kinds = [request.form.get(f"p{i}", "Random") for i in range(4)]

    # Validate at least one controller per actual count
    kinds = player_kinds[:player_count]
    if any(k is None for k in kinds):
        flash("Invalid player selection.")
        return redirect(url_for("home"))

    _get_or_create_session(player_count, charge_count, kinds)
    return redirect(url_for("game"))

@app.route("/game", methods=["GET"])
def game():
    gs = _cur_session()
    if gs is None:
        return redirect(url_for("home"))

    # Optionally fast-forward AI turns so you see something
    if len(gs.humans) == 0:
        gs.auto_advance_until_human_or_over(max_steps=1000)

    live, total = env_live_total(gs.env)
    ctx = {
        "choices": CHOICES,
        "player_count": gs.player_count,
        "charge_count": gs.start_charges,
        "player_kinds": gs.player_kinds + ["Random"]*(4-gs.player_count),
        "game_id": gs.game_id,
        "session_obj": gs,
        "cur_turn": env_current_turn(gs.env),
        "cur_is_human": env_current_turn(gs.env) in gs.humans and gs.winner() is None,
        "items": [gs.items_for(i) for i in range(gs.player_count)],
        "statuses": gs.status_flags(),
        "winner": gs.winner(),
        "legal_moves": gs.legal_moves(),
        "shotgun_len": len(gs.shotgun),
        "live": live, "total": total,
        "no_humans": len(gs.humans) == 0,
        "batch_results": None,
    }
    return render_template_string(PAGE, **ctx)

@app.route("/move", methods=["POST"])
def human_move():
    gs = _cur_session()
    if gs is None:
        return redirect(url_for("home"))
    if env_current_turn(gs.env) not in gs.humans:
        flash("It's not a human's turn.")
        return redirect(url_for("game"))
    sel = request.form.get("selection", "")
    try:
        ad_str, move = sel.split("|", 1)
        ad_target = int(ad_str) if ad_str.strip() != "" else None
    except ValueError:
        move = sel
        ad_target = None
    gs.do_move(move, ad_target)
    return redirect(url_for("game"))

@app.route("/step", methods=["GET"])
def step_ai():
    gs = _cur_session()
    if gs is None:
        return redirect(url_for("home"))
    # Only step if it's an AI turn
    if env_current_turn(gs.env) not in gs.humans and gs.winner() is None:
        gs.auto_advance_until_human_or_over(max_steps=1)
    return redirect(url_for("game"))

@app.route("/skip_to_human", methods=["GET"])
def skip_to_human():
    gs = _cur_session()
    if gs is None:
        return redirect(url_for("home"))
    gs.auto_advance_until_human_or_over(max_steps=1000)
    return redirect(url_for("game"))

@app.route("/skip_to_ai", methods=["GET"])
def skip_to_ai():
    gs = _cur_session()
    if gs is None:
        return redirect(url_for("home"))
    # Run AI until it becomes AI's turn (i.e., immediately step once if it's AI turn)
    if env_current_turn(gs.env) not in gs.humans and gs.winner() is None:
        gs.auto_advance_until_human_or_over(max_steps=1)
    return redirect(url_for("game"))

@app.route("/batch", methods=["POST"])
def batch():
    gs = _cur_session()
    if gs is None:
        flash("No active game.")
        return redirect(url_for("home"))
    if len(gs.humans) > 0:
        flash("Batch simulation is only available when no Human is selected.")
        return redirect(url_for("game"))
    try:
        N = int(request.form.get("trials", "200"))
    except Exception:
        N = 200

    wins_by_label: Dict[str, int] = {k: 0 for k in gs.player_kinds}
    # Run N independent games with same controller assignment
    for _ in range(N):
        trial = GameSession("trial", gs.player_kinds, gs.start_charges)
        trial.auto_advance_until_human_or_over(max_steps=100000)
        w = trial.winner()
        if w is not None and 0 <= w < trial.player_count:
            wins_by_label[trial.player_kinds[w]] += 1

    results = []
    for name, wins in wins_by_label.items():
        # compute rate relative to number of players with that label (e.g., if duplicate models)
        # Here we just report raw rate out of N trials.
        results.append((name, wins, wins / max(1, N)))

    # Sort by wins desc
    results.sort(key=lambda x: x[1], reverse=True)

    live, total = env_live_total(gs.env)
    ctx = {
        "choices": CHOICES,
        "player_count": gs.player_count,
        "charge_count": gs.start_charges,
        "player_kinds": gs.player_kinds + ["Random"]*(4-gs.player_count),
        "game_id": gs.game_id,
        "session_obj": gs,
        "cur_turn": env_current_turn(gs.env),
        "cur_is_human": env_current_turn(gs.env) in gs.humans and gs.winner() is None,
        "items": [gs.items_for(i) for i in range(gs.player_count)],
        "statuses": gs.status_flags(),
        "winner": gs.winner(),
        "legal_moves": gs.legal_moves(),
        "shotgun_len": len(gs.shotgun),
        "live": live, "total": total,
        "no_humans": len(gs.humans) == 0,
        "batch_results": results,
    }
    return render_template_string(PAGE, **ctx)

if __name__ == "__main__":
    app.run(debug=True)
