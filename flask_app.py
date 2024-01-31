from flask import Flask, render_template, jsonify, request
from cli_game import Board
from stat_engine import DealerEngine
import time

app = Flask(__name__)
BOARD = Board(5)
OPPONENT = DealerEngine(1)
turn_id = 0
last_action = (-1, "")
update_lock = False

@app.route("/")
def index():
    return render_template("game_page.html")

@app.route("/action")
def act():
    action = request.args.get("action")
    if not do_action(action):
        return str(False)
    while BOARD.current_turn == 1:
        do_action(OPPONENT.best_move(BOARD))
    return str(True)

@app.route("/unlock")
def unlock():
    global update_lock
    update_lock = False
    print("------------ UNLOCK -------------")
    return str(lock)

@app.route("/lock")
def lock():
    global update_lock
    update_lock = True
    print("------------ LOCK -------------")
    return str(lock)

@app.route("/data")
def board_data():
    global BOARD
    out_dict = {
        'player_items': BOARD.p1_items,
        'op_items': BOARD.p2_items,
        'active_items': BOARD._active_items,
        'shotgun': BOARD._shotgun,
        'shotgun_info': BOARD.shotgun_info(),
        'max_charges': BOARD.max_charges,
        'charges': BOARD.charges,
        'turn': BOARD.current_turn,
        'known_shell': BOARD.chamber_public,
        'moves': BOARD.moves(),
        'last_action': last_action,
        'lock': update_lock,
        'turn_id':turn_id
    }
    return jsonify(out_dict)

def do_action(action):    
    global turn_id, last_action, BOARD, update_lock
    if action not in BOARD.moves():
        return False
    update_lock = True
    print("------------ LOCK -------------")   
    last_action = (BOARD.current_turn, action, BOARD.make_move(action)) 
    turn_id += 1
    time.sleep(10)
    while update_lock:
        time.sleep(0.5)
    return True
    
    

if __name__ == "__main__":
    app.run(port=5000)