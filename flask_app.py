from flask import Flask, render_template, jsonify, request
from cli_game import Board, generate_binary_numbers
from stat_engine import StatEngine, UnCheatEngine, NewEngine
import time

app = Flask(__name__)
BOARD = Board(5)
OPPONENT = StatEngine(1)
turn_id = 0
last_action = (-1, "")
update_lock = False

@app.route("/")
def index():
    return render_template("game_page.html")

@app.route("/move_page")
def mp():
    return render_template("move_page.html")

@app.route("/action")
def act():
    action = request.args.get("action")
    if not do_action(action):
        return str(False)
    while BOARD.current_turn == 1 and BOARD.winner() == None:
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
    time.sleep(6)
    while update_lock:
        time.sleep(0.5)
    return True

@app.route('/modify_board', methods=['POST'])
def modify_board():
    data = request.form

    # Parsing form data to create a Board instance
    charge_count = int(data['charge_count'])
    total_rounds = int(data['total_rounds']) if data['total_rounds'] else None
    board = Board(charge_count, total_rounds)

    board.charges = [int(data['p1_charges']), int(data['p2_charges'])]
    board._shotgun = [True] * int(data['live_rounds']) + [False] * (int(data['total_rounds']) - int(data['live_rounds']))
    
    # Setting attributes based on form data
    board.charges = [int(data.get('p1_charges', 0)), int(data.get('p2_charges', 0))]
    board.current_turn = int(data['current_turn'])

    # Setting items for each player
    for i in range(2):
        player_items = {}
        for item in Board.POSSIBLE_ITEMS:
            item_key = f'p{i+1}_{item}'
            player_items[item] = int(data.get(item_key, 0))
        board.items[i] = player_items

    # Setting active items
    active_items = {}
    for item in Board.POSSIBLE_ITEMS:
        item_key = f'active_{item}'
        active_items[item] = int(data.get(item_key, 0))
    board._active_items = active_items

    # Setting other attributes
    board._skip_next = data.get('skip_next') == 'true'
    #board.chamber_public = 
    cham_pub = data.get('chamber_public', 'undefined').lower()
    if cham_pub == 'true':
        board.chamber_public = True
    elif cham_pub == 'false':
        board.chamber_public = False
    else:
        board.chamber_public = None
    
    # Add any additional logic here to handle board object

    output = ""
    se = StatEngine(board.current_turn)
    output += f"Stat Engine: {se.best_move(board)}<br>"
    
    uce = UnCheatEngine(board.current_turn)
    output += f"UnCheat Engine: {uce.best_move(board)}<br>"
    
    ne = NewEngine(board.current_turn)
    output += f"New Engine: {ne.best_move(board)}<br>"
    
    return output  # Or render a template with the board information
    

if __name__ == "__main__":
    app.run(port=5000)