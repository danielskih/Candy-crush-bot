from itertools import product, chain
from IPython.display import Image, display
import numpy as np
import matplotlib.pyplot as plt

from time import sleep
from PIL import Image as Imag
import pyautogui
import pandas as pd
from copy import deepcopy

# Legal moves list
candy_all = ['blue', 'red', 'green', 'violet', 'orange']


def closeButtonLookup():
    """Looks for close button and presses it returns False if close button is not found, else - True"""
    if bool(pyautogui.locateCenterOnScreen('Images/btn_close.png', confidence=0.7)):
        x, y = pyautogui.locateCenterOnScreen('Images/btn_close.png', confidence=0.7)
        pyautogui.click(x / 2, y / 2, duration=0.5, button='right')
        print('OK,I am out of here!')
        return True
    return False


class GameState:
    """Game state, board = dataframe, 'frm' and 'at' = lists of row/colum values of origin and current position"""

    def __init__(self, board, parent=None, frm=None, at=None):
        self.board = board
        self.parent = parent
        self.frm = frm
        self.at = at
        self.matches_vert = list()
        self.matches_horyz = list()
        self.children = list()


class Match:
    """Match contains info on size, position and type of candy"""

    def __init__(self, typ, pos):
        self.size = pos.shape[0]
        self.pos = pos
        self.typ = typ


def find_match(selected):
    """Takes Series as an input, returns index of a match3 in the series if there is, else: False"""
    match = [i for i in selected.index]
    diff = [1] + [i - j for j, i in zip(match[:-1], match[1:])]
    diff = np.absolute(np.array(diff))
    mask = diff == 1
    if selected[mask].size >= 3:
        return list(selected[mask].index)
    return


def move_to(global_board, fr, to):
    """Takes a Dataframe, and two lists of row/col values, swaps values in specified cells.
        Outputs resulting Data frame """
    local_board = global_board.copy()
    local_board.loc[fr[0], fr[1]], local_board.loc[to[0], to[1]] = local_board.loc[to[0], to[1]], local_board.loc[
        fr[0], fr[1]]

    return local_board


def generate_children(global_board, pos):
    """Generate valid moves from a given tile, pos=list of row/col values
    return list of gamestates"""

    positions = np.array(pos)
    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    moves = np.add(positions, directions)

    # Filter the values outside of the local_board
    moves = moves[(0 <= moves[:, 0]) & (moves[:, 0] < board_w)]
    moves = moves[(0 <= moves[:, 1]) & (moves[:, 1] < board_h)]

    # filter the candy_all moves and save them in a children nodes attribute
    children = []
    for move in moves:
        if tuple(move) in legal_tiles:
            local_board = move_to(global_board, pos, move)
            display(local_board)
            m = GameState(local_board, None, frm=pos, at=move)
            print(id(m.board))
            children.append(m)

    return children


## Create board

#### Get input for board height and width.

# board_w = int(input('Borad width is:  '))
board_w = 5

# board_h = int(input('Borad width is: '))
board_h = 5

board = pd.DataFrame('0', index=range(board_h), columns=range(board_w))
Root = GameState(board.copy())


def select_type(typ, board, ind=board.index, col=board.columns):
    """Select all instances of a given type in a Series = board.iloc[index, column]
    where board is Pandas DataFrame. When calling the function one of the kwargs MUST be given in a call"""

    selected = board.loc[ind, col][board.loc[ind, col] == typ]
    if selected.size >= 3:
        return selected

    return pd.Series(dtype='object')


all_tiles = list(product(range(5), range(5)))
# all_tiles

#### Add some candy manually for testing purpouse

Root.board.loc[1, [1, 2, 4]] = 'red'
Root.board.loc[2, [1, 2, 3]] = 'blue'
Root.board.loc[[0, 1, 2], 4] = 'red'
Root.board.loc[3, 1] = 'blue'
Root.board.loc[[1, 2, 3, 4], 0] = 'orange'
# Root.board.loc[2,0], Root.board.loc[2,1]=Root.board.loc[2,1],Root.board.loc[2,0]

Root.board = move_to(Root.board, [2, 0], [2, 1])

legal_tiles = [tile for tile in all_tiles if Root.board.iloc[tile[0], tile[1]] in candy_all]


## Look for the matches

candidates = list()

for tile in legal_tiles:
    print(tile)
    #     display(generate_children(Root.board.copy(), tile)[0].board)
    candidates.extend(generate_children(Root.board.copy(), tile))


### Look for matches in the resulting boards.

# Column - wise.

for candidate in candidates:
    #     TODO: maybe convert that into Dataframe right away and prform the operations on columns

    # Get a type of the match we are looking for
    typ = candidate.board.iloc[candidate.frm[0], candidate.frm[1]]

    # Search the column for the values of the given type
    res_vert = {i: select_type(typ, candidate.board, col=i) for i in candidate.board.columns
                if select_type(typ, candidate.board, col=i).any()}

    # Now look for actual matches in the result of the search
    for key, val in res_vert.items():
        match = find_match(val)
        if match:
            candidate.matches_vert.append(Match(val.all(), candidate.board.loc[match, key]))

    # And row-wise
    res_horyz = {i: select_type(typ, candidate.board, ind=i) for i in candidate.board.index
                 if select_type(typ, candidate.board, ind=i).any()}

    for key, val in res_horyz.items():
        match = find_match(val)
        if match:
            candidate.matches_horyz.append(Match(val.all(), candidate.board.loc[key, match]))

print([candidate.matches_vert for candidate in candidates if candidate.matches_vert])

print(pd.DataFrame([i.matches_vert[0].pos for i in candidates if i.matches_vert]))

print(pd.DataFrame([cand.matches_horyz[0].pos for cand in candidates if cand.matches_horyz]))

"""collapse
matches:
- create
empty
cells
where
matches
are
- move
remaining
candies
to
fill
the
space.

evaluate
children

Prune
the
children.Keep
boards
with matches, delete the rest.

# screen size
pyautogui.size().height

tile_w = 500 / 7
tile_h = 524 / 8

offset_x = tile_w / 2
offset_y = tile_h / 2

image = Imag.open("Images/game_board.png")
image_array = np.array(image)
imgplot = plt.imshow(image_array)

blue = image_array[:133, :145]

imgplot = plt.imshow(blue)

a = np.mean(blue, axis=1)
b = np.ones(133)
np.matmul(a.T, b) / 133

orange = image_array[:133, 145 * 2:145 * 3]
plt.imshow(orange)

# Get mean RGB values
c = np.mean(orange, axis=1)
d = np.ones(133)
np.matmul(c.T, d) / 133

round(image_array.shape[0] / 5)"""