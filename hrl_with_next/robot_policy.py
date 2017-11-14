# This file is part of OOIs.
#
# Copyright 2017, Vrije Universiteit Brussel (http://vub.ac.be)
#
# OOIs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OOIs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

# Policies for our robotic task. This script is quite complicated,
# see robot_policy_simple.py for a simpler version, with more comments.

from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import math
import sys
import random

TOP = 2         # Number of objects in each terminal
ROUNDS = 2      # Number of times terminals are emptied before ending the episode
TERMINALS = 2   # Number of terminals

# Four options go to the root, each with their "condition"
num_actions = 4
nexts = {}
subs = {}
options = []

for i in range(TERMINALS):
    options.append(('root', i, 'success'))      # Go to root, coming from i, after a success
    options.append(('root', i, 'failure'))      # Go to root, coming from i, after a failure

# Four options go to each goal, each of them can only be activated
# when its corresding condition is true
for i in range(len(options)):
    nexts[i] = [len(options) + 4 + t for t in range(TERMINALS)]

    for t in range(TERMINALS):
        nexts[len(options)] = range(4 + t*2, 4 + (t+1)*2)   # Go to goal from t-th terminal

        options.append((t, options[i]))                     # Go to terminal t after condition i

for i in range(len(options)):
    subs[i] = range(4)  # Options may only execute primitive actions

num_options = len(options)
subs[-1] = [i + 4 for i in range(num_options)]

for i in range(len(options)):
    print('Nexts of', i, options[i], 'are', nexts[i], [options[j-4] for j in nexts[i]])

# Qt code
app = QApplication(sys.argv)
wins = []
terminals = [random.randint(TOP, TOP*2)] * TERMINALS
last_reward = (0.0, False)
rounds = random.randint(ROUNDS, ROUNDS+1)
pixmaps = (
    (QPixmap('qrcodes/green_1.png'), QPixmap('qrcodes/green_2.png')),
    (QPixmap('qrcodes/blue_1.png'), QPixmap('qrcodes/blue_2.png')),
)

for i in range(TERMINALS):
    # There are two main windows, one for terminal A and one for terminal B
    win = QLabel()
    wins.append(win)

    win.setWindowFlags(Qt.FramelessWindowHint)
    win.resize(800, 800)
    win.setPixmap(pixmaps[i][0])
    win.show()

# Let the user move the windows before starting the experiment
QMessageBox.information(None, 'Startup', 'The experiment will now start')

def read_qrcode(state):
    """ Extract a QR-code index (if any visible) from a state
    """
    qrcodes = state[9:-1]

    if qrcodes.max() > 0.0:
        return qrcodes.argmax()
    else:
        return None

def shaping(option, old_state, new_state):
    """ Reward to be given to <option> when reaching <new_state> from <old_state>
    """
    global last_reward
    return last_reward

def update_qrcodes():
    global wins
    global terminals

    # Update the QR-Codes of the terminals when the root is reached
    for i in range(TERMINALS):
        wins[i].setPixmap(pixmaps[i][0 if terminals[i] > 0 else 1])
        app.processEvents()

def follow_color(option, probas, reading, qrcode, terminal):
    """ Select actions to keep the desired reading (a color) in front of the camera
    """
    global num_actions
    global terminals
    global last_reward
    global rounds

    x, y, area = reading

    if x < 0.4 or area < 0.001:
        # Object on the left, or too far
        probas[2] = 1.0
    elif x > 0.6:
        # Object on the right
        probas[1] = 1.0
    else:
        # Object in front
        if area > 0.01 and qrcode is not None:
            # A QR-Code is visible (and close enough), terminate the current option (while stopping the robot)
            probas[0 + num_options + num_actions] = 1.0
            
            # When a terminal is reached, decrease the amount of elements it contains
            if terminal != -1:
                if qrcode == 1:
                    # Terminal full, remove one element from it
                    terminals[terminal] -= 1
                    last_reward = (1.0, False)
                else:
                    # Terminal empty, refill the others
                    rounds -= 1
                    done = (rounds == 0)

                    for i in range(TERMINALS):
                        if done or i != terminal:
                            terminals[i] = random.randint(TOP, TOP*2)

                    if done:
                        # Reset for next episode
                        rounds = random.randint(ROUNDS, ROUNDS+1)
                        update_qrcodes()

                    last_reward = (-1.0, done)
            else:
                update_qrcodes()
        else:
            # Go towards the color (hoping that the QR-Code will appear)
            probas[0] = 1.0

def policy(state, option):
    """ Return a probability distribution over extended options for a state-option

        - state: current state
        - option: current option, -1 if top-level policy

        The return value must be a list of floats. If the environment contains
        A actions and O options, then N = A + O, and the list must contain 2*N
        elements. The first N elements correspond to actions/options without end,
        the others with end.

        If this function returns None, the agent falls back to predicting actions
        using the neural network.
    """
    global options
    global num_actions
    global last_reward

    # Probability vector
    probas = [0.0] * (2 * (num_actions + num_options))

    # Any QR-Code visible?
    qrcodes = state[9:-1]
    qrcode = None

    if qrcodes.max() > 0.0:
        qrcode = qrcodes.argmax()

    # Process options
    last_reward = (0.0, None)

    if option == -1:
        # Rules
        for i, desc in enumerate(options):
            if desc[0] == 'root' and desc[2] == 'success':
                # Go to root after success
                probas[i+4] = 1.0 if qrcode == 1 else 0.0
            elif desc[0] == 'root' and desc[2] == 'failure':
                # Go to root after failure
                probas[i+4] = 1.0 if qrcode == 2 else 0.0
            elif desc[0] == 0 and desc[1] in [('root', 0, 'success'), ('root', 1, 'failure')]:
                # Go to terminal 0 if 0 succeeds or 1 fails
                probas[i+4] = 1.0
            elif desc[0] == 1 and desc[1] in [('root', 1, 'success'), ('root', 0, 'failure')]:
                # Go to terminal 1 if 1 succeeeds or 0 fails
                probas[i+4] = 1.0
    else:
        # Go to the color that corresponds to the goal of the option
        goal = options[option][0]
        
        if goal == 'root':
            print('RED')
            follow_color(option, probas, state[0:3], qrcode, -1)
        elif goal == 0:
            follow_color(option, probas, state[3:6], qrcode, 0)
            print('GREEN')
        elif goal == 1:
            follow_color(option, probas, state[6:9], qrcode, 1)
            print('BLUE')

    return probas
