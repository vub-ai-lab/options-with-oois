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

###############################################################################
# Expert top-level and option policies for our simulated robotic task.
# The "policy" function, at the end of this file, implements the
# actual policy.
#
# Everything before this function creates the options
# and their initiation sets. In experiments.sh, the initiation sets
# are defined using the --nexts argument, that contains a bunch of
# numbers. This does the same as what is done here, but in a less
# readable way.

TERMINALS = 2   # Number of terminals
NUM_ACTIONS = 3

# Four options go to the root, each with their "condition"
nexts = {}
subs = {}
options = []

for i in range(TERMINALS):
    options.append(('root', i, 'success'))      # Go to root, coming from i, after a success
    options.append(('root', i, 'failure'))      # Go to root, coming from i, after a failure

# Four options go to each goal, each of them can only be activated
# when its corresding condition is true
for i in range(len(options)):
    nexts[i] = [len(options) + NUM_ACTIONS + t for t in range(TERMINALS)]

    for t in range(TERMINALS):
        nexts[len(options)] = range(NUM_ACTIONS + t*2, NUM_ACTIONS + (t+1)*2)   # Go to goal from t-th terminal

        options.append((t, options[i]))                     # Go to terminal t after condition i

for i in range(len(options)):
    subs[i] = range(NUM_ACTIONS)  # Options may only execute primitive actions

num_options = len(options)
subs[-1] = [i + NUM_ACTIONS for i in range(num_options)]

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
    global num_options
    global NUM_ACTIONS

    # Probability vector
    probas = [0.0] * (2 * (NUM_ACTIONS + num_options))

    if option == -1:
        # Top-level policy
        for i, desc in enumerate(options):
            if desc[0] == 'root' and desc[2] == 'success':
                # Go to root after success
                probas[i+NUM_ACTIONS] = 1.0 if state == 1 else 0.0
            elif desc[0] == 'root' and desc[2] == 'failure':
                # Go to root after failure
                probas[i+NUM_ACTIONS] = 1.0 if state == 0 else 0.0
            elif desc[0] == 0 and desc[1] in [('root', 0, 'success'), ('root', 1, 'failure')]:
                # Go to terminal 0 if 0 succeeds or 1 fails
                probas[i+NUM_ACTIONS] = 1.0
            elif desc[0] == 1 and desc[1] in [('root', 1, 'success'), ('root', 0, 'failure')]:
                # Go to terminal 1 if 1 succeeeds or 0 fails
                probas[i+NUM_ACTIONS] = 1.0
    else:
        # The options themselves go to the corresponding location then terminate
        goal = options[option][0]

        if goal == 'root':
            g = 0
        else:
            g = goal + 1

        probas[g + NUM_ACTIONS + num_options] = 1.0

    return probas
