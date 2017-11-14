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

    # 3 primitive actions, 8 options, N=11
    probas = [0.0] * (2 * 11)
    corridor = state // 10
    distance = (state % 10) // 2
    bit = state % 2

    if option == -1:
        # Top-level option, let's learn it
        return None

        if corridor == 0 and distance == 4:
            # First bit, choice between 0 and 4
            choice0 = [0]
            choice1 = [4]
        elif corridor == 0 and distance == 3:
            # Second bit, choice between (0, 4) and (2, 6)
            choice0 = [0, 4]
            choice1 = [2, 6]
        else:
            # Third bit, choice between (0, 2, 4, 6) and (1, 3, 5, 7)
            choice0 = [0, 2, 4, 6]
            choice1 = [1, 3, 5, 7]

        if bit == 0:
            choice = choice0
        else:
            choice = choice1

        # 8 options, ignore primitive actions
        for i in range(8):
            if i in choice:
                probas[3 + i] = 1.0
    else:
        # Each option goes to a different goal
        if corridor == 0 and distance >= 3:
            # Move right and terminate
            next_option = 2 + 11
        elif distance > 0:
            # Move right
            next_option = 2
        else:
            # Move 0 or 1, depending on option and corridor index
            next_option = {
                0: [0, 0, 0],
                1: [0, 0, 1],
                2: [0, 1, 0],
                3: [0, 1, 1],
                4: [1, 0, 0],
                5: [1, 0, 1],
                6: [1, 1, 0],
                7: [1, 1, 1]
            }[option][corridor]

        probas[next_option] = 1.0

    return probas
