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

    # 3 primitive actions, 14 options, N=17
    probas = [0.0] * (2 * 17)
    corridor = state // 10
    distance = (state % 10) // 2
    bit = state % 2
    
    if option == -1:
        # Top-level option, let's learn it
        return None

        if corridor == 0 and distance == 4:
            f, t = (0, 1)   # First bit, distribution over options 0, 1
        elif corridor == 0 and distance == 3:
            f, t = (2, 5)   # Second bit, distribution over options 2, 3, 4, 5
        else:
            f, t = (6, 13)  # Third bit, distribution over options 6, 7, 8, 9,  10, 11, 12, 13

        # 14 options, ignore primitive actions
        for i in range(f, t + 1):
            if bit == 0:
                # Options selected when bit is 0 (Next allows to really select only one of them)
                if i in [0, 2, 4, 6, 8, 10, 12]:
                    probas[3 + i] = 1.0
            else:
                # Options corresponding to bit 1
                if i in [1, 3, 5, 7, 9, 11, 13]:
                    probas[3 + i] = 1.0
    elif option <= 5:
        # Options 0-5 go right and terminate
        probas[2 + 17] = 1.0
    else:
        # Options 6-13 each go to a different goal
        if distance > 0:
            # Move right
            next_option = 2
        else:
            # Move 0 or 1, depending on option and corridor index
            next_option = {
                6:  [0, 0, 0],
                7:  [0, 0, 1],
                8:  [0, 1, 0],
                9:  [0, 1, 1],
                10: [1, 0, 0],
                11: [1, 0, 1],
                12: [1, 1, 0],
                13: [1, 1, 1]
            }[option][corridor]
        
        probas[next_option] = 1.0
    
    return probas
