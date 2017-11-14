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

import gym

from gym import spaces
from gym.utils import seeding

class TreeMazeEnv(gym.Env):
    """ T-maze with several branches

        This is discrete environment that looks like a tree :

                      ---- 4
                ---- 2
               |      ---- 5
        I ---- 1
               |      ---- 6
                ---- 3
                      ---- 7

        The maze consists of corridors of length S. After each
        corridor is a branch where the agent can go up or down. After going up
        or down, the agent must go right for S steps before reaching the next
        branching point. The tree has a height H. At each episode, a leaf L is
        choosen randomly as a goal state. Reaching the goal gives a reward of 10.
        Each move goves a reward of -0.1.

        The agent is able to observe its position in the current corridor (distance
        between the agent and the next branch), along with an additional binary
        observation, o. During the first H time-steps, o = whether to go up at the
        i-th branch.

        The goal of the environment is to have the agent learn to remember information
        during the first H time-steps, then use it to navigate the maze.
    """

    def __init__(self, size=5, height=3):
        self.size = size
        self.height = height

        self.action_space = spaces.Discrete(3)                              # Up, Down, Right
        self.observation_space = spaces.Discrete(2 * size * (height + 1))   # Bit times cell in the corridor times corridor id
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        # Move up or down if possible
        if self._before_branch == 0 and action in [0, 1]:
            self._done.append(action)           # Direction the agent went
            self._before_branch = self.size - 1 # New corridor

        # Move right if possible
        if self._before_branch != 0 and action == 2:
            self._before_branch -= 1

        # Compute reward
        if self._before_branch == 0 and len(self._done) == len(self._goal):
            # The agent has just reached the end of the corridor leading to a leaf
            reward = 10.0 if self._done == self._goal else 0.0
            done = True
        else:
            # Other action
            reward = -0.1
            done = False

        # Produce a binary observation depending on the current time-step
        if self._timestep < self.height:
            bit = self._goal[self._timestep]
        else:
            bit = 0

        self._timestep += 1
        state = len(self._done) * self.size * 2 + self._before_branch * 2 + bit

        return state, reward, done or self._timestep > 1000, {}

    def _reset(self):
        #goals = range(2 ** self.height)
        #probas = np.arange(2 ** self.height, 0, -1)     # Unnormalized probas : 8, 7, 6, 5, 4, 3, 2, 1
        #probas = probas / probas.sum()
        #goal = np.random.choice(goals, p=probas)

        #self._goal = [(goal >> i) % 2 for i in range(self.height-1, -1, -1)]

        # Choose a random goal state, identified by the sequence of up/down to
        # perform to reach it
        self._goal = list(self.np_random.random_integers(0, 1, self.height))
        self._done = []
        self._before_branch = self.size - 1
        self._timestep = 1

        return self._before_branch * 2 + self._goal[0]      # First observation and first hint
