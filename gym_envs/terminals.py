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
import random

from gym import spaces
from gym.utils import seeding

TOP = 2         # Number of objects in each terminal
ROUNDS = 2      # Number of times terminals are emptied before ending the episode
TERMINALS = 2   # Number of terminals

LOC_ROOT = (448, 197)
LOC_T1 = (224, 109)
LOC_T2 = (224, 289)

class TerminalsEnv(gym.Env):
    """ Two terminals, A and B, contain objects that a robot has to carry to the
        root R. Terminals contain a finite number of objects, and the robot gets
        a negative reward when it goes to an empty one. This means that once a
        terminal is empty, the robot must remember to now go to the other one.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)                                  # Go to root, A or B
        self.observation_space = spaces.Discrete(2)                             # "Empty" (0) or "Full" (1), the root always returns 0
        self._render_enable = False
        self._app = None
        self._last_reward = 0.0
        self._last_action = 0

        self._seed()

    def _render(self, mode, close):
        self._render_enable = True

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        # Animate the current action if needed
        if self._render_enable:
            self.render_action(action)

        self._last_action = action

        if action == 0:
            # Going to the root does nothing
            self._last_reward = 0.0
            return 0, self._last_reward, False, {}
        else:
            terminal = action - 1

            if self.terminals[terminal] > 0:
                # There are objects left in the terminal, remove one
                self.terminals[terminal] -= 1
                self._last_reward = 2.0

                return 1, self._last_reward, False, {}
            else:
                # The terminal is empty, refill the other ones
                self.rounds -= 1
                self._last_reward = -2.0
                done = (self.rounds == 0)

                for i in range(TERMINALS):
                    if i != terminal:
                        self.terminals[i] = random.randint(TOP, TOP*2)

                return 0, self._last_reward, done, {}

    def _reset(self):
        self.terminals = [random.randint(TOP, TOP*2)] * TERMINALS
        self.rounds = random.randint(ROUNDS, ROUNDS+1)
        self._last_reward = 0.0

        if self._render_enable and self._app is not None:
            # Ensure that the agent starts at the root
            self._agent.move(*LOC_ROOT)

        return 0    # Implicitly start at the root with no information

    def render_action(self, action):
        from PyQt5.QtCore import QEventLoop, QPropertyAnimation, QEasingCurve, QPoint
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtWidgets import QApplication, QWidget, QLabel

        if self._app is None:
            # Create the window
            self._app = QApplication([])
            self._loop = QEventLoop(self._app)
            self._win = QLabel()
            self._win.resize(800, 600)
            self._win.setPixmap(QPixmap("qrcodes/terminals_bg.png"))
            self._win.show()

            # Two terminal indicators
            self._win_terminals = []

            for i in range(2):
                t = QLabel(self._win)
                t.resize(128, 128)
                t.setPixmap(QPixmap("qrcodes/packet.png"))
                t.move(56, [132, 339][i])
                t.show()

                self._win_terminals.append(t)

            # The agent
            self._agent = QLabel(self._win)
            self._agent.resize(128, 128)
            self._agent.move(*LOC_ROOT)
            self._agent.show()
            self._agent_on = QPixmap("qrcodes/agent_packet.png")
            self._agent_off = QPixmap("qrcodes/agent.png")

            # Reward indicator
            self._reward_label = QLabel(self._win)

        # Update the terminal indicators
        for i in range(2):
            self._win_terminals[i].setVisible(self.terminals[i] > 0)

        # Move the agent
        ann = QPropertyAnimation(self._agent, b"pos")
        ann.setStartValue(self._agent.pos())
        ann.setEndValue(QPoint(*([LOC_ROOT, LOC_T1, LOC_T2][action])))
        ann.setDuration(1000)   # One second
        ann.setEasingCurve(QEasingCurve.InOutQuad)

        if action == 0 and self._last_reward == 2.0:
            self._agent.setPixmap(self._agent_on)   # Go to the root with an object
        else:
            self._agent.setPixmap(self._agent_off)

        # Move and display the reward label if a reward was obtained
        if self._last_reward != 0.0:
            rann = QPropertyAnimation(self._reward_label, b"pos")
            self._reward_label.setText('<h1>%+i</h1>' % self._last_reward)
            self._reward_label.show()

            start = [LOC_ROOT, LOC_T1, LOC_T2][self._last_action]
            start = (start[0] - 20, start[1] + 50)
            end = (start[0] - 20, start[1] - 50)

            rann.setStartValue(QPoint(*start))
            rann.setEndValue(QPoint(*end))
            rann.setDuration(1000)
            rann.start()

        ann.finished.connect(self._loop.quit)
        ann.start()
        self._loop.exec_()

        self._reward_label.hide()
        self._app.processEvents()
