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

from gym.envs.registration import registry, register, make, spec

register(
    id='Khepera-v0',
    entry_point='gym_envs.khepera:DiscreteKheperaEnv',
    reward_threshold=0.0,
    kwargs={'f': '/dev/rfcomm0'}
)

register(
    id='KheperaContinuous-v0',
    entry_point='gym_envs.khepera:ContinuousKheperaEnv',
    reward_threshold=0.0,
    kwargs={'f': '/dev/rfcomm1'}
)

register(
    id='TreeMaze-v0',
    entry_point='gym_envs.treemaze:TreeMazeEnv',
    kwargs={'size': 5, 'height': 3}
)

register(
    id='DuplicatedInputCond-v0',
    entry_point='gym_envs.duplicatedinputcond:DuplicatedInputCondEnv',
    kwargs={'duplication': 2, 'base': 5}
)

register(
    id='Terminals-v0',
    entry_point='gym_envs.terminals:TerminalsEnv',
    kwargs={}
)

register(
    id='Mario-v0',
    entry_point='gym_envs.mario:MarioEnv',
    kwargs={}
)

register(
    id='Robot-v0',
    entry_point='gym_envs.robot:RobotEnv',
    kwargs={}
)

register(
    id='DeepExploration20-v0',
    entry_point='gym_envs.deepexploration:DeepExplorationEnv',
    kwargs={'size': 20}
)
