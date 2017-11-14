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
import serial
import numpy as np

from gym import spaces
from gym.utils import seeding

def clamp(x, low, high):
    return min(high, max(x, low))

MAX_SPEED = 20000

class AbstractKheperaEnv(gym.Env):
    """ Perform actions on a Khepera robot (accessible by reading and writing to
        a terminal)

        How to connect to the bot:

            - Turn them on
            - Ensure that bluetoothd is started (sudo systemctl start bluetooth.service)
            - do "hcitool scan" to get the bdaddresses of all the robots
            - "sudo hcitool cc <bdaddress>" for each address
            - "sudo sudo rfcomm bind <i> <bdaddress>" for each address, with i = 0..N
            - "sudo chmod 666 /dev/rfcomm*" to be able to communicate with the robots as user

        To have robots automatically avoid obstacles and explore :

            echo "A,2" > /dev/rfcommXXX
    """

    def __init__(self, f='/dev/rfcomm0'):
        self.observation_space = spaces.Box(
            np.array([0.0] * 13),
            np.array([1.0] * 13)
        )         # Proximities of all the IR sensors (11) + speed of the two motors

        # Connect to the Khepera robot
        print('Using Khepera connected to', f)
        self._file = serial.Serial(f)

        # Test the connection and configure the robot
        version = self._command('B')

        print('Successfully connected to a Khepera robot, firmware', version.split(',')[1:])

        self._command('C,0,d21')    # Front, left and right US sensors

        # Initialize Braitenberg
        self._braitenberg_weights_l = [20, 20, 40, 60, -50, -40, -20, -20, 20]
        self._braitenberg_weights_r = [-20, -20, -40, -60, 50, 40, 20, 20, 20]
        self._speed0 = 0
        self._speed1 = 0

    def _command(self, cmd):
        """ Send a command to the robot and return the answer
        """
        self._file.write(bytes(cmd + '\n', 'ascii'))

        return str(self._file.readline().strip(), 'ascii')

    def _read_us(self, number):
        """ Read a distance measure from an ultrasonic sensor (1 to 5)
        """
        rs = self._command('G,%s' % number)

        return float(rs.split(',')[2]) / 200.0      # Max 200 cm

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self._timestep += 1

        # Observe all the IR sensors
        rs = self._command('N')
        state = [float(i) / 4096.0 for i in rs.split(',')[1:-1]]        # Ignore "n" and "timestamp"

        # Set the speed of the motors
        speed0, speed1 = self._speed_from_action(action)

        # Override the agent policy by a safety one if needed
        distance = max(state[0:9])

        if distance > 0.2:
            # Close to an obstacle, activate Braitenberg
            speed0 = sum([500 * sensor * weight for sensor, weight in zip(state, self._braitenberg_weights_l)])
            speed1 = sum([500 * sensor * weight for sensor, weight in zip(state, self._braitenberg_weights_r)])

            # Negative reward if a danger appears
            if self._danger == 0:
                reward = -5.0
            else:
                reward = 0.0

            self._danger = 1
        else:
            # No danger, give reward for speed (goal is to go as fast as possible without hitting anything)
            reward = abs(speed0 + speed1) / (2 * MAX_SPEED)                        # Don't give rewards that are too negative when the robot spins (usually to avoid an obstacle)

            self._danger = 0

        # Observe real speed of the motors, including backup policy
        state.append(speed0 / MAX_SPEED)
        state.append(speed1 / MAX_SPEED)

        self._speed0 = speed0
        self._speed1 = speed1

        self._command('D,l%i,l%i' % (speed0, speed1))
        #self._command('K,0,%i' % self._danger)

        return state, reward, False, {}

class DiscreteKheperaEnv(AbstractKheperaEnv):
    def __init__(self, f='/dev/rfcomm0'):
        super().__init__(f)
        
        self.action_space = spaces.Discrete(2 * 2)                          # 1, 0 for the two motors
    
    def _speed_from_action(self, action):
        speed0, speed1 = [
            (MAX_SPEED, MAX_SPEED),
            (MAX_SPEED//2, 0),
            (0, MAX_SPEED//2),
            (0, 0)
        ][action]

        return speed0, speed1

    def _reset(self):
        self._timestep = 0
        self._danger = 0

        return self._step(0)[0]     # Stop the motors and get initial observation

class ContinuousKheperaEnv(AbstractKheperaEnv):
    def __init__(self, f='/dev/rfcomm0'):
        super().__init__(f)
        
        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    
    def _speed_from_action(self, action):
        # 2 continous actions, one per motor
        return action[0] * MAX_SPEED, action[1] * MAX_SPEED
    
    def _reset(self):
        self._timestep = 0
        self._danger = 0

        return self._step(np.array([0.0, 0.0]))[0]     # Stop the motors and get initial observation
