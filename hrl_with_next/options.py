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

import os
import sys
import argparse

import keras
import keras.backend as K
import gym
import numpy as np
import datetime
import random

sys.path.append(os.path.abspath('..'))

import gym_envs.khepera
import gym_envs.treemaze
import gym_envs.sonywrapper

epsilon = 1e-8
LSTM_TIMESTEPS = 30

class Experience(object):
    def __init__(self, state, action, value, overriden):
        """ In <state>, executing <option>, <action> has been choosen, which lead to <reward>
        """
        self.state = state
        self.action = action
        self.overriden = overriden
        self.value = value              # Expected value of state-ocurrent
        self.reward = 0.0               # The reward is set once it is known
        self.interrupt = False          # Interrupt reward chain at option boundaries

    def __repr__(self):
        return '(s: %s, oc: %s, allowed: %s, action: %s, r: %f, int: %s)' % \
            (self.state, self.ocurrent, self.oallowed, self.action, self.reward, self.interrupt)

class Learner(object):
    def __init__(self, args, f_next, f_sub, f_alias, policy, shaping):
        """ Construct a Learner from parsed arguments
        """

        # Make environment
        self._env = gym.make(args.env)
        self._policy = policy
        self._shaping = shaping
        self._render = args.render
        self._lstm = args.lstm

        if args.sony:
            self._env = gym_envs.sonywrapper.SonyWrapper(self._env)

        # Native actions
        aspace = self._env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]               # Ensure that the action space is a list for all the environments

        self._num_actions = np.prod([a.n for a in aspace])
        self._aspace = aspace

        # Initialize random actions and OOIs if needed
        if args.random_nexts:
            # Random number of options
            N = args.option
            option_indexes = range(self._num_actions, self._num_actions + N)

            f_sub = {-1: option_indexes}
            f_next = {}

            for i in range(N):
                f_sub[i] = range(self._num_actions)
                f_next[i] = random.sample(option_indexes, N // 2)

        # Make options
        self._f_next = f_next
        self._f_sub = f_sub
        self._f_alias = f_alias

        self._num_options = args.option
        self._num_optact = self._num_actions + self._num_options
        self._total_actions = 2 * (self._num_optact)                            # Choosing between primitive actions and options, with or without ending the current option

        # Build network
        self._discrete_obs = isinstance(self._env.observation_space, gym.spaces.Discrete)

        if self._discrete_obs:
            self._state_vars = self._env.observation_space.n                    # Prepare for one-hot encoding
        else:
            self._state_vars = np.product(self._env.observation_space.shape)

        self.make_network(self._state_vars, args.hidden, args.lr)

        print('Number of primitive actions:', self._num_actions)
        print('Total number of options:', self._num_optact)
        print('Number of state variables', self._state_vars)

        # Lists for policy gradient
        self._experiences = []

    def make_shape(self, x):
        """ Return a 2D (LSTM) or 1D (no LSTM) shape
        """
        if self._lstm:
            return (LSTM_TIMESTEPS, x)
        else:
            return (x,)

    def make_network(self, num_state_var, hidden, lr):
        """ Initialize a simple multi-layer perceptron for policy gradient
        """
        # Useful functions
        def make_probas(pi, oallowed):
            # Probability distribution constrained on oallowed
            x_exp = K.sigmoid(pi)
            x_exp *= oallowed
            return x_exp / K.sum(x_exp)

        def make_function(input, noutput, activation='sigmoid'):
            if self._lstm:
                dense1 = keras.layers.recurrent.LSTM(units=hidden, activation='tanh')(input)
            else:
                dense1 = keras.layers.Dense(units=hidden, activation='tanh')(input)

            dense2 = keras.layers.Dense(units=noutput, activation=activation)(dense1)

            return dense2

        # Neural network with state and option inputs, end two outputs: a distribution
        # over options and an "end" signal
        state = keras.layers.Input(shape=self.make_shape(num_state_var))
        ocurrent = keras.layers.Input(shape=self.make_shape(self._num_options))
        oallowed = keras.layers.Input(shape=(self._total_actions,))              # Mask for options that are allowed to run

        stateoption = keras.layers.concatenate([state, ocurrent])
        pi = make_function(stateoption, self._total_actions, 'linear')          # Option to execute given current state and option
        probas = keras.layers.core.Lambda(make_probas, output_shape=(self._total_actions,), arguments={'oallowed': oallowed})(pi)
        critic = make_function(stateoption, 1, 'linear')                        # Expected value of a state-action

        self._model = keras.models.Model(inputs=[state, ocurrent, oallowed], outputs=[probas])
        self._critic = keras.models.Model(inputs=[state, ocurrent], outputs=[critic])

        # Compile model with Policy Gradient loss
        print("Compiling model", end="")
        self._model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
        self._critic.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
        print(" done")

        # Policy gradient loss for the policy
        pi_true = self._model.targets[0]
        pi_pred = self._model.outputs[0]
        logpi = K.log(pi_pred + epsilon)
        grad = K.mean(pi_true * logpi)

        self._model.total_loss = -grad

    def encode_state(self, state):
        """ Encode a raw state from Gym to a Numpy vector
        """
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(self._state_vars,))
            rs[state] = 1.0
        elif isinstance(state, np.ndarray):
            rs = state.flatten()
        else:
            rs = np.array(state)

        return rs

    def predict_probas(self, state):
        """ Return a probability distribution over options (and a terminal signal)
            based on an observation from the environment and the current option
        """
        observation, ocurrent, oallowed = state

        # Compute the allowed option mask
        output = self._model.predict_on_batch([
            observation[None, :],
            ocurrent[None, :],
            oallowed[None, :]
        ])
        critic = self._critic.predict_on_batch([
            observation[None, :],
            ocurrent[None, :]
        ])

        probas = output[0].flatten()
        value = critic[0, 0]

        return (probas, value)

    def learn_from_experiences(self):
        """ Learn from the experience pool, using Policy Gradient
        """
        N = len(self._experiences)

        if N == 0:
            return

        target_option = np.zeros((N, self._total_actions))
        source_ocurrent = np.zeros((N,) + self.make_shape(self._num_options))
        source_oallowed = np.zeros((N, self._total_actions))
        source_state = np.zeros((N,) + self.make_shape(self._state_vars))

        source_critic_ocurrent = np.zeros((N,) + self.make_shape(self._num_options))
        source_critic_state = np.zeros((N,) + self.make_shape(self._state_vars))

        # Compute cumulative rewards
        cumulative_rewards = np.zeros(shape=(N,))
        cumulative_reward = 0.0

        for i in range(N-1, -1, -1):
            e = self._experiences[i]

            if e.interrupt:
                cumulative_reward = 0.0     # Break the cumulative reward chain

            cumulative_reward = e.reward + cumulative_reward
            cumulative_rewards[i] = cumulative_reward

        # Build source and target arrays for the actor
        for i in range(N):
            e = self._experiences[i]

            if e.overriden:
                value = 0.0                                 # Disable learning for overriden actions
            else:
                value = cumulative_rewards[i] - e.value     # Use value as a baseline return

            state, ocurrent, oallowed = e.state

            source_oallowed[i, :] = oallowed
            source_state[i, :] = state
            source_ocurrent[i, :] = ocurrent
            target_option[i, e.action] = value

            source_critic_ocurrent[i, :] = ocurrent
            source_critic_state[i, :] = state

        # Build source and target arrays for the critic
        target_critic_value = cumulative_rewards.reshape((N, 1))

        # Train the neural network
        self._model.fit(
            [source_state, source_ocurrent, source_oallowed],
            [target_option],
            batch_size=N,
            epochs=1,
            verbose=0
        )
        self._critic.fit(
            [source_critic_state, source_critic_ocurrent],
            [target_critic_value],
            batch_size=10,
            epochs=10,
            verbose=0
        )

        # Prepare for next batch of episodes
        self._experiences.clear()

    def mask(self, index):
        """ Return a N-vector (with N num of options) whose index-th element
            is 1, the rest are zeros.
        """
        rs = np.zeros((self._num_options,))

        if index != -1:
            rs[index] = 1.0

        return rs

    def make_oallowed(self, allowed_indices):
        """ Make a vector containing ones where allowed_indices says that a one
            may be.
        """
        oallowed = np.ones((self._total_actions,))

        if len(allowed_indices) > 0:
            # Don't mask if the user has provided no mask
            oallowed.fill(0.0)
            oallowed[allowed_indices] = 1.0                                  # Options without end
            oallowed[self._num_optact:][allowed_indices] = 1.0               # Options with end

        return oallowed

    def make_state(self, state, ocurrent, oallowed):
        """ Return a list of states (if LSTM) or just <state> (no LSTM)
        """
        if self._lstm:
            self._prev_states.append((state, ocurrent))

            l_state = np.zeros((LSTM_TIMESTEPS, self._state_vars))
            l_ocurrent = np.zeros((LSTM_TIMESTEPS, self._num_options))
            count = min(len(self._prev_states), LSTM_TIMESTEPS)

            for i in range(-count, 0):
                s, oc = self._prev_states[i]

                l_state[i] = s
                l_ocurrent[i] = oc

            return (l_state, l_ocurrent, oallowed)
        else:
            return (state, ocurrent, oallowed)

    def execute_option(self, ocurrent_index, recur=0, env_state=None):
        """ Execute an option on the environment
        """

        # Prevent stack overflows
        if recur > 100:
            return (env_state, -1000, True)

        if env_state is None:
            # Reset the environment at the beginning of the episode
            env_state = self._env.reset()
            self._prev_states = []

        oreturned_index = None
        ocurrent = self.mask(ocurrent_index)

        end_proba = 0.0
        done = False
        end = False
        cumulative_reward = 0.0

        while (not end) and (not done):
            # Indices of options that can be taken
            allowed_indices = []

            if self._f_sub is not None:
                allowed_indices = list(self._f_sub.get(ocurrent_index, []))

            if (oreturned_index is not None) and (self._f_next is not None):
                if oreturned_index in self._f_next:
                    # Next filter on the option that has just finished, ignore
                    # f_sub and use f_next instead
                    allowed_indices = [i for i in self._f_next[oreturned_index] if i in self._f_sub.get(ocurrent_index, [])]

            # Select an action or option based on the current state
            old_env_state = env_state
            ignored_reward = 0.0
            oallowed = self.make_oallowed(allowed_indices)

            if recur == 0:
                # The top-level policy cannot terminate
                oallowed[self._num_optact:] = 0.0

            state = self.make_state(self.encode_state(env_state), ocurrent, oallowed)
            overriden = False

            probas, value = self.predict_probas(state)
            policy_probas = self._policy(env_state, ocurrent_index)

            if policy_probas is not None:
                overriden = True
                probas = np.array(policy_probas) * oallowed
                probas /= np.sum(probas)

            option = np.random.choice(self._total_actions, p=probas)

            # Store experience, without the reward, that is not yet known
            e = Experience(
                state,
                option,
                value,
                overriden
            )
            self._experiences.append(e)

            # The action set is duplicated : without and with end
            if option >= self._num_optact:
                # End the current option after this action
                option -= self._num_optact
                end = True

            # Parse the option number to get an option, primitive action and end signal
            if option < self._num_actions:
                # Primitive action
                if len(self._aspace) > 1:
                    # Choose each of the factored action depending on the composite action
                    actions = [0] * len(self._aspace)
                    o = option

                    for i in range(len(actions)):
                        actions[i] = o % self._aspace[i].n
                        o //= self._aspace[i].n

                    env_state, reward, done, __ = self._env.step(actions)
                else:
                    # Simple scalar option
                    env_state, reward, done, __ = self._env.step(option)

                if self._render:
                    self._env.render()
            else:
                # Execute a sub-option, ignore its reward
                o = option - self._num_actions
                env_state, ignored_reward, done = self.execute_option(o, recur + 1, env_state)
                reward = 0.0

                # Change which option has returned
                oreturned_index = o

            # Add the reward of the option (options can do reward shaping)
            additional_reward, d = self._shaping(ocurrent_index, old_env_state, env_state)

            if d is not None:
                done = d

            # Update the experience with its reward
            cumulative_reward += reward + ignored_reward + additional_reward
            e.reward = reward + additional_reward

        # Mark episode boundaries
        if recur == 0:
            self._experiences[-1].interrupt = True

        return (env_state, cumulative_reward, done)

def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description="Reinforcement Learning for the Gym")

    parser.add_argument("--render", action="store_true", default=False, help="Enable a graphical rendering of the environment")
    parser.add_argument("--monitor", action="store_true", default=False, help="Enable Gym monitoring for this run")
    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--sony", action="store_true", default=False, help="Replace the observations of the environment by features from a Sony live view stream")
    parser.add_argument("--loops", type=int, default=1000, help="Maximum episode size")
    parser.add_argument("--avg", type=int, default=1, help="Episodes run between gradient updates")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--name", type=str, default='', help="Experiment name")

    parser.add_argument("--hidden", default=100, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate of the neural network")
    parser.add_argument("--load", type=str, help="File from which to load the neural network weights")
    parser.add_argument("--lstm", action="store_true", default=False, help="Use an LSTM layer in the network")

    parser.add_argument("--option", type=int, default=0, help="Number of options. Reward shaping is done through policy with a shaping(option_number, old_state, new_state) function.")
    parser.add_argument("--nexts", type=str, help="Python dictionary from option numbers to lists of option numbers, used to train f_next: O => R^O. Options 0..N are primitive actions")
    parser.add_argument("--subs", type=str, help="Same a --nexts, but for sub-options")
    parser.add_argument("--alias", type=str, help="Same a --nexts, but maps options to aliased options (that share the same policy)")
    parser.add_argument("--policy", type=str, help="Python file that contains the policy, see treemaze_policy.py for an example")
    parser.add_argument("--random-nexts", default=False, action="store_true", help="Randomly initialize options and OOIs instead of obeying --subs and --nexts.")

    # Next and Sub from arguments
    args = parser.parse_args()
    f_next = None
    f_sub = None
    f_alias = None

    if args.nexts is not None:
        f_next = eval(args.nexts)

    if args.subs is not None:
        f_sub = eval(args.subs)

    if args.alias is not None:
        f_alias = eval(args.alias)

    # Load predefined policy if needed
    policy = lambda state, option: None
    shaping = lambda o, old, new: (0.0, None)

    if args.policy is not None:
        data = open(args.policy, 'r').read()
        compiled = compile(data, args.policy, 'exec')
        d = {}
        exec(compiled, d)

        if 'policy' in d:
            policy = d['policy']
        if 'shaping' in d:
            shaping = d['shaping']
        if 'num_options' in d:
            args.option = d['num_options']
        if 'nexts' in d:
            f_next = d['nexts']
            f_sub = d['subs']

    # Instantiate learner
    learner = Learner(args, f_next, f_sub, f_alias, policy, shaping)

    # Load weights if needed
    if args.load is not None:
        learner._model.load_weights(args.load)

    # Learn
    f = open('out-' + args.name, 'w')

    if args.monitor:
        learner._env.monitor.start('/tmp/monitor', force=True)

    try:
        gym.envs.algorithmic.algorithmic_env.AlgorithmicEnv.current_length = 10
    except:
        pass

    try:
        avg = 0.0

        for i in range(args.episodes):
            _, reward, done = learner.execute_option(-1)

            if i == 0:
                avg = reward
            else:
                avg = 0.999 * avg + 0.001 * reward

            # Learn when enough experience is accumulated
            if (i % args.avg) == 0:
                learner.learn_from_experiences()

            print("Cumulative reward:", reward, "; average reward:", avg, file=f)
            print(args.name, "Cumulative reward:", reward, "; average reward:", avg)
            f.flush()
    except KeyboardInterrupt:
        pass

    if args.monitor:
        learner._env.monitor.close()

    f.close()

if __name__ == '__main__':
    main()
