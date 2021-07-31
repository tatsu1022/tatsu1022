import numpy as np
import torch
import tensorflow as tf


class ReplayMemory:

    def __init__(self, state_shape, action_shape, userDefinedSettings):
        self.memory_size = int(userDefinedSettings.memory_size)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=self.state_type)
        next_state = np.array(next_state, dtype=self.state_type)

        self.states[self._p] = state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.memory_size)
        self._p = (self._p + 1) % self.memory_size

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.uint8)
            next_states = self.next_states[indices].astype(np.uint8)
            # states = torch.ByteTensor(states).to(self.device).float() / 255.
            states = tf.constant(states, dtype=tf.float64) / 255.
            # next_states = torch.ByteTensor(next_states).to(self.device).float() / 255.
            next_states = tf.constant(next_states, dtype=tf.float64) / 255.
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]
            # states = torch.FloatTensor(states).to(self.device)
            states = tf.constant(states, dtype=tf.float64)
            # next_states = torch.FloatTensor(next_states).to(self.device)
            next_states = tf.constant(next_states, dtype=tf.float64)

        # actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        actions = tf.constant(self.actions[indices], dtype=tf.float64)
        # rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        rewards = tf.constant(self.rewards[indices], dtype=tf.float64)
        # dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        dones = tf.constant(self.dones[indices], dtype=tf.float64)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.empty((self.memory_size, *self.state_shape), dtype=self.state_type)
        self.actions = np.empty((self.memory_size, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.memory_size, 1), dtype=np.float32)
        self.next_states = np.empty((self.memory_size, *self.state_shape), dtype=self.state_type)
        self.dones = np.empty((self.memory_size, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid], self.actions[valid], self.rewards[valid],
            self.next_states[valid], self.dones[valid])

    def load(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.memory_size:
            self._insert(
                slice(self._p, self._p + num_data), batch,
                slice(0, num_data))
        else:
            mid_index = self.memory_size - self._p
            end_index = num_data - mid_index
            self._insert(
                slice(self._p, self.memory_size), batch,
                slice(0, mid_index))
            self._insert(
                slice(0, end_index), batch,
                slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.memory_size)
        self._p = (self._p + num_data) % self.memory_size

    def _insert(self, mem_indices, batch, batch_indices):
        states, actions, rewards, next_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
