import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, prev_state, prev_action, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (prev_state, prev_action, state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        prev_states, prev_action, state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return prev_states, prev_action, state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)