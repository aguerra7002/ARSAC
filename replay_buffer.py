import numpy as np
import random
import json

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, prev_states, prev_actions, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (prev_states, prev_actions, state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))

    def percent_remaining(self):
        return (self.capacity - len(self.buffer)) / self.capacity

    def save_local(self, path):
        tmp_dict = {
            "capacity": self.capacity,
            "position": self.position,
            "buffer": self.buffer
        }
        with open(path, 'w') as f:
            json.dump(tmp_dict, f)

    def load_local(self, path):
        with open(path) as f:
            tmp_dict = json.load(f)
            self.capacity = tmp_dict['capacity']
            self.position = tmp_dict['position']
            self.buffer = tmp_dict['buffer']

    def __len__(self):
        return len(self.buffer)
