import threading
import gym
import numpy as np
import mujoco_py
from env_wrapper import EnvWrapper


class PixelState:
    env = None

    def __init__(self, num_threads, env_name, task_name, resolution, state_size):
        # Number of threads to use
        self.num_threads = num_threads
        if PixelState.env is None:
            PixelState.env = EnvWrapper(env_name, task_name, True, resolution)
        # Determine state size
        self.state_size = state_size
        # Determine qpos size
        self.qpos_size = PixelState.env.get_state_space_size(position_only=True)
        # Resolution of Pixel images we will get
        self.resolution = resolution

    def thread_get_pixel_state(self, i, state_minibatch, thread_ret):
        lb = int(state_minibatch.shape[1] / self.state_size)
        res = np.zeros((state_minibatch.shape[0], 3 * lb, self.resolution, self.resolution))
        for j, states in enumerate(state_minibatch):
            for l, state in enumerate(np.array_split(states, lb)):
                img = PixelState.env.flattened_to_pixel(state)
                res[j, 3 * l:3 * (l + 1)] = img
        thread_ret[i] = res

    def get_pixel_state(self, state_batch, batch=True, parallel=False):
        if batch:
            if parallel:
                minibatches = np.array_split(state_batch, self.num_threads)
                thread_ret = [None] * self.num_threads
                threads = [threading.Thread(target=self.thread_get_pixel_state,
                                            args=(i, minibatches[i], thread_ret)) for i in range(self.num_threads)]
                # Start all the threads
                for thread in threads:
                    thread.start()
                # Now join to wait until they are done
                for thread in threads:
                    thread.join()
                # while thread_ret[0] is None:
                #     print("None")
                return np.concatenate(thread_ret, axis=0)
            else:
                thread_ret = [None]
                self.thread_get_pixel_state(0, state_batch, thread_ret)
                return thread_ret[0]
        else:
            # If we are here we are just converting a single instance of a state(s) to image(s). No multithreading necessary.
            lb = int(state_batch.shape[0] / self.state_size)
            res = np.zeros((3 * lb, self.resolution, self.resolution))
            for l, state in enumerate(np.array_split(state_batch, lb)):
                res[3 * l:3 * (l + 1)] = PixelState.env.flattened_to_pixel(state)
            return res
