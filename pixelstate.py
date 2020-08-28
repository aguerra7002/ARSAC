import threading
# from concurrent.futures.thread import ThreadPoolExecutor # TODO Try something using this
import gym
import numpy as np
import mujoco_py


class PixelState:
    env = None

    def __init__(self, num_threads, env_name, resolution, state_size):
        # Number of threads to use
        self.num_threads = num_threads
        if PixelState.env is None:
            PixelState.env = gym.make(env_name)
            PixelState.env.env.viewer = mujoco_py.MjRenderContextOffscreen(PixelState.env.env.sim, 0)
        # Determine state size
        self.state_size = state_size
        # Determine qpos size
        self.qpos_size = PixelState.env.sim.data.qpos.shape[0]
        # Resolution of Pixel images we will get
        self.resolution = resolution
        pass

    def thread_get_pixel_state(self, i, state_minibatch, thread_ret):
        lb = int(state_minibatch.shape[1] / self.state_size)
        res = np.zeros((state_minibatch.shape[0], 3 * lb, self.resolution, self.resolution))
        for j, states in enumerate(state_minibatch):
            for l, state in enumerate(np.array_split(states, lb)):
                PixelState.env.sim.set_state_from_flattened(state)
                img = PixelState.env.env.sim.render(camera_name='track', width=self.resolution, height=self.resolution, depth=False)
                res[j, 3 * l:3 * (l + 1)] = (img.reshape((img.shape[2], img.shape[1], img.shape[0])) / 255) - 0.5
        thread_ret[i] = res

    def get_pixel_state(self, state_batch, batch=True):
        if batch:
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
            return np.concatenate(thread_ret, axis=0)
        else:
            # If we are here we are just converting a single instance of a state(s) to image(s). No multithreading necessary.
            lb = int(state_batch.shape[0] / self.state_size)
            res = np.zeros((3 * lb, self.resolution, self.resolution))
            for l, state in enumerate(np.array_split(state_batch, lb)):
                PixelState.env.sim.set_state_from_flattened(state)
                img = PixelState.env.env.sim.render(camera_name='track', width=self.resolution, height=self.resolution,
                                                    depth=False)
                img = img.reshape((img.shape[2], img.shape[1], img.shape[0]))
                res[3 * l:3 * (l + 1)] = (img / 255) - 0.5
            return res
