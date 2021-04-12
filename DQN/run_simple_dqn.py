import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F


from human_observation.human_obs import DFAWrapper
from gym_minigrid.wrappers import *






LTL_PATH = "./ltl_2_dfa/neverClaimFiles/never_claim_4.txt"

dfa = DFAWrapper(LTL_PATH)
n = 6

env_name = "MiniGrid-Empty-{}x{}-v0".format(n+2,n+2)
env = gym.make(env_name)
env = RGBImgObsWrapper(env)
env.reset()





model_name = 'Dueling_DDQN_Prior_Memory'
save_name = 'checkpoints/' + model_name
resume = False

class Config():

    def __init__(self):
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 10
        self.TARGET_UPDATE = 200
        self.BATCH_SIZE = 256
        self.start_from = 512
        self.GAMMA = 1
        self.dueling = True
        self.plot_every = 5
        self.lr = 3e-5
        self.optim_method = optim.Adam
        self.memory_size = 10000
        self.conv_layer_settings = [
            (3, 8, 5, 2),
            (8, 16, 5, 2),
            (16, 32, 5, 2),
            (32, 32, 5, 2)
        ]







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition', ['state', 'action', 'reward', 'next_state', 'terminal'])


class ReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0**self.prob_alpha

        total = len(self.buffer)
        if total < self.capacity:
            pos = total
            self.buffer.append(transition)
        else:
            prios = self.priorities[:total]
            probs = (1 - prios / prios.sum()) / (total - 1)
            pos = np.random.choice(total, 1, p=probs)

        self.priorities[pos] = max_prio

    def sample(self, batch_size):
        total = len(self.buffer)
        prios = self.priorities[:total]
        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min*total)**(-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights = torch.tensor(weights, device=device, dtype=torch.float)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5)**self.prob_alpha

    def __len__(self):
        return len(self.buffer)



def init_params(net):

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            init.constant_(m.bias, 0)


class ConvBlock(nn.Module):

    def __init__(self, input_size, output_size, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=kernel_size, stride=stride, padding=self.padding)
        self.bn = nn.BatchNorm2d(output_size)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

    def size_out(self, size):
        return (size - self.kernel_size + self.padding * 2) // self.stride + 1


class DQN(nn.Module):

    def __init__(self, h, w, conv_layer_settings, dueling=False):
        super(DQN, self).__init__()
        self.dueling = dueling

        conv_blocks = []
        size = np.array([h, w])
        for s in conv_layer_settings:
            block = ConvBlock(s[0], s[1], s[2], s[3])
            conv_blocks.append(block)
            size = block.size_out(size)
        self.conv_step = nn.Sequential(*conv_blocks)
        linear_input_size = size[0] * size[1] * conv_layer_settings[-1][1]

        if self.dueling:
            self.adv = nn.Linear(linear_input_size, 2)
            self.val = nn.Linear(linear_input_size, 1)
        else:
            self.head = nn.Linear(linear_input_size, 2)

    def forward(self, x):
        x = self.conv_step(x)
        x = x.view(x.size(0), -1)

        if self.dueling:
            adv = F.relu(self.adv(x))
            val = F.relu(self.val(x))
            return val + adv - val.mean()
        else:
            return self.head(x)








from threading import Event, Thread
import torchvision.transforms as T
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt




resize = T.Compose([T.ToPILImage(),
                    T.CenterCrop((250, 500)),
                    T.Resize(64),
                    T.Grayscale(),
                    T.ToTensor()])


class RenderThread(Thread):
    # Usage:
    # 0. call env.step() or env.reset() to update env state
    # 1. call begin_render() to schedule a rendering task (non-blocking)
    # 2. call get_screen() to get the lastest scheduled result (block main thread if rendering not done)

    def __init__(self, env):
        super(RenderThread, self).__init__(target=self.render)
        self._stop_event = Event()
        self._state_event = Event()
        self._render_event = Event()
        self.env = env

    def stop(self):
        self._stop_event.set()
        self._state_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def begin_render(self):
        self._state_event.set()

    def get_screen(self):
        self._render_event.wait()
        self._render_event.clear()
        return self.screen

    def render(self):
        while not self.stopped():
            self._state_event.wait()
            self._state_event.clear()

            self.screen = self.env.render(
                mode='rgb_array').transpose((2, 0, 1))
            self.screen = np.ascontiguousarray(
                self.screen, dtype=np.float32) / 255
            self.screen = torch.from_numpy(self.screen)
            self.screen = resize(self.screen).unsqueeze(0).to(device)
            self._render_event.set()


# A simple test
renderer = RenderThread(env)
renderer.start()

env.reset()
renderer.begin_render()
screen = renderer.get_screen()

plt.figure()
plt.imshow(screen.cpu().squeeze(0).permute(
    1, 2, 0).numpy().squeeze(), cmap='gray')
plt.title('Example extracted screen')
plt.show()
renderer.stop()
renderer.join()

_, _, screen_height, screen_width = screen.shape





















class History():

    def __init__(self, plot_size=300, plot_every=5):
        self.plot_size = plot_size
        self.episode_durations = deque([], self.plot_size)
        self.means = deque([], self.plot_size)
        self.episode_loss = deque([], self.plot_size)
        self.indexes = deque([], self.plot_size)
        self.step_loss = []
        self.step_eps = []
        self.peak_reward = 0
        self.peak_mean = 0
        self.moving_avg = 0
        self.step_count = 0
        self.total_episode = 0
        self.plot_every = plot_every

    def update(self, t, episode_loss):
        self.episode_durations.append(t + 1)
        self.episode_loss.append(episode_loss / (t + 1))
        self.indexes.append(self.total_episode)
        if t + 1 > self.peak_reward:
            self.peak_reward = t + 1
        if len(self.episode_durations) >= 100:
            self.means.append(sum(list(self.episode_durations)[-100:]) / 100)
        else:
            self.moving_avg = self.moving_avg + \
                (t - self.moving_avg) / (self.total_episode + 1)
            self.means.append(self.moving_avg)
        if self.means[-1] > self.peak_mean:
            self.peak_mean = self.means[-1]

        if self.total_episode % self.plot_every == 0:
            self.plot()

    def plot(self):
        display.clear_output(wait=True)

        f, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(self.indexes, self.episode_durations)
        ax1.plot(self.indexes, self.means)
        ax1.axhline(self.peak_reward, color='g')
        ax1.axhline(self.peak_mean, color='g')

        ax2 = ax1.twinx()
        ax2.plot(self.indexes, self.episode_loss, 'r')

        ax4 = ax3.twinx()
        total_step = len(self.step_loss)
        sample_rate = total_step // self.plot_size if total_step > (
            self.plot_size * 10) else 1
        ax3.set_title('total: {0}'.format(total_step))
        ax3.plot(self.step_eps[::sample_rate], 'g')
        ax4.plot(self.step_loss[::sample_rate], 'b')

        plt.pause(0.00001)





















