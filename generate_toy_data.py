#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import torch.nn as nn
import cv2
import os, sys
sys.path.insert(0, '../')
import goal_prox.envs.fetch
import goal_prox.envs.hand
import argparse
import numpy as np
import scipy.stats
from geomloss import SamplesLoss
import gym
import d4rl # Import required to register environments
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler
from tqdm import tqdm
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from rlf.exp_mgr.viz_utils import save_mp4


left_1 = np.repeat(np.array([[-1, 0]]), 20, axis=0)
left_2 = np.repeat(np.array([[0, 0]]), 75, axis=0)
left_3 = np.repeat(np.array([[1, 0]]), 17, axis=0)
up_1 = np.repeat(np.array([[0, 1]]), 20, axis=0)
up_2 = np.repeat(np.array([[0, 0]]), 75, axis=0)
up_3 = np.repeat(np.array([[0, -1]]), 16, axis=0)
# action_list = np.concatenate((left_1, left_2, left_3, left_4), axis=0)
# action_list = np.concatenate((left_2, left_3, left_4), axis=0)
action_list = np.concatenate((left_1, left_2, left_3,
                              up_1, up_2, up_3), axis=0)

x1 = np.repeat(np.array([[0.5, 0]]), 40, axis=0)
y1 = np.repeat(np.array([[0, 0.5]]), 40, axis=0)
x2 = np.repeat(np.array([[-0.7, 0]]), 40, axis=0)
y2 = np.repeat(np.array([[0, -0.7]]), 41, axis=0)
# x2 = np.repeat(np.array([[-1, 0]]), 20, axis=0)
# y2 = np.repeat(np.array([[0, -1]]), 20, axis=0)
action_list_2 = np.concatenate((x1, y1, x2, y2), axis=0)


def get_action(idx, obs):
    action = action_list[idx]
    if idx < action_list.shape[0]/2:
        x = action[0]
        if obs[1] > 1 and obs[3] > -0.2:
            y = -0.05
        elif obs[1] < 1 and obs[3] < 0.2:
            y = 0.05
        else:
            y = 0
    elif idx > action_list.shape[0]/2 and idx < (action_list.shape[0]-1):
        y = action[1]
        if obs[0] > 1 and obs[2] > -0.2:
            x = -0.05
        elif obs[0] < 1 and obs[2] < 0.2:
            x = 0.05
        else:
            x = 0
    else:
        x = y = 1
    return np.array([x, y])

def get_action_det(idx):
    if idx < action_list_2.shape[0] - 1:
        return action_list_2[idx]
    else:
        return np.array([1, 1])

def run(env_name, num_episode = 10):
    env = gym.make(env_name)
    obs_space = env.observation_space
    # video = VideoRecorder(env, 'toy.mp4')
    print(f'Obs space high: {obs_space.high}')
    print(f'Obs space low: {obs_space.low}')
    print(f'Action space high: {env.action_space.high}')
    print(f'Action space low: {env.action_space.low}')
    obs = env.reset()
    frame_idx = 0
    frames = [env.render('rgb_array')]
    traj = {}
    episode = 0
    obs_list = []
    next_obs = []
    done_list = []
    action_list = []
    while episode < num_episode:
        # action = get_action(frame_idx, obs)
        action = get_action_det(frame_idx)
        # print(obs)
        # print(action)
        if action.sum() == 2:
            print("stop")
            input()
        obs_list.append(torch.from_numpy(obs).unsqueeze(0))
        action_list.append(torch.from_numpy(action).unsqueeze(0))
        obs, reward, done, info = env.step(action)
        next_obs.append(torch.from_numpy(obs).unsqueeze(0))
        frame_idx += 1
        if done:
            print(obs)
            # input()
            done_list.append(1)
            obs = env.reset()
            frame_idx = 0
            episode += 1
        else:
            done_list.append(0)
        frames.append(env.render('rgb_array'))
    traj['obs'] = torch.cat(obs_list, dim=0)
    traj['next_obs'] = torch.cat(next_obs, dim=0)
    traj['done'] = torch.tensor(done_list)
    traj['actions'] = torch.cat(action_list, dim=0)
    save_mp4(frames, '.', 'toy', fps=50)
    for k, v in traj.items():
        print(k)
        print(v.shape)
        if k == 'done':
            print(v.sum())
    # torch.save(traj, f'expert_datasets/toy_v4.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--env-name')
    args = parser.parse_args()
    path = args.path
    env_name = args.env_name
    if env_name == None:
        env_name = 'maze2d-toy-v0'
    run(env_name)
