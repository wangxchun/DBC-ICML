import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import os, sys
import argparse

def seperate_traj(demo):
    trajs = []
    x = []
    y = []
    for i, d in enumerate(demo['done']):
        if d == 1:
            x.append(demo['obs'][i][0].numpy())
            y.append(demo['obs'][i][1].numpy())
            trajs.append((x, y))
            x = []
            y = []
        else:
            x.append(demo['obs'][i][0].numpy())
            y.append(demo['obs'][i][1].numpy())
    return trajs


def denorm_vec(x, mean, std):
    obs_x = x*(std+1e-8) + mean
    return obs_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path')
    # args = parser.parse_args()
    expert = torch.load('expert_datasets/toy_v4.pt')
    expert_obs = expert['obs']
    obs_mean = expert_obs.mean(0)
    obs_std = expert_obs.std(0)

    bc = torch.load('expert_datasets/toy_v4_bc.pt')
    bc['obs'] = denorm_vec(bc['obs'], obs_mean, obs_std)

    diff = torch.load('expert_datasets/toy_v4_diff.pt')
    diff['obs'] = denorm_vec(diff['obs'], obs_mean, obs_std)

    # circle = plt.Circle((1, 5), 0.3, color='lightgreen')
    fig, ax = plt.subplots()
    # ax.add_patch(circle)
    # rect = patches.Rectangle((1.3, 1.3), 3.75, 3.75, facecolor='rosybrown')
    # ax.add_patch(rect)
    # Plot expert (black)#
    trajs = seperate_traj(expert)
    for traj in trajs[:3]:
        x, y = traj
        ax.plot(x[0], y[0], 'ko')
        ax.plot(x, y, 'k')
    # Plot BC (blue)#
    trajs = seperate_traj(bc)
    for idx in [2, 3, 5]:
        x, y = trajs[idx]
        ax.plot(x[0], y[0], 'o', color='cornflowerblue')
        ax.plot(x, y, color='cornflowerblue')
    
    # Plot Diff (orange)#
    trajs = seperate_traj(diff)
    for idx in [1, 3, 4]:
        x, y = trajs[idx]
        ax.plot(x[0], y[0], 'o', color='darkorange')
        ax.plot(x, y, color='darkorange')
    plt.savefig('test.png')
