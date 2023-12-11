# Diffusion Model-Augmented Behavioral Cloning

[[Project website]](https://github.com/NTURobotLearningLab/dbc) [[Paper]](https://arxiv.org/abs/2302.13335) <br/>
[Hsiang-Chun Wang\*](https://openreview.net/profile?id=~Hsiang-Chun_Wang1),
[Shang-Fu Chen\*](https://openreview.net/profile?id=~Shang-Fu_Chen2),
[Ming-Hao Hsu](https://qaz159qaz159.github.io/),
[Chun-Mao Lai](https://mecoli1219.github.io/),
[Shao-Hua Sun](https://shaohua0116.github.io) at [NTU RLL lab](https://github.com/NTURobotLearningLab/)

This is the official PyTorch implementation of the paper ["Diffusion Model-Augmented Behavioral Cloning"](https://nturobotlearninglab.github.io/dbc/) (NeurIPS 2023).

<p align="center">
    <img src="docs/img/framework.jpeg" width="800">
</p>

## Installation

1. This code base requires `Python 3.7.2` or higher. All package requirements are in
   `requirements.txt`. To install from scratch using Anaconda, use the following
   commands.

```
conda create -n [your_env_name] python=3.7.2
conda activate [your_env_name]
pip install -r requirements.txt

cd d4rl
pip install -e .
cd ../rl-toolkit
pip install -e .

mkdir -p data/trained_models
```

2. Setup [Weights and Biases](https://wandb.ai/site) by first logging in with `wandb login <YOUR_API_KEY>` and then editing `config.yaml` with your W&B username and project name.

3. Download expert demonstration datasets to `./expert_datasets`. We include the expert demonstration datasets on [Google Drive](https://drive.google.com/drive/folders/1SIsRg0QKOHFKXOn14EndsNbRPkA-vG96?usp=sharing) and provide a script for downloading them.

```
python download_demos.py
```

## How to reproduce experiments
- For diffusion model pretraining, run `rl-toolkit/dm/ddpm.py` or `rl-toolkit/dm/ddpm_norm.py` depending on whether batch normalization and dropout layers are applied.

- For policy learning, run `dbc/main.py`, e.g., you can run the following command to run DBC on Maze environment:
`python dbc/main.py --seed 1 --prefix dbc --alg dbc --traj-load-path ./expert_datasets/maze2d_100.pt --ddpm-path [path-to-ddpm] --bc-num-epochs 2000 --coeff 5 --coeff-bc 1 --env-name maze2d-medium-v2 --eval-num-processes 1 --num-eval 100 --cuda True --num-render 3 --vid-fps 60 --lr 0.0001 --log-interval 200 --save-interval 20000 --eval-interval 2000 --hidden-dim 256 --depth 2 --clip-actions True --normalize-env False --bc-state-norm False --il-in-action-norm False --il-out-action-norm False`

- Configuration files for policy learning of all tasks can be found at `configs`. If you have a wandb account, you can run wandb sweeps with the according yaml file configuration:
`wandb sweep configs/maze/maze_dbc.yaml`

We specify how to train diffusion models and the location of configuration files as following:

### Maze2D

- Ours:
    - DM pretraining: `python rl-toolkit/dm/ddpm.py --traj-load-path expert_datasets/maze2d_100.pt --hidden-dim 128 --norm False`
    - Policy learning configuration: `configs/maze/maze_dbc.yaml`
- BC: `configs/maze/maze_bc.yaml`
- Implicit BC: `configs/maze/maze_ibc.yaml`
- Diffusion Policy: `configs/maze/maze_dp.yaml`

### Fetch Pick

- Ours:
    - DM pretraining: `python rl-toolkit/dm/ddpm_norm.py --traj-load-path expert_datasets/pick_10000_clip.pt --hidden-dim 1024`
    - Policy learning configuration: `configs/fetchPick/pick_dbc.yaml`
- BC: `configs/fetchPick/pick_bc.yaml`
- Implicit BC: `configs/fetchPick/pick_ibc.yaml`
- Diffusion Policy: `configs/fetchPick/pick_dp.yaml`

### Fetch Push

- Ours:
    - DM pretraining: `python rl-toolkit/dm/ddpm_norm.py --traj-load-path expert_datasets/push_10000_clip.pt --hidden-dim 1024`
    - Policy learning configuration: `configs/fetchPush/push_dbc.yaml`
- BC: `configs/fetchPush/push_bc.yaml`
- Implicit BC: `configs/fetchPush/push_ibc.yaml`
- Diffusion Policy: `configs/fetchPush/push_dp.yaml`

### Hand Rotate

- Ours:
    - DM pretraining: `python rl-toolkit/dm/ddpm.py --traj-load-path expert_datasets/hand_10000_v2.pt --hidden-dim 2048`
    - Policy learning configuration: `configs/hand/hand_dbc.yaml`
- BC: `configs/hand/hand_bc.yaml`
- Implicit BC: `configs/hand/hand_ibc.yaml`
- Diffusion Policy: `configs/hand/hand_dp.yaml`

### Walker

- Ours:
    - DM pretraining: `python rl-toolkit/dm/ddpm.py --traj-load-path expert_datasets/walker_5traj_processed.pt --hidden-dim 1024`
    - Policy learning configuration: `configs/walker/walker_dbc.yaml`
- BC: `configs/walker/walker_bc.yaml`
- Implicit BC: `configs/walker/walker_ibc.yaml`
- Diffusion Policy: `configs/walker/walker_dp.yaml`

## Code Structure

- `dbc`: method and custom environment code.
  - `rl-toolkit/rlf/algos/il/dbc.py`: Algorithm of our method
  - `rl-toolkit/rlf/algos/il/bc.py`: Algorithm of BC
  - `rl-toolkit/rlf/algos/il/ibc.py`: Algorithm of our Implicit BC
  - `rl-toolkit/rlf/algos/il/dp.py`: Algorithm of our Diffusion Policy
  - `d4rl/d4rl/pointmaze/maze_model.py`: Maze2D task
  - `dbc/envs/fetch/custom_push.py`: Fetch Push task.
  - `dbc/envs/fetch/custom_fetch.py`: Fetch Pick task.
  - `dbc/envs/hand/manipulate.py`: Hand Rotate task.
  - `dbc/envs/ant.py`: Ant locomotion task.
  - `download_demos.py`: script to generate demonstrations of our environments.
- `rl-toolkit`: base RL code and code for imitation learning baselines from [rl-toolkit](https://github.com/ASzot/rl-toolkit).
  - `rl-toolkit/algos/on_policy/ppo.py`: the PPO policy updater code we use for RL.
- `d4rl`: Codebase from [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl) for Maze2D.

## Acknowledgement

- The Fetch and Hand Rotate environments are with some tweaking from [OpenAI](https://github.com/openai/gym/tree/6df1b994bae791667a556e193d2a215b8a1e397a/gym/envs/robotics)
- The Ant environment is with some tweaking from [DnC](https://github.com/dibyaghosh/dnc)
- The Maze2D environment is based on [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl).
- The Walker2d environment is in [OpenAI Gym](https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v3.py).

## Reference

This repo is based on the official PyTorch [implementation](https://github.com/clvrai/goal_prox_il) of the paper ["Generalizable Imitation Learning from Observation via Inferring Goal Proximity"](https://clvrai.github.io/goal_prox_il/)

## Citation

```
@article{wang2023diffusion,
  title={Diffusion Model-Augmented Behavioral Cloning},
  author={Wang, Hsiang-Chun and Chen, Shang-Fu and Hsu, Ming-Hao and Lai, Chun-Mao and Sun, Shao-Hua},
  journal={arXiv preprint arXiv:2302.13335},
  year={2023}
}
```
# ICML
