import sys, os
sys.path.insert(0, "./")

from functools import partial

import d4rl
import torch
import torch.nn as nn
import numpy as np
from rlf import run_policy, evaluate_policy
from rlf.algos import (PPO, BaseAlgo, BehavioralCloning, DBC, 
                       DiffPolicy, Ae_bc, Eng_bc, GANBC, IBC)
from rlf.algos.il.base_il import BaseILAlgo
from rlf.algos.il.gaifo import GAIFO
from rlf.algos.il.sqil import SQIL
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.off_policy.sac import SAC
from rlf.args import str2bool
from rlf.policies import BasicPolicy, DistActorCritic, RandomPolicy
from rlf.policies.action_replay_policy import ActionReplayPolicy
from rlf.policies.actor_critic.dist_actor_q import (DistActorQ, get_sac_actor,
                                                    get_sac_critic)
from rlf.policies.actor_critic.reg_actor_critic import RegActorCritic
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.loggers.wb_logger import (WbLogger, get_wb_ray_config,
                                      get_wb_ray_kwargs)
from rlf.rl.model import CNNBase, MLPBase, MLPBasic, TwoLayerMlpWithAction
from rlf.run_settings import RunSettings
import dm.policy_model as policy_model
from ibc import dataset, models, optimizers, trainer, utils
from ibc.trainer import ImplicitTrainState
from ibc.experiment import Experiment

import dbc.envs.ball_in_cup
import dbc.envs.d4rl
import dbc.envs.fetch
import dbc.envs.goal_check
import dbc.envs.gridworld
import dbc.envs.hand
import dbc.gym_minigrid
from dbc.envs.goal_traj_saver import GoalTrajSaver
from dbc.method.airl import ProxAirl
from dbc.method.discounted_pf import DiscountedProxFunc, DiscountedProxIL
from dbc.method.goal_gail_discriminator import GoalGAIL
from dbc.method.ranked_pf import RankedProxIL
from dbc.method.uncert_discrim import UncertGAIL
from dbc.method.utils import trim_episodes_trans
from dbc.models import GwImgEncoder
from dbc.policies.grid_world_expert import GridWorldExpert
from typing import Dict, Optional, Tuple
from torch.utils.data import Dataset
import time


def get_ppo_policy(env_name, args):
    if env_name.startswith("MiniGrid") and args.gw_img:
        return DistActorCritic(get_base_net_fn=lambda i_shape: GwImgEncoder(i_shape))

    return DistActorCritic()


def get_deep_ppo_policy(env_name, args):
    return DistActorCritic(
        get_actor_fn=lambda _, i_shape: MLPBasic(
            i_shape[0], hidden_size=256, num_layers=2
        ),
        get_critic_fn=lambda _, i_shape, asp: MLPBasic(
            i_shape[0], hidden_size=256, num_layers=2
        ),
    )


def get_deep_sac_policy(env_name, args):
    return DistActorQ(
        get_critic_fn=get_sac_critic,
        get_actor_fn=get_sac_actor,
    )


def get_deep_ddpg_policy(env_name, args):
    def get_actor_head(hidden_dim, action_dim):
        return nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

    return RegActorCritic(
        get_actor_fn=lambda _, i_shape: MLPBase(i_shape[0], False, (256, 256)),
        get_actor_head_fn=get_actor_head,
        get_critic_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
            i_shape[0], (256, 256), a_space.shape[0]
        ),
    )


def get_basic_policy(env_name, args, is_stoch):
    if env_name.startswith("MiniGrid") and args.gw_img:
        return BasicPolicy(
            is_stoch=is_stoch, get_base_net_fn=lambda i_shape: GwImgEncoder(i_shape)
        )
    else:
        return BasicPolicy(
            is_stoch=is_stoch,
            get_base_net_fn=lambda i_shape: MLPBasic(
                i_shape[0],
                hidden_size=args.hidden_dim,
                num_layers=args.depth
            ),
        )

    return BasicPolicy()

def get_diffusion_policy(env_name, args, is_stoch):    
    if env_name[:9] == 'FetchPush':
        state_dim = 16
        action_dim = 3
        return policy_model.MLPDiffusion(
            n_steps = 1000,
            action_dim=action_dim, 
            state_dim=state_dim,
            num_units=args.hidden_dim,
            depth=args.depth,
            is_stoch=is_stoch,
            scheduler_type=args.dp_scheduler_type,
            )
    if env_name[:9] == 'FetchPick':
        state_dim = 16
        action_dim = 4
        return policy_model.MLPDiffusion(
            n_steps = 1000,
            action_dim=action_dim, 
            state_dim=state_dim,
            num_units=args.hidden_dim,
            depth=args.depth,
            is_stoch=is_stoch,
            scheduler_type=args.dp_scheduler_type,
            )
    if env_name[:10] == 'CustomHand':
        state_dim = 68
        action_dim = 20
        return policy_model.MLPDiffusion(
            n_steps = 1000,
            action_dim=action_dim, 
            state_dim=state_dim,
            num_units=args.hidden_dim,
            depth=args.depth,
            is_stoch=is_stoch,
            scheduler_type=args.dp_scheduler_type,
            )
    if env_name[:4] == 'maze':
        state_dim = 6
        action_dim = 2
        return policy_model.MLPDiffusion(
            n_steps = 1000,
            action_dim=action_dim, 
            state_dim=state_dim,
            num_units=args.hidden_dim,
            depth=args.depth,
            is_stoch=is_stoch,
            scheduler_type=args.dp_scheduler_type,
            )
    if env_name[:6] == 'Walker':
        state_dim = 17
        action_dim = 6
        return policy_model.MLPDiffusion(
            n_steps = 1000,
            action_dim=action_dim, 
            state_dim=state_dim,
            num_units=1024,
            depth=6,
            is_stoch=is_stoch,
            scheduler_type=args.dp_scheduler_type,
            )
    if env_name[:11] == 'HalfCheetah':
        state_dim = 17
        action_dim = 6
        return policy_model.MLPDiffusion(
            n_steps = 1000,
            action_dim=action_dim, 
            state_dim=state_dim,
            num_units=args.hidden_dim,
            depth=args.depth,
            is_stoch=is_stoch,
            scheduler_type=args.dp_scheduler_type,
            )
    if env_name[:7] == 'AntGoal':
        state_dim = 42
        action_dim = 8
        return policy_model.MLPDiffusion(
            n_steps = 1000,
            action_dim=action_dim, 
            state_dim=state_dim,
            num_units=args.hidden_dim,
            depth=args.depth,
            is_stoch=is_stoch,
            scheduler_type=args.dp_scheduler_type,
            )


def get_ibc_policy(env_name, args, is_stoch):
    train_config = args
    hidden_dim = args.hidden_dim
    depth = args.depth
    ##### config #####
    if env_name[:9] == 'FetchPush':
        state_dim = 16
        action_dim = 3
    if env_name[:9] == 'FetchPick':
        state_dim = 16
        action_dim = 4
    if env_name[:10] == 'CustomHand':
        state_dim = 68
        action_dim = 20
    if env_name[:6] == 'Walker':
        state_dim = 17
        action_dim = 6
    if env_name[:11] == 'HalfCheetah':
        state_dim = 17
        action_dim = 6
    if env_name[:4] == 'maze':
        state_dim = 6
        action_dim = 2
    if env_name[:7] == 'AntGoal':
        state_dim = 42
        action_dim = 8
    input_dim = state_dim + action_dim
    
    mlp_config = models.MLPConfig(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = 1,
        hidden_depth = depth,
        dropout_prob = 0,
    )
    optim_config = optimizers.OptimizerConfig(
        learning_rate=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    target_bounds = np.array([-np.ones(action_dim), np.ones(action_dim)])
    stochastic_optim_config = optimizers.DerivativeFreeConfig(
        bounds=target_bounds,
        train_samples=train_config.stochastic_optimizer_train_samples,
    )
    return ImplicitTrainState(
        model_config=mlp_config,
        optim_config=optim_config,
        stochastic_optim_config=stochastic_optim_config,
        is_stoch=is_stoch,
    )

def get_deep_basic_policy(env_name, args):
    return BasicPolicy(
        get_base_net_fn=lambda i_shape: MLPBase(i_shape[0], False, (512, 512, 256, 128))
    )


def get_setup_dict():
    return {
        # main experiments & baselines
        "bc": (BehavioralCloning(), partial(get_basic_policy, is_stoch=False)),
        "ibc": (IBC(), partial(get_ibc_policy, is_stoch=False)),
        "dp": (DiffPolicy(), partial(get_diffusion_policy, is_stoch=False)),
        "dbc": (DBC(), partial(get_basic_policy, is_stoch=False)),
        # generative model experiments
        "eng-bc": (Eng_bc(), partial(get_basic_policy, is_stoch=False)),
        "gan-bc": (GANBC(), partial(get_basic_policy, is_stoch=False)),
        "ae-bc": (Ae_bc(), partial(get_basic_policy, is_stoch=False)),
        # RL policy for collecting expert data
        "ppo": (PPO(), get_ppo_policy),
        "sac": (SAC(), get_deep_sac_policy),
    }


class GoalProxSettings(RunSettings):
    def get_policy(self):
        return get_setup_dict()[self.base_args.alg][1](
            self.base_args.env_name, self.base_args
        )

    def create_traj_saver(self, save_path):
        return GoalTrajSaver(save_path, False)

    def get_algo(self):
        algo = get_setup_dict()[self.base_args.alg][0]
        if isinstance(algo, NestedAlgo) and isinstance(algo.modules[0], BaseILAlgo):
            algo.modules[0].set_transform_dem_dataset_fn(trim_episodes_trans)
        if isinstance(algo, SQIL):
            algo.il_algo.set_transform_dem_dataset_fn(trim_episodes_trans)
        return algo

    def get_logger(self):
        if self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger()

    def get_add_args(self, parser):
        parser.add_argument("--alg")
        parser.add_argument("--env-name")
        # Should always be true!
        parser.add_argument("--gw-img", type=str2bool, default=True)
        parser.add_argument("--no-wb", action="store_true", default=False)
        # ibc args
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--hidden-dim", type=int, default=1024)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--stochastic-optimizer-train_samples", type=int, default=64)
        parser.add_argument("--dp-scheduler-type", type=str, default='linear')

    def import_add(self):
        import dbc.envs.fetch
        import dbc.envs.goal_check

    def get_add_ray_config(self, config):
        if self.base_args.no_wb:
            return config
        return get_wb_ray_config(config)

    def get_add_ray_kwargs(self):
        if self.base_args.no_wb:
            return {}
        return get_wb_ray_kwargs()


if __name__ == "__main__":
    start = time.time()
    run_policy(GoalProxSettings())
    end = time.time()
    print("The time used to execute this is:", end - start)
    # evaluate_policy(GoalProxSettings())
