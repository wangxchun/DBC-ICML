import sys, os

sys.path.insert(0, "./")

from functools import partial

import d4rl
import torch.nn as nn
from rlf import run_policy, evaluate_policy
from rlf.algos import (GAIL, Diff_gail, PPO, BaseAlgo, BehavioralCloning, Diff_bc,
                       BehavioralCloningFromObs, BehavioralCloningPretrain,
                       GailDiscrim, Ae_bc)
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

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        get_critic_fn=partial(get_sac_critic, hidden_dim=256),
        get_actor_fn=partial(get_sac_actor, hidden_dim=256),
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
                i_shape[0], hidden_size=256, num_layers=2
            ),
        )

    return BasicPolicy()


def get_deep_basic_policy(env_name, args):
    return BasicPolicy(
        get_base_net_fn=lambda i_shape: MLPBase(i_shape[0], False, (512, 512, 256, 128))
    )


def get_setup_dict():
    return {
        "gail": (GAIL(), get_ppo_policy),
        "gail-deep": (GAIL(), get_deep_ppo_policy),
        "diff-gail": (Diff_gail(), get_deep_ppo_policy),
        "uncert-gail-deep": (UncertGAIL(), get_deep_ppo_policy),
        "uncert-gail": (UncertGAIL(), get_ppo_policy),
        "gaifo": (GAIFO(), get_ppo_policy),
        "gaifo-deep": (GAIFO(), get_deep_ppo_policy),
        "ppo": (PPO(), get_ppo_policy),
        "ppo-deep": (PPO(), get_deep_ppo_policy),
        "gw-exp": (BaseAlgo(), lambda env_name, _: GridWorldExpert()),
        "action-replay": (BaseAlgo(), lambda env_name, _: ActionReplayPolicy()),
        "rnd": (BaseAlgo(), lambda env_name, _: RandomPolicy()),
        "bc": (BehavioralCloning(), partial(get_basic_policy, is_stoch=False)),
        "diff-bc": (Diff_bc(), partial(get_basic_policy, is_stoch=False)),
        "ae-bc": (Ae_bc(), partial(get_basic_policy, is_stoch=False)),
        "bco": (BehavioralCloningFromObs(), partial(get_basic_policy, is_stoch=True)),
        "bc-deep": (BehavioralCloning(), get_deep_basic_policy),
        "dpf": (DiscountedProxIL(), get_ppo_policy),
        "dpf-deep": (DiscountedProxIL(), get_deep_ppo_policy),
        "dpf-deep-im": (
            DiscountedProxIL(
                get_pf_base=lambda i_shape: CNNBase(i_shape[0], False, 256),
            ),
            get_deep_ppo_policy,
        ),
        "prox-deep": (ProxAirl(), get_deep_ppo_policy),
        "rpf": (RankedProxIL(), get_ppo_policy),
        "rpf-deep": (RankedProxIL(), get_deep_ppo_policy),
        "sqil-deep": (SQIL(), get_deep_sac_policy),
        "sac": (SAC(), get_deep_sac_policy),
        "goal-gail": (GoalGAIL(), get_deep_ddpg_policy),
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
    evaluate_policy(GoalProxSettings())
