from __future__ import annotations

import dataclasses
import enum
from typing_extensions import Protocol

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
from ibc.eval import evaluation

from . import experiment, models, optimizers

from rlf.algos.base_algo import BaseAlgo
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.envs.env_interface import get_env_interface

from rlf.policies.basic_policy import BasicPolicy
from functools import partial
from typing import Callable, Optional, Sequence


import inspect
from functools import partial
import rlf.rl.utils as rutils

class CreateAction():
    def __init__(self, action):
        self.action = action 
        self.hxs = {}
        self.extra = {}
        self.take_action = action

class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU

@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU

class ImplicitTrainState(BasicPolicy, nn.Module):
    def __init__(self, 
            model_config: models.MLPConfig,
            optim_config: optimizers.OptimizerConfig,
            stochastic_optim_config: optimizers.DerivativeFreeConfig,
            is_stoch=False,
            fuse_states=[],
            use_goal=False,
            get_base_net_fn=None) -> None:
        
        super().__init__()
        
        device = 'cuda'
        self.steps = 0
        self.model = models.PolicyMLP(config=model_config).cuda()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = optim_config.learning_rate,
            weight_decay = optim_config.weight_decay,
            betas = (optim_config.beta1, optim_config.beta2),
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = optim_config.lr_scheduler_step,
            gamma = optim_config.lr_scheduler_gamma,
        )

        self.stochastic_optimizer = optimizers.DerivativeFreeOptimizer.initialize(
            stochastic_optim_config,
            device,
        )
    '''
    def init(self, obs_space, action_space, args):
        self.action_space = action_space
        self.obs_space = obs_space
        self.args = args

        if 'recurrent' in inspect.getfullargspec(self.get_base_net_fn).args:
            self.get_base_net_fn = partial(self.get_base_net_fn,
                    recurrent=self.args.recurrent_policy)

        if self.use_goal:
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)
            if len(use_obs_shape) != 1:
                raise ValueError(('Goal conditioning only ',
                    'works with flat state representation'))
            
            use_obs_shape = (use_obs_shape[0] + obs_space['desired_goal'].shape[0],)
        else:
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)

        self.base_net = self.get_base_net_fn(use_obs_shape)
        base_out_dim = self.base_net.output_shape[0]
        for k in self.fuse_states:
            if len(obs_space.spaces[k].shape) != 1:
                raise ValueError('Can only fuse 1D states')
            base_out_dim += obs_space.spaces[k].shape[0]
        self.base_out_shape = (base_out_dim,)
    '''

    #def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x,y)
    
    def get_action(self, state, add_state, rnn_hxs, mask, step_info):
        action = self.stochastic_optimizer.infer(state, self.model)
        return CreateAction(action)

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> experiment.TensorboardLogData:
        self.model.train()

        input = input.cuda()
        target = target.cuda()
        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.stochastic_optimizer.sample(input.size(0), self.model)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].cuda()
        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        # import ipdb; ipdb.set_trace()
        energy = self.model(input, targets)
        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)
        wandb.log({'Training loss': loss})
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.steps += 1

        return experiment.TensorboardLogData(
            scalars={
                "train/loss": loss.item(),
                "train/learning_rate": self.scheduler.get_last_lr()[0],
            }
        )

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        # settings = BaseAlgo.get_env_settings()
        env_interface = get_env_interface('maze2d-medium-v2', 1)
        log = BaseLogger()
        ret_info = evaluation(None, self, env_interface, 100, log)
        for k, v in ret_info.items():
            wandb.log({k: v})
        # print("succ_rate:", succ_rate)
        # print("goal_distance:", goal_distance)
        # wandb.log({'succ_rate': succ_rate})
        # wandb.log({'eval_step': sum(eval_step_list) / len(eval_step_list)})

    @torch.no_grad()
    def predict(self, input: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.stochastic_optimizer.infer(input.to(self.device), self.model)
    
    def save(self, epoch, name):
        path = f'ibc/data/{name}'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), f'ibc/data/{name}/model_{epoch}.pt')


class PolicyType(enum.Enum):
    IMPLICIT = ImplicitTrainState
    """An implicit policy is a conditional EBM trained with an InfoNCE objective."""
