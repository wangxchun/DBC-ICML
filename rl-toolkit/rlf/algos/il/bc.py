import copy

import gym
import numpy as np
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn.functional as F
from rlf.algos.il.base_il import BaseILAlgo
from rlf.args import str2bool
from rlf.storage.base_storage import BaseStorage
from tqdm import tqdm
from dm import ddpm
from dm import ddpm_norm
from dm import ddpm_ant

class BehavioralCloning(BaseILAlgo):
    """
    When used as a standalone updater, BC will perform a single update per call
    to update. The total number of udpates is # epochs * # batches in expert
    dataset. num-steps must be 0 and num-procs 1 as no experience should be collected in the
    environment. To see the performance, you must evaluate. To just evaluate at
    the end of training, set eval-interval to a large number that is greater
    than the number of updates. There will always be a final evaluation.
    """

    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, args):
        super().init(policy, args)
        self.num_epochs = 0
        self.action_dim = rutils.get_ac_dim(self.policy.action_space)
        if self.args.bc_state_norm:
            self.norm_mean = self.expert_stats["state"][0]
            self.norm_var = torch.pow(self.expert_stats["state"][1], 2)
        else:
            self.norm_mean = None
            self.norm_var = None
        self.num_bc_updates = 0

        if self.args.log_diff_loss:
            num_steps = 1000 #sigmoid scheduler
            if args.env_name[:4] == 'maze':
                dim = 8
                self.diff_model = ddpm.MLPDiffusion(num_steps,
                                                    input_dim=dim,
                                                    num_units=self.args.num_units,
                                                    depth=self.args.ddpm_depth).to(self.args.device)
            elif args.env_name[:9] == 'FetchPick':
                dim = 20
                self.diff_model = ddpm.MLPDiffusion(num_steps,
                                                        input_dim=dim,
                                                        num_units=self.args.num_units,
                                                        depth=self.args.ddpm_depth).to(self.args.device)
            elif args.env_name[:9] == 'FetchPush':
                dim = 19
                self.diff_model = ddpm.MLPDiffusion(num_steps, 
                                                        input_dim=dim,
                                                        num_units=self.args.num_units,
                                                        depth=self.args.ddpm_depth).to(self.args.device)
            elif args.env_name[:10] == 'CustomHand':
                dim = 88
                self.diff_model = ddpm.MLPDiffusion(num_steps,
                                                    input_dim=dim,
                                                    nnum_units=self.args.num_units,).to(self.args.device)
            elif args.env_name[:6] == 'Walker':
                dim = 23
                self.diff_model = ddpm.MLPDiffusion(num_steps, 
                                                    input_dim=dim,
                                                    num_units=1024).to(self.args.device)
            elif args.env_name[:11] == 'HalfCheetah':
                dim = 23
                self.diff_model = ddpm.MLPDiffusion(num_steps, 
                                                    input_dim=dim,
                                                    num_units=1024).to(self.args.device)
            elif args.env_name[:3] == 'Ant':
                dim = 50
                self.diff_model = ddpm_ant.MLPDiffusion(num_steps, input_dim = dim).to(self.args.device)
            weight_path = self.args.ddpm_path
            self.diff_model.load_state_dict(torch.load(weight_path))

    def diffusion_loss_fn(self, model, x_0_pred, x_0_expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        batch_size = x_0_pred.shape[0]
        t = torch.randint(0, n_steps, size=(batch_size//2,)).to(self.args.device)
        t = torch.cat([t, n_steps-1-t], dim=0) #[batch_size, 1]
        t = t.unsqueeze(-1)

        # coefficient of x0
        a = alphas_bar_sqrt[t].to(self.args.device)
        
        # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t].to(self.args.device)
        
        # generate random noise eps
        e = torch.randn_like(x_0_pred).to(self.args.device)
        
        # model input
        x = x_0_pred*a + e*aml
        x2 = x_0_expert*a + e*aml

        # get predicted randome noise at time t
        output = model(x, t.squeeze(-1).to(self.args.device))
        output2 = model(x2, t.squeeze(-1).to(self.args.device))
        # print(f"output: {output}")
        # # input()
        # print(f"output2: {output2}")
        # input()
        
        # calculate the loss between actual noise and predicted noise
        loss = (e - output).square().mean()
        loss2 = (e - output2).square().mean()
        return loss, loss2


    def get_density(self, states, pred_action, expert_action):
        #sigmoid scheduler
        num_steps = 1000
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0).to(self.args.device)
        alphas_prod_p = torch.cat([torch.tensor([1]).float().to(self.args.device),alphas_prod[:-1]],0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        
        pred = torch.cat((states, pred_action), 1)
        expert = torch.cat((states, expert_action), 1)
        pred_loss, expert_loss = self.diffusion_loss_fn(self.diff_model, pred, expert, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps) 
        return pred_loss, expert_loss

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            print("Setting environment state normalization")
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs_x = torch.clamp(
            (rutils.get_def_obs(x) - self.norm_mean)
            / torch.pow(self.norm_var + 1e-8, 0.5),
            -10.0,
            10.0,
        )
        if isinstance(x, dict):
            x["observation"] = obs_x
            return x
        else:
            return obs_x

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def full_train(self, update_iter=0):
        action_loss = []
        prev_num = 0

        # First BC
        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step(False)
                action_loss.append(log_vals["_pr_action_loss"])

                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        rutils.plot_line(
            action_loss,
            f"action_loss_{update_iter}.png",
            self.args.vid_dir,
            not self.args.no_wb,
            self.get_completed_update_steps(self.update_i),
        )
        self.num_epochs = 0

    def pre_update(self, cur_update):
        # Override the learning rate decay
        pass

    def _bc_step(self, decay_lr):
        if decay_lr:
            super().pre_update(self.num_bc_updates)
        expert_batch = self._get_next_data()

        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()

        states, true_actions = self._get_data(expert_batch)
        log_dict = {}
        pred_actions, _,  _ = self.policy(states, None, None)
        if rutils.is_discrete(self.policy.action_space):
            pred_label = rutils.get_ac_compact(self.policy.action_space, pred_actions)
            acc = (pred_label == true_actions.long()).sum().float() / pred_label.shape[
                0
            ]
            log_dict["_pr_acc"] = acc.item()
        loss = autils.compute_ac_loss(
            pred_actions,
            true_actions.view(-1, self.action_dim),
            self.policy.action_space,
        )

        self._standard_step(loss)
        self.num_bc_updates += 1

        val_loss = self._compute_val_loss()
        if val_loss is not None:
            log_dict["_pr_val_loss"] = val_loss.item()

        if self.args.log_diff_loss:
            pred_loss, expert_loss = self.get_density(states, pred_actions, true_actions)
            diff_loss_ = torch.clip((pred_loss - expert_loss), min=0)
            log_dict["_pr_diff_loss"] = diff_loss_.item()
        log_dict["_pr_action_loss"] = loss.item()
        return log_dict

    def _get_data(self, batch):
        states = batch["state"].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = batch["actions"].to(self.args.device)
        true_actions = self._adjust_action(true_actions)
        return states, true_actions

    def _compute_val_loss(self):
        if self.update_i % self.args.eval_interval != 0:
            return None
        if self.val_train_loader is None:
            return None
        with torch.no_grad():
            losses = []
            for batch in self.val_train_loader:
                states, true_actions = self._get_data(batch)
                pred_actions, _, _ = self.policy(states, None, None)
                loss = autils.compute_ac_loss(
                    pred_actions,
                    true_actions.view(-1, self.action_dim),
                    self.policy.action_space,
                )
                losses.append(loss.item())

            return np.mean(losses)

    def update(self, storage):
        top_log_vals = super().update(storage)
        log_vals = self._bc_step(True)
        log_vals.update(top_log_vals)
        return log_vals

    def get_storage_buffer(self, policy, envs, args):
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            # This is set when BC is used at the same time as another optimizer
            # that also has a learning rate.
            self.set_arg_prefix("bc")

        super().get_add_args(parser)
        #########################################
        # Overrides
        if self.set_arg_defs:
            parser.add_argument("--num-processes", type=int, default=1)
            parser.add_argument("--num-steps", type=int, default=0)
            ADJUSTED_INTERVAL = 200
            parser.add_argument("--log-interval", type=int, default=ADJUSTED_INTERVAL)
            parser.add_argument(
                "--save-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
            parser.add_argument(
                "--eval-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
        parser.add_argument("--no-wb", default=False, action="store_true")

        #########################################
        # New args
        parser.add_argument("--bc-num-epochs", type=int, default=1)
        parser.add_argument("--bc-state-norm", type=str2bool, default=False)
        parser.add_argument("--bc-noise", type=float, default=None)
        parser.add_argument('--num-units', type=int, default=128) #hidden dim of ddpm
        parser.add_argument('--ddpm-depth', type=int, default=4)
        parser.add_argument('--log-diff-loss', type=str2bool, default=False)
