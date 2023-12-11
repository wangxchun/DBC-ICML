import os
import os.path as osp
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from rlf.exp_mgr.viz_utils import save_mp4
#from rlf.il.traj_mgr import TrajSaver
from rlf.policies.base_policy import get_empty_step_info
from rlf.rl import utils
from envs import get_vec_normalize, make_env
from rlf.envs.env_interface import get_env_interface
from tqdm import tqdm

def save_frames(frames, mode, num_steps, args):
    #args
    eval_save = False
    traj_dir = './data/traj'
    vid_dir = './data/vids'
    env_name = 'maze2d-medium-v2'
    prefix = 'diff-bc'
    render_succ_fails = False
    eval_num_processes = 1
    num_render = 1
    eval_only = False
    vid_fps = 30.0
    load_file = ""

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    add = ""
    if load_file != "":
        add = load_file.split("/")[-2]
        add += "_"

    save_name = "%s%s_%s" % (add, utils.human_format_int(num_steps), mode)

    save_dir = osp.join(vid_dir, env_name, prefix)

    fps = vid_fps

    if len(frames) > 0:
        save_mp4(frames, save_dir, save_name, fps=vid_fps, no_frame_drop=True)
        return osp.join(save_dir, save_name)
    return None

def get_env_interface(self, args, task_id=None):
    env_interface = get_env_interface('maze2d-medium-v2')(args)
    env_interface.setup(args, task_id)
    return env_interface


def get_render_frames(
    eval_envs,
    env_interface,
    obs,
    next_obs,
    action,
    masks,
    infos,
    evaluated_episode_count,
):
    add_kwargs = {}
    '''
    if args.render_metric:
        add_kwargs = {}
        if obs is not None:
            add_kwargs = {
                "obs": utils.ob_to_cpu(obs),
                "action": action.cpu(),
                "next_obs": utils.ob_to_cpu(next_obs),
                "info": infos,
                "next_mask": masks.cpu(),
            }
    '''
    try:
        cur_frame = eval_envs.render(**env_interface.get_render_args(), **add_kwargs)
    except EOFError as e:
        print("This problem can likely be fixed by setting --eval-num-processes 1")
        raise e

    if not isinstance(cur_frame, list):
        cur_frame = [cur_frame]
    return cur_frame

'''
alg_env_settings: AlgorithmSettings(ret_raw_obs=False, state_fn=None, action_fn=None, include_info_keys=[], mod_render_frames_fn=<function mod_render_frames_identity at 0x7fef716ab430>)
policy: BasicPolicy(
  (base_net): MLPBasic(
    (net): Sequential(
      (0): Linear(in_features=6, out_features=256, bias=True)
      (1): Tanh()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): Tanh()
    )
  )
  (action_head): Linear(in_features=256, out_features=2, bias=True)
)
true_vec_norm: None
env_interface: <goal_prox.envs.d4rl.D4rlInterface object at 0x7ff02ffe1220>
num_steps: 44160
mode: train
eval_envs: <rlf.rl.envs.VecPyTorch object at 0x7fef61623520>
log: <rlf.rl.loggers.wb_logger.WbLogger object at 0x7ff032faf070>
create_traj_saver_fn: None
'''

def evaluate(
    alg_env_settings,
    policy,
    true_vec_norm,
    env_interface,
    num_steps,
    mode,
    eval_envs,
    log,
    create_traj_saver_fn = None,
    num_processes = 1,
    num_eval = 100,
    device = 'cuda'
):
    #args
    eval_save = False
    traj_dir = './data/traj'
    env_name = 'maze2d-medium-v2'
    prefix = 'diff-bc'
    render_succ_fails = False
    eval_num_processes = 1
    num_render = 1
    eval_only = False
    vid_fps = 30.0

    env_interface = get_env_interface(args)
    
    eval_envs = make_env(
        rank = 0,
        env_id = 'maze2d-medium-v2',
        seed = 31,
        allow_early_resets = True,
        env_interface = env_interface,
        gamma = 0.99,
        set_eval = True,
        alg_env_settings = alg_env_settings
    )

    #def make_env(rank, env_id, seed, allow_early_resets, env_interface,
    #    set_eval, alg_env_settings, args, immediate_call=False):

    assert get_vec_normalize(eval_envs) is None, "Norm is manually applied"

    if true_vec_norm is not None:
        obfilt = true_vec_norm._obfilt
    else:

        def obfilt(x, update):
            return x

    eval_episode_rewards = []
    eval_def_stats = defaultdict(list)
    ep_stats = defaultdict(list)

    obs = eval_envs.reset()

    hidden_states = {}
    for k, dim in policy.get_storage_hidden_states().items():
        hidden_states[k] = torch.zeros(num_processes, dim).to(device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    frames = []
    infos = None

    policy.eval()
    if eval_save and create_traj_saver_fn is not None:
        traj_saver = create_traj_saver_fn(
            osp.join(traj_dir, env_name, prefix)
        )
    else:
        assert not eval_save, (
            "Cannot save evaluation without ",
            "specifying the eval saver creator function",
        )

    total_num_eval = num_processes * num_eval
    
    # Measure the number of episodes completed
    pbar = tqdm(total=total_num_eval)
    evaluated_episode_count = 0
    n_succs = 0
    n_fails = 0
    succ_frames = []
    fail_frames = []

    if render_succ_fails and eval_num_processes > 1:
        raise ValueError(
            """
                Can only render successes and failures when the number of
                processes is 1.
                """
        )

    if num_render is None or num_render > 0:

        frames.extend(
            get_render_frames(
                eval_envs,
                env_interface,
                None,
                None,
                None,
                None,
                None,
                evaluated_episode_count,
            )
        )
    
    is_succ = False
    goal_achieved = []
    goal_distance = []
    flag = 0
    count_flag = False
    step_num = 0 # the number counting time steps
    eval_step_list = []
    fig, axs = plt.subplots(1, 2, figsize=(28, 3))
    initial_point = obs
    while evaluated_episode_count < total_num_eval:
        step_info = get_empty_step_info()
        with torch.no_grad():
            act_obs = obfilt(utils.ob_to_np(obs), update=False)
            act_obs = utils.ob_to_tensor(act_obs, device)

            ac_info = policy.get_action(
                utils.get_def_obs(act_obs),
                utils.get_other_obs(obs),
                hidden_states,
                eval_masks,
                step_info,
            )
            
            hidden_states = ac_info.hxs

        # Observe reward and next obs
        next_obs, _, done, infos = eval_envs.step(ac_info.take_action)

        if eval_save:
            finished_count = traj_saver.collect(
                obs, next_obs, done, ac_info.take_action, infos
            )
        else:
            finished_count = sum([int(d) for d in done])
            #finished_count = int(infos[0]["goal_achieved"])
        
        pbar.update(finished_count)
        evaluated_episode_count += finished_count

        cur_frame = None

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        should_render = (num_render) is None or (
            evaluated_episode_count < num_render
        )
        if render_succ_fails:
            should_render = n_succs < num_render or n_fails < num_render

        if should_render and flag>=0:
            frames.extend(
                get_render_frames(
                    eval_envs,
                    env_interface,
                    obs,
                    next_obs,
                    ac_info.take_action,
                    eval_masks,
                    infos,
                    evaluated_episode_count,
                )
            )
        obs = next_obs

        step_log_vals = utils.agg_ep_log_stats(infos, ac_info.extra)
        
        for k, v in step_log_vals.items():
            ep_stats[k].extend(v)
        
        if count_flag:
            flag  = flag - 1
        
        if is_succ == False:
            step_num += 1
            if env_name == 'maze2d-medium-v2':
                is_succ = infos[0]["goal_achieved"]
            else:
                is_succ = infos[0]["ep_found_goal"]
            if is_succ:
                flag = 6
                count_flag = True
        
        if finished_count == 1:
            #is_succ = step_log_vals["ep_found_goal"][0]
            goal_achieved.append(is_succ)
            goal_distance.append(infos[0]["goal_distance"])
            next_ob =  next_obs.detach().cpu().numpy().squeeze()
            initial = initial_point.detach().cpu().numpy().squeeze()
 
            if is_succ:
                axs[0].scatter(initial[4], initial[5], color='red', edgecolor='white')
                axs[0].scatter(next_ob[4], next_ob[5], color='blue', edgecolor='white')
                axs[0].plot([initial[4], next_ob[4]], [initial[5], next_ob[5]], color='black')
                axs[0].title.set_text('succ')
            else:
                axs[1].scatter(initial[4], initial[5], color='red', edgecolor='white')
                axs[1].scatter(next_ob[4], next_ob[5], color='blue', edgecolor='white')
                axs[1].plot([initial[4], next_ob[4]], [initial[5], next_ob[5]],  color='black')
                axs[1].title.set_text('fail')
            if eval_only:
                #save_frames(frames, 'each', evaluated_episode_count, args)
                frames = []
            eval_step_list.append(step_num)
            is_succ = False
            flag = 0
            count_flag = False
            step_num = 0 # number counting time-steps return to zero

        if "ep_success" in step_log_vals and render_succ_fails:
            is_succ = step_log_vals["ep_success"][0]
            if is_succ == 1.0:
                if n_succs < num_render:
                    succ_frames.extend(frames)
                n_succs += 1
            else:
                if n_fails < num_render:
                    fail_frames.extend(frames)
                n_fails += 1
    plt.savefig('goal.png')
    plt.close()  
    pbar.close()
    info = {}
    if eval_save:
        traj_saver.save()

    ret_info = {}

    print(" Evaluation using %i episodes:" % len(ep_stats["r"]))
    for k, v in ep_stats.items():
        print(" - %s: %.5f" % (k, np.mean(v)))
        ret_info[k] = np.mean(v)
    
    succ_rate = np.sum(goal_achieved) / num_eval
    ret_info['goal_completion'] = succ_rate
    #print("timestep:", sum(eval_step_list) / len(eval_step_list))
    ret_info['time_step'] = sum(eval_step_list) / len(eval_step_list)

    if render_succ_fails:
        # Render the success and failures to two separate files.
        save_frames(succ_frames, "succ_" + mode, num_steps)
        save_frames(fail_frames, "fail_" + mode, num_steps)
    else:
        save_file = save_frames(frames, mode, num_steps)
        if save_file is not None:
            log.log_video(save_file, num_steps, vid_fps)

    # Switch policy back to train mode
    policy.train()

    return ret_info, eval_envs, goal_achieved, goal_distance, eval_step_list

if __name__ == "__main__":
    evaluate(
    alg_env_settings,
    policy,
    true_vec_norm,
    env_interface,
    num_steps,
    mode,
    eval_envs,
    log,
    create_traj_saver_fn = None,
    num_processes = 1,
    num_eval = 100,
    device = 'cuda')