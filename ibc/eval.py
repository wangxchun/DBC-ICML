import os
import os.path as osp
import time
import wandb
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from rlf.exp_mgr.viz_utils import save_mp4
#from rlf.il.traj_mgr import TrajSaver
from rlf.policies.base_policy import get_empty_step_info
from rlf.rl import utils
#from envs import get_vec_normalize, make_env
from ibc.envs import get_vec_normalize, make_env
#from envs import get_vec_normalize, make_vec_envs
from rlf.envs.env_interface import get_env_interface
from tqdm import tqdm
from rlf.algos.base_algo import BaseAlgo
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.baselines.vec_env.dummy_vec_env import DummyVecEnv


def save_frames(frames, mode, num_steps):
    #args
    eval_save = False
    traj_dir = './data/traj'
    vid_dir = './data/vids'
    env_name = 'maze2d-medium-v2'
    prefix = 'diff-bc'
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

#def get_env_interface(task_id=None):
#    env_interface = get_env_interface('maze2d-medium-v2')(args)
#    env_interface.setup(args, task_id)
#    return env_interface


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
        cur_frame = eval_envs.render(**env_interface.get_render_args(env_interface), **add_kwargs)
    except EOFError as e:
        print("This problem can likely be fixed by setting --eval-num-processes 1")
        raise e

    if not isinstance(cur_frame, list):
        cur_frame = [cur_frame]
    return cur_frame


def evaluation(
    alg_env_settings,
    model,
    env_interface,
    num_steps,
    log,
    mode = 'train',
    create_traj_saver_fn = None,
    num_processes = 1,
    num_eval = 50,
    device = 'cuda'
):
    #args
    eval_save = False
    traj_dir = './data/traj'
    env_name = 'maze2d-medium-v2'
    prefix = 'eng-bc'
    eval_num_processes = 1
    num_render = 1
    eval_only = False
    vid_fps = 30.0
    
    envs = make_env(
        0,
        env_name,
        1,
        True,
        env_interface,
        True,
        alg_env_settings,

    )
    eval_envs = DummyVecEnv([envs])
    assert get_vec_normalize(eval_envs) is None, "Norm is manually applied"

    def obfilt(x, update):
            return x

    eval_episode_rewards = []
    eval_def_stats = defaultdict(list)
    ep_stats = defaultdict(list)

    obs = eval_envs.reset()

    frames = []
    infos = None

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

    goal_achieved = []
    flag = 0
    count_flag = False
    step_num = 0 # the number counting time steps
    eval_step_list = []
    fig, axs = plt.subplots(1, 2, figsize=(28, 3))
    initial_point = obs
    initial_point = torch.from_numpy(initial_point).to('cuda')
    take_action = model.stochastic_optimizer.infer(initial_point, model.model)
    initial_point = initial_point.cpu()
    while evaluated_episode_count < total_num_eval:
        obs = torch.from_numpy(obs).to('cuda')
        #obs = obs.to('cuda')
        take_action = model.stochastic_optimizer.infer(obs, model.model)
        step_info = get_empty_step_info()
        # Observe reward and next obs
        take_action = take_action.cpu()
        next_obs, _, done, infos = eval_envs.step(take_action)

        if eval_save:
            finished_count = traj_saver.collect(
                obs, next_obs, done, take_action, infos
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
        if should_render and flag>=0:
            frames.extend(
                get_render_frames(
                    eval_envs,
                    env_interface,
                    obs,
                    next_obs,
                    take_action,
                    eval_masks,
                    infos,
                    evaluated_episode_count,
                )
            )
        obs = next_obs
        step_log_vals = utils.agg_ep_log_stats(infos, {})
        for k, v in step_log_vals.items():
            ep_stats[k].extend(v)
        
        if count_flag:
            flag  = flag - 1
        goal_achieved.append(infos[0]["goal_achieved"])
        '''
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
            #is_succ = step_log_vals["ep_fsound_goal"][0]
            goal_achieved.append(is_succ)
            goal_distance.append(infos[0]["goal_distance"])
            next_ob =  next_obs.squeeze()
            initial = initial_point.squeeze()
 
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
        '''
    # plt.savefig('goal.png')
    # plt.close()  
    pbar.close()
    if eval_save:
        traj_saver.save()

    ret_info = {}

    print(" Evaluation using %i episodes:" % len(ep_stats["r"]))
    for k, v in ep_stats.items():
        print(" - %s: %.5f" % (k, np.mean(v)))
        ret_info[k] = np.mean(v)
    succ_rate = np.sum(goal_achieved) / num_eval
    print(f'Success rate: {succ_rate}')
    wandb.log({'succ_rate': succ_rate})
    # ret_info['goal_completion'] = succ_rate
    #print("timestep:", sum(eval_step_list) / len(eval_step_list))
    # ret_info['time_step'] = sum(eval_step_list) / len(eval_step_list)
    save_file = save_frames(frames, mode, num_steps)
    if save_file is not None:
        log.log_video(save_file, num_steps, vid_fps)

    #return ret_info, eval_envs, goal_achieved, goal_distance, eval_step_list
    return ret_info

if __name__ == '__main__':
    action = torch.Tensor([1,2])
    settings = BaseAlgo.get_env_settings(None)
    env_interface = get_env_interface('maze2d-medium-v2', 1)
    log = BaseLogger()

    evaluate(
    settings,
    action,
    env_interface,
    100,
    log)
