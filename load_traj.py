import torch

data = torch.load('expert_datasets/hand_10000_v2.pt')
#data = torch.load('expert_datasets/hand_5586_v2.pt')

obs = data['obs']
next_obs = data['next_obs']
done = data['done']
actions = data['actions']
ep_found_goal = data['ep_found_goal']


indices_of_ones = [index for index, value in enumerate(ep_found_goal) if value == 1]
clip_index = indices_of_ones[300] - 1

obs = obs[:clip_index,:]
next_obs = next_obs[:clip_index,:]
done = done[:clip_index]
actions = actions[:clip_index,:]
ep_found_goal = ep_found_goal[:clip_index]

print("clip_index:", clip_index)
num = clip_index +1

tensor_dict = {
    'obs': obs,
    'next_obs': next_obs,
    'done': done,
    'actions': actions,
    'ep_found_goal': ep_found_goal
}

torch.save(tensor_dict, 'expert_datasets/hand_v2.pt')