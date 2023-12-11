import torch

# Load the model
model = torch.load('/home/mhhsu/dbc-private/rl-toolkit/walker_5traj_processed_ddpm_1e-5.pt')

# Print the model's structure
print(model)

# Print the shape of the model's weights
for param in model.parameters():
    print(param.size())