import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os, sys
import argparse
import numpy as np
import gym
import d4rl # Import required to register environments
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.03 #0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, input_dim=6, num_units=1024, depth=4, device='cuda'):
        super(MLPDiffusion,self).__init__()
        linears_list = []
        linears_list.append(nn.Linear(input_dim, num_units))
        linears_list.append(nn.ReLU())
        if depth > 1:
            for i in range(depth-1):
                linears_list.append(nn.Linear(num_units, num_units))
                linears_list.append(nn.ReLU())
        linears_list.append(nn.Linear(num_units, input_dim))
        self.linears = nn.ModuleList(linears_list).to(device)

        embed_list = []
        for i in range(depth-1):
            embed_list.append(nn.Embedding(n_steps, num_units))
        self.step_embeddings = nn.ModuleList(embed_list).to(device)

    def forward(self, x ,t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)
        return x


def norm_vec(x, mean, std):
    obs_x = torch.clamp((x - mean)
        / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x


########### training loss funciton,  sample at any given time t, and calculate sampling loss
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]
    # generate eandom t for a batch data
    t = torch.randint(0, n_steps, size=(batch_size//2,)).to(device)
    t = torch.cat([t, n_steps-1-t], dim=0) #[batch_size, 1]
    t = t.unsqueeze(-1)
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    # generate random noise eps
    e = torch.randn_like(x_0)
    # model input
    x = x_0*a + e*aml
    # get predicted random noise at time t
    if x.dtype == torch.float64:
        x = x.to(torch.float32)
    output = model(x, t.squeeze(-1))
    # calculate the loss between actual noise and predicted noise
    return (e - output).square().mean()


########### reverse diffusion sample function（inference）
def p_sample_loop(model, shape,n_steps, betas, one_minus_alphas_bar_sqrt):
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas,one_minus_alphas_bar_sqrt):
    # sample reconstruction data at time t drom x[T]
    t = torch.tensor([t]).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x,t)
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)


def reconstruct(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    # generate random t for a batch data
    t = torch.ones_like(x_0, dtype = torch.long).to(device) * n_steps
    # coefficient of x0
    a = alphas_bar_sqrt[t]
    # coefficient of eps
    aml = one_minus_alphas_bar_sqrt[t]
    # generate random noise eps
    e = torch.randn_like(x_0).to(device)
    # model input
    x_T = x_0*a + e*aml
    if x_T.dtype == torch.float64:
        x_T = x_T.to(torch.float32)
    # generate[T-1]、x[T-2]|...x[0] from x[T]
    for i in reversed(range(n_steps)):
        x_T = p_sample(model, x_T, i, betas, one_minus_alphas_bar_sqrt)
    x_construct = x_T
    return x_construct


########### start training, print loss and print the medium reconstrction result
if __name__ == '__main__':
    # Create the environment
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-name', type=str, default='maze2d-medium-v2')
    parser.add_argument('--traj-load-path', type=str, default='/home/mhhsu/dbc-private/expert_datasets/ant_25000.pt')
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler-type', type=str, default='linear')
    args = parser.parse_args()
    print(f'Hidden dimension = {args.hidden_dim}')
    print(f'Depth = {args.depth}')

    ########### hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    num_epoch = 20000

    # decide beta
    if args.scheduler_type == 'linear':
        num_steps = 1000
        betas = linear_beta_schedule(num_steps)
    elif args.scheduler_type == 'cosine':
        num_steps = 100
        betas = cosine_beta_schedule(num_steps)
    elif args.scheduler_type == 'sigmoid':
        num_steps = 1000
        betas = sigmoid_beta_schedule(num_steps)
    elif args.scheduler_type == 'quadratic':
        num_steps = 1000
        betas = quadratic_beta_schedule(num_steps)
    betas = torch.clip(betas, 0.0001, 0.9999).to(device)

    # calculate alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device), alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
    alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
    ==one_minus_alphas_bar_sqrt.shape
    print("all the same shape", betas.shape)

    env = args.traj_load_path.split('/')[-1][:-3]
    data = torch.load(args.traj_load_path)
    # Demonstration normalization
    obs = data['obs']
    actions = data['actions']
    obs = torch.Tensor(obs)
    actions = torch.Tensor(actions)

    dataset = torch.cat((obs, actions), 1)
    sample_num = dataset.size()[0]
    if sample_num % 2 == 1:
        dataset = dataset[1:sample_num, :]
    print("after", dataset.size())
    print("actions.dtype:", actions.dtype)

    print('Training model...')
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True)

    # output dimension is state_dim + action_dim，inputs are x and step
    model = MLPDiffusion(num_steps,
                         input_dim=dataset.shape[1],
                         num_units=args.hidden_dim,
                         depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    train_loss_list = []
    for t in tqdm(range(0, num_epoch)):
        total_loss = 0
        for idx, batch_x in enumerate(dataloader):
            batch_x = batch_x.squeeze().to(device)
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            loss = loss.cpu().detach()
            total_loss += loss
        ave_loss = total_loss / len(dataloader)
        train_loss_list.append(ave_loss)
        if t % 100 == 0:
            print(t, ":", ave_loss)
            with torch.no_grad():
                out = reconstruct(model, dataset.squeeze().to(device), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps-1)
                out = out.cpu().detach()
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                axs[0].scatter(dataset[:300, 0], dataset[:300, 1], color='blue', edgecolor='white')
                axs[1].scatter(out[:300, 0], out[:300, 1], color='red', edgecolor='white')
                plt.savefig(f'{env}-pos.png')
                plt.close()
        if t % 20 == 0:
            train_iteration_list = list(range(len(train_loss_list)))
            plt.plot(train_iteration_list, train_loss_list, color='r')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(env + '_ddpm_loss.png')
            plt.savefig(env + '_ddpm_loss.png')
            plt.close()    
        if t % 1000 == 0:
            torch.save(model.state_dict(), f'{env}_ddpm.pt')
    torch.save(model.state_dict(), f'{env}_ddpm.pt')
