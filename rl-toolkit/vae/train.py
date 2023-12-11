import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib import MazeDataset as Dataset
from torch.utils.data import DataLoader
from lib import SimpleVAE, FetchVAE
from lib import show
import torch
import argparse

def norm_vec(x, mean, std):
    obs_x = torch.clamp((x - mean)
        / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x

def temp_show(ax, gt_samples, data, model, batch_size, fix=False):
    cur_ax = ax[0]
    cur_ax.clear()
    show(gt_samples, name='GT samples', ax=cur_ax)
    cur_ax = ax[1]
    cur_ax.clear()
    show(data, name='Sampled Z', ax=cur_ax)
    cur_ax = ax[2]
    cur_ax.clear()
    model.show(batch_size, ax=cur_ax)
    if fix:
        plt.show()
    else:
        plt.pause(0.002)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path')
    args = parser.parse_args()
    env = args.traj_load_path.split('/')[-1].split('.')[0]
    # env += '_1e-5'
    data = torch.load(args.traj_load_path)
    obs = data['obs']
    obs_mean = obs.mean(0)
    obs_std = obs.std(0)
    print(f'obs std: {obs_std}')
    if env[:4] == 'push' or  env[:4] == 'pick':
        obs_std[15] = 1
    obs = norm_vec(obs, obs_mean, obs_std)

    actions = data['actions']
    actions_mean = actions.mean(0)
    actions_std = actions.std(0)
    print(f'actions std: {actions_std}')
    actions = norm_vec(actions, actions_mean, actions_std)
    
    dataset = torch.cat((obs, actions), 1)
    dataset = dataset.numpy()

     # visualize ground truth data
    # data = Dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if env[:6] == 'walker':
        in_channels = 23
        model = SimpleVAE(in_channels=in_channels, latent_dim=512)
    elif env[:4] == 'maze':
        in_channels = 8
        model = SimpleVAE(in_channels=in_channels, latent_dim=128)
    elif env[:4] == 'push':
        in_channels = 19
        model = FetchVAE(in_channels=in_channels, latent_dim=512)
    elif env[:4] == 'pick':
        in_channels = 20
        model = FetchVAE(in_channels=in_channels, latent_dim=512)
    print(f'Input channel size: {in_channels}')
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=5000, gamma=.5)

    # training
    batch_size = 128 #512
    iter_num = 100000
    qbar = tqdm(total=iter_num)
    train_loss_list = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
    
    for i in range(iter_num):
        optim.zero_grad()
        gt_samples = next(iter(trainloader)) # batch_size X 2
        tensor_gt_samples = torch.Tensor(gt_samples).to(device)
        tensor_gt_samples = tensor_gt_samples.to(torch.float32)
        forward_res = model(tensor_gt_samples) # [self.decode(z), input, mu, log_var, z] X batch_size

        loss = model.loss_function(forward_res, 0.0000001)
        loss['loss'].backward()

        optim.step()
        scheduler.step()
        train_loss_list.append(loss['loss'].detach().item())
        # model.show()
        #if iter % 200 == 0:
        #    temp_show(ax, gt_samples, forward_res[4].cpu().detach().numpy(), model, batch_size)

        if i % 500 == 0:
            train_iteration_list = list(range(len(train_loss_list[100:])))
            plt.plot(train_iteration_list, train_loss_list[100:], color='r')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'{env}_loss.png')
            plt.savefig(f'{env}_loss.png')
            plt.close()

        if i % 1000 == 0:
            with torch.no_grad():
                # random sample images
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                for idx, batch_x in tqdm(enumerate(dataloader)):
                    batch_x = batch_x.to(torch.float32)
                    out = model(batch_x.to(device))
                    out = out[0].detach().cpu()

                    axs[0].scatter(batch_x[:300, 0], batch_x[:300, 1], color='blue', edgecolor='white')
                    axs[1].scatter(out[:300, 0], out[:300, 1], color='red', edgecolor='white')
                    file_name = f'{env}-reconstruct.png'
                plt.savefig(file_name)
                plt.close()  
            torch.save(model.state_dict(), f'rl-toolkit/vae/trained_models/{env}_vae_norm.pt')
        qbar.update(1)
        # qbar.set_description(desc=f"step: {iter}, lr: {format(optim.param_groups[0]['lr'], '.2e')}, loss: {format(loss['loss'], '.3f')}, Reconstruction_Loss: {format(loss['Reconstruction_Loss'], '.3f')}, KLD: {format(loss['KLD'], '.3f')}.")
    torch.save(model.state_dict(), f'rl-toolkit/vae/trained_models/{env}_vae_norm.pt')
    pass
