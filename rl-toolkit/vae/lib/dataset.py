from sklearn.datasets import make_swiss_roll
from .utils import show
import argparse
import torch

#python ddpm.py  --traj-load-path ../expert_datasets/maze2d_100.pt

class DatasetBase(object):
    def gen_data_xy(self, size=512):
        raise NotImplementedError('you need implement gen_data_xy')

    def __len__(self):
        return len(self.samples)

    def show(self, fix=False, ax=None):
        samples = self.gen_data_xy()
        show(samples, type(self).__name__, fix=fix, ax=ax)


class MazeDataset(DatasetBase):
    def gen_data_xy(self, size=1024):
        parser = argparse.ArgumentParser()
        parser.add_argument('--traj-load-path')
        args = parser.parse_args()
        env = args.traj_load_path.split('/')[-1].split('.')[0]

        data = torch.load(args.traj_load_path)
        obs = data['obs']
        actions = data['actions']
        dataset = torch.cat((obs, actions), 1)

        '''
        sample_num = dataset.size()[0]
        if  sample_num % 2 == 1:
            dataset = dataset[1:sample_num, :]
        print("after", dataset.size())
        '''

        return dataset
