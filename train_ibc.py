"""Script for training. Pass in --help flag for options."""

import dataclasses
from typing import Dict, Optional, Tuple
import numpy as np
import dcargs
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import wandb
from tqdm.auto import tqdm

from ibc import dataset, models, optimizers, trainer, utils
from ibc.experiment import Experiment


@dataclasses.dataclass
class TrainConfig:
    experiment_name: str
    seed: int = 0
    device_type: str = "cuda"
    train_dataset_size: int = 10
    test_dataset_size: int = 500
    max_epochs: int = 200
    lr: float = 0.0005
    weight_decay: float = 0.0
    train_batch_size: int = 128
    test_batch_size: int = 64
    spatial_reduction: models.SpatialReduction = models.SpatialReduction.SPATIAL_SOFTMAX
    coord_conv: bool = False
    dropout_prob: Optional[float] = None
    num_workers: int = 1
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 100
    eval_every_n_steps: int = 50
    policy_type: trainer.PolicyType = trainer.PolicyType.IMPLICIT
    stochastic_optimizer_train_samples: int = 64

class TrajDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
        self.state = self.data['obs']
        self.action = self.data['actions']
        self.length = self.state.shape[0]
        print(self.state.shape) 
        print(self.action.shape) 
    def get_target_bounds(self):
        return np.array([-np.ones(self.action.shape[1]),
                         np.ones(self.action.shape[1])])
    
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.state[index], self.action[index]


def make_dataloaders(
    train_config: TrainConfig,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Initialize train/test dataloaders based on config values."""
    # Train split.
    train_dataset_config = dataset.DatasetConfig(
        dataset_size=train_config.train_dataset_size,
        seed=train_config.seed,
    )
    # train_dataset = dataset.CoordinateRegression(train_dataset_config)
    train_dataset = TrajDataset('./expert_datasets/maze2d_100.pt')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataloader


def make_train_state(
    train_config: TrainConfig,
    train_dataloader: torch.utils.data.DataLoader,):
    """Initialize train state based on config values."""
    state_dim = 6
    action_dim = 2
    input_dim = state_dim + action_dim
    output_dim = 1
    hidden_dim = 128
    depth = 4
    mlp_config = models.MLPConfig(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = 1,
        hidden_depth = depth,
        dropout_prob = 0,
    )
    optim_config = optimizers.OptimizerConfig(
        learning_rate=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    target_bounds = np.array([-np.ones(action_dim), np.ones(action_dim)])
    stochastic_optim_config = optimizers.DerivativeFreeConfig(
        bounds=target_bounds,
        train_samples=train_config.stochastic_optimizer_train_samples,
    )
    train_state = trainer.ImplicitTrainState(
        model_config=mlp_config,
        optim_config=optim_config,
        stochastic_optim_config=stochastic_optim_config,
    )
    return train_state


def main(train_config: TrainConfig) -> None:
    # Seed RNGs.
    utils.seed_rngs(train_config.seed)

    # CUDA/CUDNN-related shenanigans.
    utils.set_cudnn(train_config.cudnn_deterministic, train_config.cudnn_benchmark)

    experiment = Experiment(
        identifier=train_config.experiment_name,
    ).assert_new()

    # Write some metadata.
    experiment.write_metadata("config", train_config)

    # Initialize train and test dataloaders.
    dataloader = make_dataloaders(train_config)
    train_state = make_train_state(train_config, dataloader)
    name = f'128_4'
    for epoch in tqdm(range(train_config.max_epochs)):
        for batch in dataloader:
            train_log_data = train_state.training_step(*batch)
        if not (epoch+1) % train_config.eval_every_n_steps:
            train_state.evaluate()
        if not epoch % train_config.checkpoint_every_n_steps:
            train_state.save(epoch, name)
    # Save one final checkpoint.
    train_state.save(epoch, name)


if __name__ == "__main__":
    run = wandb.init(project='ibc')
    main(dcargs.parse(TrainConfig, description=__doc__))
