import typing as tp

import numpy as np
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class ConvQualityEstimator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        hid_channel: tp.List[int],
        hid_dims: tp.List[int],
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.q_function = nn.Sequential()

        self.q_function.add_module(
            "Img processer",
            nn.Sequential(
                nn.Conv2d(in_channels, hid_channel[0], kernel_size=(5, 5)),
                nn.LeakyReLU()
            )
        )

        for i in range(1, len(hid_channel)):
            self.q_function.add_module(
                f"Conv layer{i}",
                nn.Sequential(
                    nn.Conv2d(
                        hid_channel[i-1],
                        hid_channel[i],
                        kernel_size=(5, 5)
                    ),
                    nn.LeakyReLU(),
                    nn.InstanceNorm2d(hid_channel[i])
                )
            )

        self.q_function.add_module(
            "Pooling layer",
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.AdaptiveAvgPool1d(hid_dims[0])
            )
        )

        for i in range(1, len(hid_dims)):
            self.q_function.add_module(
                f"Linear layer{i}",
                nn.Sequential(
                    nn.Linear(hid_dims[i-1], hid_dims[i]),
                    nn.LeakyReLU(),
                    nn.Dropout1d(p=dropout)
                )
            )

        self.q_function.add_module(
            "Regression layer",
            nn.Sequential(
                nn.Linear(hid_dims[-1], action_dim)
            )
        )

    def forward(self, state: torch.Tensor):
        return self.q_function(state)


class QualityEstimator(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hid_dims: tp.List[int],
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.q_function = nn.Sequential()

        self.q_function.add_module(
            "Input layer",
            nn.Sequential(
                nn.Linear(state_dim, hid_dims[0]),
                nn.LeakyReLU()
            )
        )

        for i in range(1, len(hid_dims)):
            self.q_function.add_module(
                f"Layer{i}",
                nn.Sequential(
                    nn.Linear(hid_dims[i-1], hid_dims[i]),
                    nn.LeakyReLU(),
                    nn.Dropout1d(p=dropout)
                )
            )

        self.q_function.add_module(
            "Regression layer",
            nn.Sequential(
                nn.Linear(hid_dims[-1], action_dim)
            )
        )

    def forward(self, state: torch.Tensor):
        return self.q_function(state)


class DoubleDQN(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hid_dims: tp.List[int]
    ) -> None:
        super().__init__()

        self.qa = QualityEstimator(state_dim, action_dim, hid_dims)
        self.qb = QualityEstimator(state_dim, action_dim, hid_dims)
        self.qa.apply(init_weights)
        self.qb.apply(init_weights)

    def forward(self, state: torch.Tensor):
        return self.qa(state), self.qb(state)


def policy(
    model: QualityEstimator,
    state: torch.Tensor,
    epsilon: float
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """ Epsilon greedy """

    predicted_actions_rewards = model(state)[0]

    if np.random.uniform() < epsilon:
        if np.random.uniform() < 0.1:
            action = torch.tensor(1).view(1, 1).to(state.device)
        else:
            action = torch.tensor(0).view(1, 1).to(state.device)
    else:
        action = torch.argmax(
            predicted_actions_rewards
        ).view(1, 1).to(state.device)

    return action, predicted_actions_rewards[action].view(1, 1).to(state.device)
