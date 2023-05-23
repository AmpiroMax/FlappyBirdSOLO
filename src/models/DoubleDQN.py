import typing as tp

import numpy as np
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class QualityEstimator(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hid_dims: tp.List[int]
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
                    nn.LeakyReLU()
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
        action = torch.randint(
            high=len(predicted_actions_rewards),
            size=(1, ),
            dtype=torch.long
        ).view(1, 1).to(state.device)
    else:
        action = torch.argmax(
            predicted_actions_rewards
        ).view(1, 1).to(state.device)

    return action, predicted_actions_rewards[action].view(1, 1).to(state.device)
