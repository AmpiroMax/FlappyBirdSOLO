import torch
import torch.nn as nn
import torch.optim as optim
from src.data.replay_memory import Transition

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def optimize_model(model, target_model, memory, optimizer, batch_size, gamma, grad_clip):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE, dtype=torch.bool
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(
            non_final_next_states
        ).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = nn.functional.smooth_l1_loss(
        state_action_values,
        expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
    optimizer.step()
