import torch
import torch.nn as nn
import torch.optim as optim
from src.data.replay_memory import Transition
from src.data.shemas import ConfigData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def optimize_model_with_target(model, target_model, memory, optimizer, cfg: ConfigData):
    if len(memory) < cfg.batch_size:
        return

    transitions = memory.sample(cfg.batch_size)
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

    next_state_values = torch.zeros(cfg.batch_size, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(
            non_final_next_states
        ).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * cfg.gamma) + reward_batch

    # Compute Huber loss
    loss = nn.functional.smooth_l1_loss(
        state_action_values,
        expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    return loss.detach().cpu().item()


def optimize_model_double_dqn(model, target_model, memory, optimizer, cfg: ConfigData):
    if len(memory) < 30 * cfg.batch_size:
        return None

    transitions = memory.sample(cfg.batch_size)
    batch = Transition(*zip(*transitions))

    s0 = torch.cat(batch.state).to(cfg.device)
    a0 = torch.cat(batch.action).to(cfg.device)
    s_ = torch.cat(batch.next_state).to(cfg.device)
    r = torch.cat(batch.reward).to(cfg.device)

    Qsa = model(s0).gather(1, a0)
    a_ = torch.argmax(model(s_), dim=1).view(-1, 1)

    with torch.no_grad():
        newQ = target_model(s_)
    maxQ = newQ.gather(1, a_)

    loss = torch.mean((r + cfg.gamma * maxQ - Qsa).pow(2))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    return loss.detach().cpu().item()
