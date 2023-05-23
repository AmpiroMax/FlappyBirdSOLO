import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_state(terminal):
    return torch.tensor(
        list(terminal.state.values()),
        dtype=torch.float32
    ).view(1, -1).to(DEVICE)
