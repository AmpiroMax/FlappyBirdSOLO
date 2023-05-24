import torch
import albumentations as albu
from src.data.preprocessing import pre_transform, post_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


basic_transformation = albu.Compose([
    pre_transform(), post_transform()
])


def get_state(terminal):
    return torch.tensor(
        list(terminal.state.values()),
        dtype=torch.float32
    ).view(1, -1).to(DEVICE)


def get_image_state(terminal):
    return basic_transformation(image=terminal.image)["image"][0].unsqueeze(0)
