""" Module of image transformations """

from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albu


IMG_H = 32
IMG_W = 32


def pre_transform() -> albu.BasicTransform:
    """
    Creating preporation transformation before augmentation
    """
    return albu.Resize(IMG_H, IMG_W, always_apply=True)


def post_transform() -> albu.BaseCompose:
    """
    Creating final transformation with normalization and
    casting to torch Tensor
    """
    return albu.Compose([
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            always_apply=True
        ),
        albu.augmentations.transforms.ToGray(always_apply=True),
        ToTensorV2()
    ])
