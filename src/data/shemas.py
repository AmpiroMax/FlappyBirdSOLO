from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ConfigData(BaseModel):
    session_num: int = Field(default=1000, gt=0, type=int)
    gamma: float = Field(default=0.99, ge=0, le=1, type=float)
    lr: float = Field(default=1e-2, gt=0, type=float)
    lr_decay: float = Field(default=0.99, gt=0, le=1, type=float)
    state_dim: int = Field(default=4, gt=0, type=int)
    action_dim: int = Field(default=2, gt=0, type=int)
    hid_dim: List[int] = [64, 256, 256, 64]
    eps_init: float = Field(default=0.5, ge=0, le=1, type=float)
    eps_last: float = Field(default=1e-5, ge=0, lt=1, type=float)
    eps_max_iter: int = Field(default=1000, gt=0, type=int)
    temperature: float = Field(default=1, gt=0, type=float)
    batch_size: int = Field(default=500, gt=1, type=int)
    grad_clip: int = Field(default=50, type=int)
    memory_size: int = Field(default=10000, gt=0, type=int)
    model_load_path: Optional[str] = None
    model_version: str = Field(default='_v0', min_length=3, type=str)
    model_save_path: str = '../models/'
    model_swap_time: int = Field(default=10, gt=0, type=int)
    max_session_score: int = Field(default=150, ge=0, type=int)
    player_flap_acc: int = Field(default=-4, lt=0, type=int)
    reward_threshold: int = -90

    @validator('hid_dim')
    def validate_hid_dim(cls, field):
        
        if not len(field):
            message = 'hid_dim must not be empty;'
            raise ValueError(message)
        
        if not all(map(lambda x: x > 0, field)):
            message = 'all values in hid_dim should be '
            message += 'greater than zero;'
            raise ValueError(message)
        
        return field
