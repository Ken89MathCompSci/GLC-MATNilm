import torch
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainConfig:
    input_size: int
    batch_size: int
    hidden: int
    lr: float
    dropout: float
    logname: str
    outputLength: int
    inputLength: int
    subName: str
    dataAug: bool
    prob0: float
    prob1: float
    prob2: float
    prob3: float
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainConfig':
        return cls(**config_dict)

class Basic:
    def __init__(self, model, optimizer):
        self.model = model
        self.model_opt = optimizer
