from dataclasses import dataclass, field
from torch import nn
import torch


@dataclass
class MINEConfig():
    integer_embedding_dimension: int = 32
    prime: int = 113
    num_integers: int = 2
    model_embedding_dimension: int = 128
    activation_function: str = 'relu'
    dimensions: list[int] = field(default_factory=lambda: [0, 128, 64, 32, 1])
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01
    
    def post_init(self):
        self.dimensions[0] = self.num_integers * self.integer_embedding_dimension + self.model_embedding_dimension
        assert self.dimensions[0] >= 1
    