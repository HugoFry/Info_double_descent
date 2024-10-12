from dataclasses import dataclass, field
from torch import nn
from ..models.config import GNN_config as GNN_model_config
import torch


@dataclass
class MLP_config():
    GNN_config: GNN_model_config = GNN_model_config()
    activations_dataset_size: int = 50_000
    activation_function: str = 'relu'
    dimensions: list[int] = field(default_factory=lambda: [200, 256, 128, 64, 1])
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01
    optimizer: str = 'adam'
    
@dataclass
class MINE_config():
    GNN_config: GNN_model_config = GNN_model_config()
    activations_dataset_size: int = 50_000
    activation_function: str = 'relu'
    hidden_dimensions: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0.01
    optimizer: str = 'adam'
    
    def __post_init__(self):
        self.GNN_input_dimensions = self.GNN_config.dimensions[0]
        self.GNN_label_dimensions = self.GNN_config.dimensions[-1]
        
    
    def get_individual_configs(self):
        individual_configs_input_MI = []
        individual_configs_label_MI = []
        for network in range(1,len(self.GNN_config.dimensions), 1):
            #Create the input MI config
            dimensions_input_MI = [self.GNN_input_dimensions + self.GNN_config.dimensions[network], *self.hidden_dimensions, 1]
            input_config = MLP_config(
                GNN_config= self.GNN_config,
                activations_dataset_size = self.activations_dataset_size,
                activation_function = self.activation_function,
                dimensions = dimensions_input_MI,
                dropout_prob = self.dropout_prob,
                learning_rate = self.learning_rate,
                epochs = self.epochs,
                batch_size = self.batch_size,
                weight_decay = self.weight_decay,
                optimizer = self.optimizer,
            )
            individual_configs_input_MI.append(input_config)
            
            #Create the label MI config
            dimensions_label_MI = [self.GNN_label_dimensions + self.GNN_config.dimensions[network], *self.hidden_dimensions, 1]
            label_config = MLP_config(
                GNN_config= self.GNN_config,
                activations_dataset_size = self.activations_dataset_size,
                activation_function = self.activation_function,
                dimensions = dimensions_label_MI,
                dropout_prob = self.dropout_prob,
                learning_rate = self.learning_rate,
                epochs = self.epochs,
                batch_size = self.batch_size,
                weight_decay = self.weight_decay,
                optimizer = self.optimizer,
            )
            individual_configs_label_MI.append(label_config)
            
            return {"input": individual_configs_input_MI, "label": individual_configs_label_MI}