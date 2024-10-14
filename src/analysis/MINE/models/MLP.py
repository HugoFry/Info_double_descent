from dataclasses import dataclass, field
from torch import nn
from config import MINEConfig
import torch
    
class FullyConnectedLayer(nn.module):
    def __init__(self, config: MINEConfig, input_dimensions: int, output_dimensions: int):
        super().__init__()
        self.linear = nn.Linear(input_dimensions, output_dimensions)
        self.batch_norm = nn.BatchNorm1d(output_dimensions)
        if config.activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif config.activation_function == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        else:
            raise Exception(f'Unrecognised activation function {config.activation_function}')
        self.dropout = nn.Dropout(config.dropout_prob)
        
        #Initialise the weights and biases.
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)
        x = self.dropout(x)
        return x
    
class LinearLayer(nn.module):
    def __init__(self, input_dimensions: int, output_dimensions: int):
        super().__init__()
        self.linear = nn.Linear(input_dimensions, output_dimensions)
        
        #Initialise the weights and biases.
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class MLP(nn.module):
    def __init__(self, config: MINEConfig):
        super().__init__()
        self.config = config
        layers = []
        
        #Define all but the last layer
        for i in range(len(config.dimensions)-2):
            layers.append(FullyConnectedLayer(config, config.dimensions[i], config.dimensions[i+1]))
        
        #Just a linear layer for the output layer
        layers.append(LinearLayer(config.dimensions[-2], config.dimensions[-1]))
                
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers[x]