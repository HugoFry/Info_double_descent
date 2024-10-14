import torch
from torch import nn
from MLP import MLP
from config import MINEConfig

class MINENetwork(nn.module):
    def __init__(self, config: MINEConfig):
        super().__init__()
        self.embed = nn.Embedding(config.prime, config.integer_embedding_dimension)
        self.mlp = MLP(config)
    
    def forward(self, model_input, hidden_embedding):
        embedded_input = self.embed(model_input)
        embedded_input = embedded_input.reshape(*embedded_input.shape[:-2], -1)
        concatinated_embedding = torch.cat((embedded_input, hidden_embedding), dim = -1)
        output = self.mlp(concatinated_embedding)
        return output