from torch import nn
from config import transformer_config
import torch

class HookPoint(nn.module):
    def __init__(self):
        super().__init__()
        self.forward_hooks = []
        self.backward_hooks = []
        pass
    
    def give_name(self, name):
        # Called by the model at initialisation - apparently?
        self.name = name
    
    def create_hook(self, hook, dir = "forward"):
        def hook_fn(module, input, output):
            return hook(output, name=self.name)
        if dir == "forward":
            handle = self.register_forward_hook(hook_fn)
            self.forward_hooks.append(handle)
        elif dir == "backward":
            handle = self.register_backward_hook(hook_fn)
            self.backward_hooks.append(handle)
        else:
            raise ValueError(f"Unrecognised hook direction: {dir}")
    
    def remove_hooks(self, dir = "forward"):
        if dir == "forward" or dir == "both":
            for handle in self.forward_hooks:
                handle.remove()
            self.forward_hooks = []
        if dir == "backward" or dir == "both":
            for handle in self.forward_hooks:
                handle.remove()
            self.forward_hooks = []
            for handle in self.backward_hooks:
                handle.remove()
            self.backward_hooks = []
        if dir not in ['forward', 'backward', 'both']:
            raise ValueError(f"Unrecognised hook direction: {dir}")
    
    def forward(self, x):
        return x

class Embed(nn.module):
    def __init__(self, config: transformer_config):
        super().__init__()
        pass
    
class Unembed(nn.module):
    def __init__(self, config: transformer_config):
        super().__init__()
        pass
    
class PosEmbed(nn.module): #Learnable positional embeddings.
    def __init__(self, config: transformer_config):
        super().__init__()
        pass

class MLP(nn.module):
    def __init__(self, config: transformer_config):
        super().__init__()
        pass
    
class Attention(nn.module):
    def __init__(self, config: transformer_config):
        super().__init__()
        pass
    
class LayerNorm(nn.module):
    def __init__(self, config: transformer_config):
        super().__init__()
        pass

class transformer(nn.module):
    def __init__(self, config: transformer_config):
        super().__init__()
        pass