from torch import nn
from .config import transformer_config
import torch
import einops
import numpy as np

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_hooks = []
        self.backward_hooks = []
        
    def add_name(self, name):
        # Used to as the key in the model's cache dictionary.
        # Name is the named_modules name of the parent model
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


class Embed(nn.Module):
    def __init__(self, config: transformer_config):
        super().__init__()
        self.embedding = nn.Embedding(config.d_vocab, config.d_model)
    
    def forward(self, x):
        x = self.embedding(x)
        return x
    
    
class Unembed(nn.Module):
    def __init__(self, config: transformer_config):
        super().__init__()
        self.unembedding = nn.Linear(config.d_model, config.d_vocab, bias = False)
    
    def forward(self, x):
        x = self.unembedding(x)
        return x
    

#Learnable positional embeddings.
class PosEmbed(nn.Module):
    def __init__(self, config: transformer_config):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(config.n_ctx, config.d_model)/np.sqrt(config.d_model)) #Xavier initialisation?
    
    def forward(self, x):
        x = x + self.positional_embedding[:x.shape[-2]]
        return x
    

class MLP(nn.Module):
    def __init__(self, config: transformer_config):
        super().__init__()
        self.encoder = nn.Linear(config.d_model, config.d_mlp)
        if config.act_type == "relu":
            self.activation_fn = nn.ReLU()
        elif config.act_type == "gelu":
            self.activation_fn = nn.GELU()
        self.decoder = nn.Linear(config.d_mlp, config.d_model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.hook_pre(x)
        x = self.activation_fn(x)
        x = self.hook_post(x)
        x = self.decoder(x)
        return x
    
    
class Attention(nn.Module):
    def __init__(self, config: transformer_config):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(config.num_heads, config.d_head, config.d_model)/np.sqrt(config.d_model))
        self.K = nn.Parameter(torch.randn(config.num_heads, config.d_head, config.d_model)/np.sqrt(config.d_model))
        self.V = nn.Parameter(torch.randn(config.num_heads, config.d_head, config.d_model)/np.sqrt(config.d_model))
        self.O = nn.Parameter(torch.randn(config.d_model, config.num_heads * config.d_head)/np.sqrt(config.num_heads * config.d_head))
        self.register_buffer('mask', torch.tril(torch.ones(config.n_ctx, config.n_ctx)))
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attention_pre = HookPoint()
        self.hook_attention = HookPoint()
    
    def forward(self, x):
        q = self.hook_q(torch.einsum('nhm,btm->bnth', self.Q, x))
        k = self.hook_k(torch.einsum('nhm,btm->bnth', self.K, x))
        v = self.hook_v(torch.einsum('nhm,btm->bnth', self.V, x))
        attn_pattern_pre = torch.einsum('bnqh,bnkh->bnqk', q, k)
        masked_attn_pattern_pre = self.hook_attention_pre(torch.tril(attn_pattern_pre) - (1 - self.mask[:x.shape[-2], :x.shape[-2]]) * 1e10)
        attn_pattern = self.hook_attention(torch.nn.functional.softmax(masked_attn_pattern_pre, dim = -1))
        z = self.hook_z(torch.einsum('bnqk,bnkh->bnqh', attn_pattern, v))
        z = einops.rearrange(z,'b n q h -> b q (n h)')
        x = torch.einsum('ma,bqa->bqm', self.O, z)
        return x
    
    
class LayerNorm(nn.Module):
    def __init__(self, config: transformer_config, epsilon = 1e-4):
        super().__init__()
        self.use_ln = config.use_ln
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(config.d_model))
        self.bias = nn.Parameter(torch.zeros(config.d_model))
        pass
    
    def forward(self, x):
        if self.use_ln:
            x = x - x.mean(dim = -1, keepdim = True)
            x = x/(x.std(dim = -1, keepdim = True) + self.epsilon)
            x = x * self.weight
            x = x + self.bias
            return x
        else:
            return x
        

class TransformerBlock(nn.Module):
    def __init__(self, config: transformer_config):
        super().__init__()
        self.ln1 = LayerNorm(config)
        self.attention = Attention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        
    def forward(self, x):
        x = self.hook_resid_pre(x)
        x = x + self.hook_attn_out(self.attention(self.ln1(x)))
        x = self.hook_resid_mid(x)
        x = x + self.hook_mlp_out(self.mlp(self.ln2(x)))
        x = self.hook_resid_post(x)
        return x
    

class Transformer(nn.Module):
    def __init__(self, config: transformer_config):
        super().__init__()
        self.config = config
        self.cache = {}
        self.embed = Embed(config)
        self.pos_embed = PosEmbed(config)
        layers = []
        for layer in range(config.num_layers):
            layers.append(TransformerBlock(config))
        self.layers = nn.Sequential(*layers)
        self.unembed = Unembed(config)
        
        for name, module in self.named_modules():
            if 'hook' in name:
                module.add_name(name)
        
    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_embed(x)
        x = self.layers(x)
        x = self.unembed(x)
        return x
    
    def run_with_cache(self, x):
        pass
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]
    
    def remove_all_hooks(self):
        for hook_point in self.hook_points():
            hook_point.remove_hooks('both')
    
    def cache_all(self, include_backward = False):
        def forward_hook_fn(tensor, name):
            self.cache[name] = tensor.detach()
            
        def backward_hook_fn(tensor, name):
            self.chache[name + '_grad'] = tensor[0].detach()
            
        for hook_point in self.hook_points():
            hook_point.create_hook(forward_hook_fn, dir = 'forward')
            if include_backward:
                hook_point.create_hook(backward_hook_fn, dir = 'backward')