from dataclasses import dataclass

@dataclass
class transformer_config():
    has_trained: bool = False
    seed: int = 0
    num_layers: int = 1
    d_model: int = 128
    prime: int = 113
    d_vocab: int = prime + 1 # + 1 originates from the equals sign, needed to ensure learned commutitivity.
    d_mlp: int = 4*d_model
    num_heads: int = 4
    d_head: int = d_model//num_heads # Ensures number of parameters is independent of the number of heads.
    n_ctx: int = 3
    act_type: str = 'relu'
    use_cache: bool = False
    use_ln: bool = False
    lr: float = 1e-3
    weight_decay: float = 1.0
    frac_train: float = 0.3
    num_epochs: int = 100_000
    save_models: bool = False
    save_every: int = 100
    # Stop training when test loss is <stopping_thresh
    stopping_thresh: float = -1
    save_checkpoints: bool = True
    wandb: bool = True
    log_weight_norms: bool = True
    log_activation_norms: bool = True
    log_gradient_norms: bool = True
    log_weight_changes: bool = True
    log_optimizer_moments_norm: bool = True
    
    def __post_init__(self):
        assert self.act_type in ['relu', 'gelu'], f"Activation function {self.act_type} is not recognised."