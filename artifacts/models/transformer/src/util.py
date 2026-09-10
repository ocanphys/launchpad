import torch

def seed_everything(seed: int) -> None:
    """Seed every RNG that affects model initialization (weights in every
    Linear/Embedding/MultiHeadAttention/SwiGLU), on both CPU and CUDA."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
