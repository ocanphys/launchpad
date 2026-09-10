import math


def lr_cosine_schedule(t, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if t < warmup_iters:
        return t * max_learning_rate / warmup_iters
    if t > cosine_cycle_iters:
        return min_learning_rate
    else:
        return (
            min_learning_rate
            + (1 + math.cos((t - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)))
            * (max_learning_rate - min_learning_rate)
            / 2.0
        )
