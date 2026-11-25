"""The param configs a leg is declared with: each constraint declared on a field
refuses a value at construction, and strict validation converts nothing but an
int in a float field."""

import pytest
from pydantic import ValidationError

from artifacts.core.artifact import Resources
from artifacts.stages.pretraining.evaluations import TokenLoss
from models.mamba import Mamba
from models.transformer import Transformer
from models.transformerFA import TransformerFA
from training.leg import LoopConfig, TrainingParameters
from training.lr_schedule import WarmupCosine
from training.optimizers import AdamW

TRANSFORMER = {
    "vocab_size": 64, "sequence_length": 8, "num_layers": 1, "d_model": 16, "d_ff": 32, "num_heads": 2,
    "rope_theta": 10000,
}
VALID = {
    TrainingParameters: {
        "total_steps": 4,
        "batch_size": 4,
        "max_norm": 1.0,
        "lr_schedule": WarmupCosine(max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_iters=1, cosine_cycle_iters=100),
        "optimizer": AdamW(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
    },
    LoopConfig: {"evaluations": (TokenLoss(eval_every=2, batch_size=4),), "gpu_check_every": 2},
    TokenLoss: {"eval_every": 2, "batch_size": 4},
    Transformer: TRANSFORMER,
    TransformerFA: TRANSFORMER,
    Mamba: {"vocab_size": 64, "sequence_length": 8, "num_layers": 1, "d_model": 16},
}
POSITIVE = {
    TrainingParameters: ("total_steps", "batch_size"),
    LoopConfig: ("gpu_check_every", "checkpoint_every"),
    TokenLoss: ("eval_every", "batch_size"),
    Transformer: ("vocab_size", "sequence_length", "num_layers", "d_model", "d_ff", "num_heads", "rope_theta"),
    TransformerFA: ("vocab_size", "sequence_length", "num_layers", "d_model", "d_ff", "num_heads", "rope_theta"),
    Mamba: ("vocab_size", "sequence_length", "num_layers", "d_model", "d_state", "d_conv", "expand"),
}


@pytest.mark.parametrize(("cls", "field"), [(cls, field) for cls, fields in POSITIVE.items() for field in fields])
@pytest.mark.parametrize("value", [0, -1])
def test_a_field_that_must_be_positive_refuses_zero_and_below(cls, field, value):
    cls(**VALID[cls])
    with pytest.raises(ValidationError, match=field):
        cls(**{**VALID[cls], field: value})


@pytest.mark.parametrize("cls", [Transformer, TransformerFA])
def test_d_model_must_be_divisible_by_num_heads(cls):
    with pytest.raises(ValidationError, match="d_model must be divisible by num_heads"):
        cls(**{**TRANSFORMER, "d_model": 15})


def test_an_int_in_a_float_field_becomes_that_float():
    assert type(TrainingParameters(**{**VALID[TrainingParameters], "max_norm": 1}).max_norm) is float
    assert Resources(cpu=4).cpu == 4.0 and type(Resources(cpu=4).cpu) is float


@pytest.mark.parametrize(
    ("cls", "field", "value"),
    [
        (LoopConfig, "gpu_check_every", True),  # a bool is not an int
        (LoopConfig, "gpu_check_every", "2"),
        (TrainingParameters, "max_norm", "1.0"),
        (TrainingParameters, "max_norm", float("nan")),
        (LoopConfig, "optimizer_checkpoint_policy", "every"),
        (LoopConfig, "unknown", 1),
    ],
)
def test_strict_validation_refuses_anything_it_would_have_to_convert(cls, field, value):
    with pytest.raises(ValidationError, match=field):
        cls(**{**VALID[cls], field: value})


def test_a_param_config_is_frozen():
    loop = LoopConfig(**VALID[LoopConfig])
    with pytest.raises(ValidationError, match="frozen"):
        loop.gpu_check_every = 3
