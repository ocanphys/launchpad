"""Golden manifests: the exact path and manifest bytes the code writes for one
artifact of every family, pinned so a refactor can prove it changed neither.

Regenerate only for a change that means to alter paths or bytes:

    python test/test_golden.py
"""

import json
from pathlib import Path

import pytest

from artifacts.core.artifact import Artifact, Resources
from artifacts.core.manifest import manifest_json
from artifacts.core.SGD.training import LoopConfig, LRSchedule, OptimizerParameters, TrainingParameters
from artifacts.dataset import DataSet
from artifacts.mappeddataset import MappedDataSet
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer
from artifacts.tokenizers.template import TemplateTokenizer
from models.mamba import Mamba
from models.transformer import Transformer
from models.transformerV2 import Transformer as TransformerV2

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
COMMIT = "0123456789abcdef0123456789abcdef01234567"  # every node sets it, so no byte depends on HEAD

ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt", commit=COMMIT)
ILIAD = SourceURL(name="iliad", url="https://example.org/iliad.txt.gz", commit=COMMIT)
BPE = Tokenizer(
    vocab_size=1000,
    special_tokens=("<|endoftext|>", "<pad>"),
    sources=(ODYSSEY, ILIAD),
    commit=COMMIT,
    allocated_resources=Resources(cpu=4),
)
TOKENS_ODYSSEY = TokenizedSource(tokenizer=BPE, source=ODYSSEY, commit=COMMIT)
TOKENS_ILIAD = TokenizedSource(tokenizer=BPE, source=ILIAD, commit=COMMIT)
DATASET = DataSet(train_set=(TOKENS_ODYSSEY, TOKENS_ILIAD), valid_set=(TOKENS_ILIAD,), commit=COMMIT)
MAPPED = MappedDataSet(train_set=(TOKENS_ODYSSEY, TOKENS_ILIAD), valid_set=(TOKENS_ILIAD,), commit=COMMIT)

TRAINING = TrainingParameters(
    total_steps=1000,
    batch_size=64,
    max_norm=1,  # an int in a float field: pins that it is written as 1.0
    lr_schedule=LRSchedule(max_learning_rate=3e-4, min_learning_rate=3e-5, warmup_iters=200, cosine_cycle_iters=5000),
    optimizer="torch.optim.AdamW",
    optimizer_parameters=OptimizerParameters(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
)
LOOP = LoopConfig(val_every=200, gpu_check_every=200, checkpoint_every=500)
TRANSFORMER_FIELDS = {
    "vocab_size": 1000, "sequence_length": 32, "num_layers": 4, "d_model": 128, "d_ff": 256, "num_heads": 16,
    "rope_theta": 10000,
}
FIRST_LEG = Pretraining(
    run_id="GOLDEN",
    dataset=MAPPED,
    model=Transformer(**TRANSFORMER_FIELDS),
    training_parameters=TRAINING,
    loop_config=LOOP,
    commit=COMMIT,
    allocated_resources=Resources(gpu_type="T4", gpu_count=1),
)

CASES: dict[str, Artifact] = {
    "source": ODYSSEY,
    "bpe_tokenizer": BPE,
    "template_tokenizer": TemplateTokenizer(
        vocab_size=500, special_tokens=("<unk>",), sources=(ODYSSEY,), commit=COMMIT
    ),
    "tokenized_source": TOKENS_ODYSSEY,
    "dataset": DATASET,
    "mapped_dataset": MAPPED,
    "pretraining_first_leg": FIRST_LEG,
    # no model: inherited from the starting checkpoint
    "pretraining_second_leg": Pretraining(
        run_id="GOLDEN",
        dataset=MAPPED,
        starting_checkpoint=FIRST_LEG,
        training_parameters=TRAINING,
        loop_config=LOOP,
        commit=COMMIT,
    ),
    "pretraining_transformerV2": Pretraining(
        run_id="GOLDEN-V2",
        dataset=DATASET,
        model=TransformerV2(**TRANSFORMER_FIELDS),
        training_parameters=TRAINING,
        loop_config=LOOP,
        commit=COMMIT,
    ),
    "pretraining_mamba": Pretraining(
        run_id="GOLDEN-MAMBA",
        dataset=DATASET,
        model=Mamba(vocab_size=1000, sequence_length=32, num_layers=2, d_model=64),
        training_parameters=TRAINING,
        loop_config=LOOP,
        commit=COMMIT,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_path_and_manifest_bytes_are_unchanged(name):
    artifact = CASES[name]
    paths = json.loads((GOLDEN / "paths.json").read_text())
    assert artifact.artifact_path.as_posix() == paths[name]
    assert manifest_json(artifact.to_manifest()) == (GOLDEN / f"{name}.json").read_text()


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_golden_manifest_decodes_to_its_artifact_and_back_to_its_bytes(name):
    text = (GOLDEN / f"{name}.json").read_text()
    decoded = Artifact.from_manifest(json.loads(text))
    assert decoded == CASES[name]
    assert manifest_json(decoded.to_manifest()) == text


if __name__ == "__main__":
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name, artifact in CASES.items():
        (GOLDEN / f"{name}.json").write_text(manifest_json(artifact.to_manifest()))
    paths = {name: CASES[name].artifact_path.as_posix() for name in sorted(CASES)}
    (GOLDEN / "paths.json").write_text(json.dumps(paths, indent=2) + "\n")
