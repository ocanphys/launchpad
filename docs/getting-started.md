# Getting started

Launchpad runs ML pipelines on [Modal](https://modal.com), a cloud platform that
runs your Python code on CPUs and GPUs on demand and gives it shared storage. You will need a Modal account and an IDE such as [VS Code](https://code.visualstudio.com) to create your own artifacts and models.

## Setup

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), a Modal account.

```bash
uv sync                      # install dependencies
modal setup                  # log in to Modal (once)

# One secret per Modal workspace: the dashboard password and the JupyterLab token
modal secret create launchpad-lab \
  DASHBOARD_PASSWORD='<YOUR-PASSWORD>' \
  JUPYTER_TOKEN=$(openssl rand -hex 24)

modal deploy main.py         # deploys the dashboard, launcher, workers and lab
```

## Your first workflow

This creates the whole pipeline for a small language model at once, from raw
text to a trained pretraining leg, with the app deployed as in [Setup](#setup).

### 1. Create a new artifact from the dashboard

The dashboard header's **new artifact** link opens JupyterLab. The copy lives on the lab container's own disk, and nothing is written to the volume yet.

### 2. Describe the artifacts

The template's second code cell already describes the full pipeline, one
artifact at a time. Edit it to what you want to train.

**Sources.** Each `SourceURL` downloads one text file to
`sources/<name>/body.txt`. These are three books from Project Gutenberg:

```python
from artifacts.sources import SourceURL

mobydick = SourceURL(name="mobydick", url="https://www.gutenberg.org/cache/epub/2701/pg2701.txt")
montecristo = SourceURL(name="montecristo", url="https://www.gutenberg.org/cache/epub/1184/pg1184.txt")
romeojuliet = SourceURL(name="romeojuliet", url="https://www.gutenberg.org/cache/epub/1513/pg1513.txt")
```

**Tokenizer.** A BPE tokenizer learns its merges from the two training books.
`<|endoftext|>` is the separator the dataset puts between books, so no sequence
runs from one into the next. Learning merges is CPU work, so it asks for four
CPUs:

```python
from artifacts.core.artifact import Resources
from artifacts.tokenizers.bpe import Tokenizer as BPETokenizer

tokenizer = BPETokenizer(vocab_size=10000, special_tokens=("<|endoftext|>",),
                         sources=(mobydick, montecristo),
                         allocated_resources=Resources(cpu=4))
```

**Dataset.** Every book is tokenized with that tokenizer, and the token files
are read in place. Two books train the model and the third validates it:

```python
from artifacts.mappeddataset import MappedDataSet

dataset = MappedDataSet.from_sources(tokenizer=tokenizer,
                                     train_sources=(mobydick, montecristo),
                                     valid_sources=(romeojuliet,))
```

**Pretraining.** The leg names the dataset, the model and how it is trained.
`run_id` groups legs of one run, and `allocated_resources` picks the GPU:

```python
from artifacts.core.SGD.training import LoopConfig, TrainingParameters
from artifacts.stages.pretraining import Pretraining
from models.transformer import Transformer

pretraining = Pretraining(
    run_id="my-run",
    dataset=dataset,
    model=Transformer(vocab_size=tokenizer.vocab_size, d_model=64, d_ff=128,
                      sequence_length=32, num_layers=4, num_heads=16,
                      rope_theta=10000),
    training_parameters=TrainingParameters(total_steps=20000, batch_size=64, ...),
    loop_config=LoopConfig(checkpoint_every=5000, val_every=200, ...),
    allocated_resources=Resources(gpu_type="T4", gpu_count=1),
    tags={"experiment": ["my-experiment"]},
)
```

Naming `dataset` is enough: `pretraining` already carries the dataset, the
tokenizer and the sources beneath it, so you never list them separately.
Nothing has run. These are only definitions, validated as they are built: a
wrong type or an out-of-range value raises at that line.
`pretraining.pprint()` prints the one you made.

### 3. Declare

```python
import lab

report = lab.declare(pretraining, verbose=True)                # preview: writes nothing
report = lab.declare(pretraining, verbose=True, commit=True)   # writes the manifests
```

The preview has one row per artifact with a state ([Lab](lab.md#declaration)
has the full table): `new` artifacts are written on commit, while `conflict`
and `undeclared` block it. A commit with a blocker raises `DeclarationError`
and writes nothing. Save the notebook first: a commit copies it to the
artifact's folder as `declaring.ipynb`, so the recipe outlives the lab's
temporary disk.

From a laptop, use `local.declare_on_volume(pretraining, commit=True)`. It is
the same call, run inside a container that has the volume.

### 4. Run the jobs from the dashboard

Go back to the dashboard. If the new rows are missing, press the refresh
button. Each artifact is one row, and each row walks `blocked` (a dependency is
not done) -> `runnable` -> `starting` -> `running` -> `done`. A `failed` row can
be run again.

Press **run** on a `runnable` row to start that job, and work up the graph:

1. The three sources are runnable at once. Run them.
2. Then the tokenizer, then the tokenized sources, which wait on the tokenizer.
3. The mapped dataset has no job. It is done when its inputs are.
4. Last, the `Pretraining` row. It runs on the GPU it asked for.

Press **stop** on a running row to cancel its job.

### 5. Watch a run

Click an artifact's path in the table to open its page. It shows the
parameters it was declared with, links to its dependencies (click one to jump
to it), and the job's live logs. For the pretraining leg it also draws the
run's metrics: training and validation loss, and the GPU memory and MFU, one
line per attempt. Cancelling a job and running it again resumes from its newest
checkpoint ([Dashboard](dashboard.md#the-artifact-page) has the details).

### 6. Use the result

When the leg is `done`, its row also offers **serve**, which opens the model in
the chat app. In a notebook, a finished artifact loads back from the volume:

```python
tokenizer = tokenizer.bind()    # loads the built files onto a copy
tokenizer.encode("hello")
```

`Artifact.load("tokenizers/<uid>")` reads a declared artifact back without
building anything. `.bind()` additionally requires that it is done and loads
its files. [Lab](lab.md#inspecting-a-trained-model) shows how to rebuild a
trained model from its leg.

## Things worth knowing about training

[Training a Model](training.md) has all of it.

- **GPUs.** Ask for one on the artifact:
  `allocated_resources=Resources(gpu_type="A100")`. With none, the job runs on
  the default CPU pool. Resources are not part of identity, so changing them
  does not create a new artifact, and a declared artifact keeps the resources
  it was declared with.
- **Tags.** Label an artifact with `tags={"project": ["language_modeling"]}` and
  filter the dashboard by them. Like resources they are not identity, and a
  later declaration adds the tags it asks for and removes none.
- **Architectures.** `Transformer` and `Mamba` ship in `models/`, and one
  `Pretraining` class trains either ([Models](models.md)).
- **Checkpoints and resume.** A job writes failsafe checkpoints every
  `checkpoint_every` steps. If it is cancelled or dies, launching it again
  resumes from the newest one.
- **Legs.** A long run can be split into legs. A leg names the previous one as
  `starting_checkpoint` and continues from its `model.pt`.

## Common questions

**Declaring did nothing visible.** Declaring only writes manifests. Launch from
the dashboard to run jobs. A committed declaration shows up in the table on
its own; if it is not there, press the dashboard's refresh button.

**A row is `blocked`.** A dependency is not done. The row's page lists which.

**I changed a parameter and got a new row.** Parameters define identity, so a
changed parameter is a new artifact. That is intended: the old one stays.

**`conflict` on declare.** Two different definitions map to the same path, for
example two sources named `odyssey` with different URLs. Rename one.

**Can I work without Modal?** Declaring and resolving work against any local
folder: pass `root=`. Launching and the dashboard need the deployment.

**Run the tests.** `uv run pytest`.

**Preview these docs.** `mkdocs serve`; a deploy publishes them at `/guide/`.
