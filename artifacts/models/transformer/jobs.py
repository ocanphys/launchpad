import json
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from artifacts.core.job import Job
from artifacts.core.locate import locate
from artifacts.models.transformer import Pretraining
from artifacts.models.transformer.src import TransformerLM
from artifacts.models.transformer.src.util import seed_everything
if TYPE_CHECKING:
    from system.runtime import Worker


class PretrainJob(Job):
    artifact: Pretraining

    def __init__(self, artifact: Pretraining):
        super().__init__(artifact)
        self.model_class = TransformerLM
        self.model_parameters = artifact.model_parameters
        self.dataset = artifact.dataset
        self.tokenizer = artifact.tokenizer
        self.training_parameters = artifact.training_parameters
        self.loop_config = artifact.loop_config

        self.optimizer = artifact.training_parameters.optimizer
        self.optimizer_parameters = artifact.training_parameters.optimizer_parameters

        seed_everything(self.training_parameters.seed)
        self.train_data = np.memmap(split_bin(self.rdir, "train"), dtype=np.uint16, mode="r")
        self.valid_data = np.memmap(split_bin(self.rdir, "valid"), dtype=np.uint16, mode="r")

        if artifact.starting_checkpoint:
            
        self.model = TransformerLM(**self.model_parameters)
        self.optimizer = self.optimizer(**self.optimizer_parameters)
        
    def run(self, root: Path, worker: "Worker") -> None:
        folder = root / self.artifact.artifact_path
        folder.mkdir(parents=True, exist_ok=True)

        if start_step == 0:
            

        ## load model, optimizers
        dtype = locate(self.artifact.model_parameters.dtype)
        optimizer_cls = locate(self.training_parameters.optimizer)
        schedule_fn = locate(self.training_parameters.lr_schedule.fn)


        worker.log.info()
        worker.progress.update({"step": step, "total_steps": self.training_parameters.total_steps, "loss": loss})
        worker.log.info("training complete")



