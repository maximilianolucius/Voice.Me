"""Base trainer model definition for Voice.Me TTS.

Provides the abstract base class that all Voice.Me TTS models must inherit,
defining the required interface for model initialization, inference, and
checkpoint loading.
"""

from abc import abstractmethod
from typing import Dict

import torch
from coqpit import Coqpit
from trainer import TrainerModel

# pylint: skip-file


class BaseTrainerModel(TrainerModel):
    """Abstract base model expanding TrainerModel with required functions for Voice.Me TTS.

    Every new Voice.Me TTS model must inherit this class and implement its abstract
    methods to ensure compatibility with the training and inference pipelines.

    Subclasses must implement:
        - ``init_from_config``: Initialize the model from a Coqpit config.
        - ``inference``: Run forward pass for inference.
        - ``load_checkpoint``: Load model weights from a checkpoint file.
    """

    @staticmethod
    @abstractmethod
    def init_from_config(config: Coqpit):
        """Init the model and all its attributes from the given config.

        Override this depending on your model.
        """
        ...

    @abstractmethod
    def inference(self, input: torch.Tensor, aux_input={}) -> Dict:
        """Forward pass for inference.

        It must return a dictionary with the main model output and all the auxiliary outputs. The key ```model_outputs```
        is considered to be the main output and you can add any other auxiliary outputs as you want.

        We don't use `*kwargs` since it is problematic with the TorchScript API.

        Args:
            input (torch.Tensor): [description]
            aux_input (Dict): Auxiliary inputs like speaker embeddings, durations etc.

        Returns:
            Dict: [description]
        """
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    @abstractmethod
    def load_checkpoint(
        self, config: Coqpit, checkpoint_path: str, eval: bool = False, strict: bool = True, cache=False
    ) -> None:
        """Load a model checkpoint gile and get ready for training or inference.

        Args:
            config (Coqpit): Model configuration.
            checkpoint_path (str): Path to the model checkpoint file.
            eval (bool, optional): If true, init model for inference else for training. Defaults to False.
            strict (bool, optional): Match all checkpoint keys to model's keys. Defaults to True.
            cache (bool, optional): If True, cache the file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to False.
        """
        ...
