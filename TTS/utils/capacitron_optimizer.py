"""Voice.Me TTS Capacitron optimizer.

Provides the :class:`CapacitronOptimizer` which manages two separate optimizers
for the Capacitron model: a primary optimizer for the main model parameters and
a secondary optimizer for the VAE beta parameter.
"""

from typing import Generator

from trainer.trainer_utils import get_optimizer


class CapacitronOptimizer:
    """Double optimizer for the Capacitron model.

    Manages a primary optimizer for the main model parameters and a secondary
    optimizer for the Capacitron VAE layer's beta parameter, enabling independent
    learning rate schedules for each group.

    Attributes:
        primary_optimizer: Optimizer for all parameters except the VAE beta.
        secondary_optimizer: Optimizer for the VAE beta parameter.
        param_groups: Parameter groups from the primary optimizer.
    """

    def __init__(self, config: dict, model_params: Generator) -> None:
        self.primary_params, self.secondary_params = self.split_model_parameters(model_params)

        optimizer_names = list(config.optimizer_params.keys())
        optimizer_parameters = list(config.optimizer_params.values())

        self.primary_optimizer = get_optimizer(
            optimizer_names[0],
            optimizer_parameters[0],
            config.lr,
            parameters=self.primary_params,
        )

        self.secondary_optimizer = get_optimizer(
            optimizer_names[1],
            self.extract_optimizer_parameters(optimizer_parameters[1]),
            optimizer_parameters[1]["lr"],
            parameters=self.secondary_params,
        )

        self.param_groups = self.primary_optimizer.param_groups

    def first_step(self):
        """Perform the secondary optimizer step and zero all gradients."""
        self.secondary_optimizer.step()
        self.secondary_optimizer.zero_grad()
        self.primary_optimizer.zero_grad()

    def step(self):
        """Perform the primary optimizer step."""
        # Update param groups to display the correct learning rate
        self.param_groups = self.primary_optimizer.param_groups
        self.primary_optimizer.step()

    def zero_grad(self, set_to_none=False):
        """Zero gradients of both optimizers."""
        self.primary_optimizer.zero_grad(set_to_none)
        self.secondary_optimizer.zero_grad(set_to_none)

    def load_state_dict(self, state_dict):
        """Load state dicts into both optimizers.

        Args:
            state_dict (list): A two-element list of state dicts for primary and secondary optimizers.
        """
        self.primary_optimizer.load_state_dict(state_dict[0])
        self.secondary_optimizer.load_state_dict(state_dict[1])

    def state_dict(self):
        """Return state dicts from both optimizers as a list."""
        return [self.primary_optimizer.state_dict(), self.secondary_optimizer.state_dict()]

    @staticmethod
    def split_model_parameters(model_params: Generator) -> list:
        """Split model parameters into primary and secondary groups.

        The VAE beta parameter goes to the secondary group; all other
        trainable parameters go to the primary group.

        Args:
            model_params (Generator): Named parameters from the model.

        Returns:
            list: [primary_params_iter, secondary_params_iter].
        """
        primary_params = []
        secondary_params = []
        for name, param in model_params:
            if param.requires_grad:
                if name == "capacitron_vae_layer.beta":
                    secondary_params.append(param)
                else:
                    primary_params.append(param)
        return [iter(primary_params), iter(secondary_params)]

    @staticmethod
    def extract_optimizer_parameters(params: dict) -> dict:
        """Extract parameters that are not the learning rate"""
        return {k: v for k, v in params.items() if k != "lr"}
