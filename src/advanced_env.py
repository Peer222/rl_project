import gymnasium as gym
import numpy as np
import torch
from typing import Any, SupportsFloat
from .basis import ProductionFacilities, ConsumptionFacilities, Observation
from .util import huber_loss


class AdvancedProductionFacilities(ProductionFacilities):

    def carbon_emission_costs(self) -> SupportsFloat:
        """Calculates the carbon emission costs of the currently produced electricity (rescaled to real values) based on the following formula
        and the carbon emission values from the config for each production type:

        Sum{facilities} ( carbon [kg of CO2/MWh] * power{facility}[MW] * count{facility}) / 1000 * carbon_price [EUR/ton]

        Returns:
            SupportsFloat: Sum of carbon emission costs in EUR/h
        """
        emissions_kg_per_facility = [
            self.config["carbon"][facility_name]
            * self.config["power"][facility_name]
            * facility_count.item()
            * self.config["BIGGEST_STATE_NUM"]
            for facility_name, facility_count in zip(
                self.facility_idxs.keys(), self.facilities
            )
        ]
        assert np.all(np.array(emissions_kg_per_facility) >= 0)
        return (
            sum(emissions_kg_per_facility) / 1000 * self.config["carbon_costs"]
        )  # in EUR/h

    def production_costs(self) -> SupportsFloat:
        """Computes the levelized cost of electricity (lcoe) of the currently produced electricity (rescaled to real values) based on the following formula
        and the lcoe values from the config for each production type:

        Sum{facilities} ( lcoe [Cent/kWh] * 1000 [->MWh] / 100 [->EUR] * power{facility}[MW] * count{facility}))

        Returns:
            SupportsFloat: Sum of lcoe in EUR/h.
        """
        eur_per_facility = [
            self.config["lcoe"][facility_name]
            * (1000 / 100)
            * self.config["power"][facility_name]
            * facility_count.item()
            * self.config["BIGGEST_STATE_NUM"]
            for facility_name, facility_count in zip(
                self.facility_idxs.keys(), self.facilities
            )
        ]
        assert np.all(np.array(eur_per_facility) >= 0)
        return sum(eur_per_facility)

    def power_output(self) -> SupportsFloat:
        """Computes the currently produced electricity (rescaled to real values) of all production facilities with the following formula:

        Sum{facilities} ( power{facility} [MW] * count{facility})

        Returns:
            SupportsFloat: Sum of power output in MW.
        """
        return sum(
            [
                self.config["power"][facility_name]
                * facility_count.item()
                * self.config["BIGGEST_STATE_NUM"]
                for facility_name, facility_count in zip(
                    self.facility_idxs.keys(), self.facilities
                )
            ]
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "AdvancedProductionFacilities":
        """Creates AdvancedProductionFacilities object from its tensor representation:

        By BIGGEST_STATE_NUM normalized values of all currently installed production facilities (1 value per type) in the same order as in the config file
        with the following shape: (num_facility_types).

        Args:
            tensor (torch.Tensor): Tensor with shape (num_facility_types)

        Returns:
            AdvancedProductionFacilities: AdvancedProductionFacilities object.
        """
        new_obj = cls.__new__(cls)  # skip __init__ to reuse tensor
        new_obj.facilities = tensor
        return new_obj


class AdvancedConsumptionFacilities(ConsumptionFacilities):
    def power_consumption(self) -> SupportsFloat:
        """Sum of (fixed) power consumption of households and industry.

        Returns:
            SupportsFloat: Current power consumption in MW
        """
        return sum(
            [
                self.config["power"][facility_name]
                for facility_name in self.facility_idxs.keys()
            ]
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "AdvancedConsumptionFacilities":
        """Create AdvancedConsumptionFacilities from its tensor represenation.

        Args:
            tensor (torch.Tensor): Tensor representation [homes, industrial].

        Returns:
            AdvancedConsumptionFacilities: New AdvancedConsumptionFacilities object
        """
        new_obj = cls.__new__(cls)  # skip __init__ to reuse tensor
        new_obj.facilities = tensor
        return new_obj


class AdvancedObservation(Observation):
    def __init__(
        self,
        production_facilities: AdvancedProductionFacilities,
        consumption_facilities: AdvancedConsumptionFacilities,
    ):
        """

        Args:
            production_facilities (AdvancedProductionFacilities): Production facilities of observation state.
            consumption_facilities (AdvancedConsumptionFacilities): Consumption facilities of observation state.
        """
        self.production_facilities = production_facilities
        self.consumption_facilities = consumption_facilities
        self.balance = float(self.production_facilities.power_output()) - float(
            self.consumption_facilities.power_consumption()
        )
        power_produced = float(self.production_facilities.power_output())
        power_consumed = float(self.consumption_facilities.power_consumption())
        self.balance = power_produced - power_consumed
        return

    @classmethod
    def __len__(cls) -> int:
        """Getter for number of values in observation space.
        Number of producers + number of consumers + balance

        Returns:
            int: Number of values in observation space.
        """
        return (
            AdvancedProductionFacilities.__len__()
            + AdvancedConsumptionFacilities.__len__()
            + 1
        )

    def to_tensor(self) -> torch.Tensor:
        """Creates tensor representation.

        Returns:
            torch.Tensor: Tensor with shape (len(producers) + len(consumers) + 1).
        """
        return torch.cat(
            (
                self.production_facilities.to_tensor(),
                self.consumption_facilities.to_tensor(),
                torch.tensor([self.balance]),
            )
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "AdvancedObservation":
        """Creates AdvancedObservation object from tensor representation.

        Args:
            tensor (torch.Tensor): Tensor with shape (len(producers) + len(consumers) + 1).

        Returns:
            AdvancedObservation: New AdvancedObservation object.
        """
        production_facilities = AdvancedProductionFacilities.from_tensor(
            tensor[: ProductionFacilities.__len__()]
        )
        consumption_facilities = AdvancedConsumptionFacilities.from_tensor(
            tensor[ProductionFacilities.__len__() : -1]
        )
        return AdvancedObservation(production_facilities, consumption_facilities)


class AdvancedElectricityGridEnv(gym.Env):
    def __init__(
        self,
        config: dict,
        alpha: float = 1.0e-6,
        beta: float = 1.0e-9,
        max_steps: int = 1000,
    ):
        """

        Args:
            config (dict): Config from environment.toml file.
            alpha (float, optional): Coefficient of production loss (squared obsolute power balance). Defaults to 1.0e-6.
            beta (float, optional): Coefficient of carbon emission loss (squared carbon emissions in kg/h). Defaults to 1.0e-9.
            max_steps (int, optional): Maximum episode length. Defaults to 1000.
        """
        self.config = config
        self.config["Production"]["carbon_costs"] = config["carbon_costs"]
        self.production = AdvancedProductionFacilities(config["Production"])
        self.consumption = AdvancedConsumptionFacilities(config["Consumption"])
        self.alpha = alpha  # weight of grid stability loss
        self.beta = beta  # weight of electricity costs
        self.steps = 0
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Dict(
            {
                "production_facilities": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(AdvancedProductionFacilities.__len__(),),
                    dtype=np.float32,
                ),
                "balance": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(AdvancedProductionFacilities.__len__(),),
            dtype=np.float32,
        )
        return

    def generate_observation(self, obs: AdvancedObservation) -> dict[str, np.ndarray]:
        """Generates dict representation of current state/observation.

        Args:
            obs (AdvancedObservation): Observation.

        Returns:
            dict: Observation dictionary with the following structure: {"production_facilities": np.array^pn, "consumption_facilities": np.array^cn, "balance": np.array^1}
        """
        return {
            "production_facilities": obs.production_facilities.to_tensor()
            .cpu()
            .numpy(),
            "balance": np.array([obs.balance], dtype=np.float32),
        }

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[str, Any]]:
        """Resets the environment.

        Args:
            seed (int | None, optional): Seed for determinism. Defaults to None.
            options (dict | None, optional): Unused. Defaults to None.

        Returns:
            tuple[dict, dict[str, Any]]: Observation dictionary, empty info_dict.
        """
        super().reset(seed=seed)
        self.steps = 0

        self.production = AdvancedProductionFacilities(self.config["Production"])
        self.consumption = AdvancedConsumptionFacilities(
            self.config["Consumption"]
        )  # just use total values of power usage multiplied by 1

        return (
            self.generate_observation(
                AdvancedObservation(self.production, self.consumption)
            ),
            {},
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Performs an action in the environment and updates its state.

        Args:
            action: build/demolish production buildings ([-1, +1]^n)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        self.production.facilities = torch.clamp(
            self.production.facilities
            + (
                action
                * (
                    self.config["Production"]["BIGGEST_ACTION_NUM"]
                    / self.config["Production"]["BIGGEST_STATE_NUM"]
                )
            ),
            min=0,
            max=1,
        )
        power_produced = float(self.production.power_output())
        power_consumed = float(self.consumption.power_consumption())
        carbon_costs = float(self.production.carbon_emission_costs())
        production_costs = float(self.production.production_costs())
        # min -> only apply loss if not enough power
        power_loss = (self.alpha * min(power_produced - power_consumed, 0)) ** 2
        cost_loss = huber_loss(self.beta * (carbon_costs + production_costs))
        loss = power_loss + cost_loss

        observation = AdvancedObservation(self.production, self.consumption)
        reward = -loss
        terminated = False
        truncated = self.steps >= self.max_steps
        info: dict = {
            "obs": self.generate_observation(observation),
            "reward": reward,
            "power_loss": power_loss,
            "cost_loss": cost_loss,
            "power_produced": power_produced,
            "power_consumed": power_consumed,
            "carbon_cost": carbon_costs,
            "production_cost": production_costs,
            "storage": 0,
            "step": self.steps,
        }
        return (
            self.generate_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )
