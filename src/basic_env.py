import gymnasium as gym
import numpy as np
import torch
from typing import Any, SupportsFloat
from .basis import ProductionFacilities, ConsumptionFacilities, Observation


class BasicProductionFacilities(ProductionFacilities):
    def carbon_emission_costs(self) -> SupportsFloat:
        """Computes carbon emission costs based on current produced electricity and carbon costs from config.

        Returns:
            SupportsFloat: Carbon emission costs in kilograms/h of CO2
        """
        return sum(
            [
                self.config["carbon"][facility_name]
                * self.config["power"][facility_name]
                * facility_count.item()
                * self.config["BIGGEST_STATE_NUM"]
                for facility_name, facility_count in zip(
                    self.facility_idxs.keys(), self.facilities
                )
            ]
        )

    def power_output(self) -> SupportsFloat:
        """Computes the power output of all production facilities.

        Returns:
            SupportsFloat: Power output in MW
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
    def from_tensor(cls, tensor: torch.Tensor) -> "BasicProductionFacilities":
        """Creates BasicProductionFacilities object from tensor with order of facilities from config.

        Args:
            tensor (torch.Tensor): Tensor representation of ProductionFacilities.

        Returns:
            BasicProductionFacilities: new BasicProductionFacilities object.
        """
        new_obj = cls.__new__(cls)  # skip __init__ to reuse tensor
        new_obj.facilities = tensor
        return new_obj


class BasicConsumptionFacilities(ConsumptionFacilities):
    def power_consumption(self) -> SupportsFloat:
        """Getter for power consumption from consumers (homes and industry)

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
    def from_tensor(cls, tensor: torch.Tensor) -> "BasicConsumptionFacilities":
        """Creates BasicConsumptionFacilities object from tensor.

        Args:
            tensor (torch.Tensor): Tensor representation of BasicConsumptionFacilities object [homes, industrial].

        Returns:
            BasicConsumptionFacilities: New BasicConsumptionFacility object.
        """
        new_obj = cls.__new__(cls)  # skip __init__ to reuse tensor
        new_obj.facilities = tensor
        return new_obj


class BasicObservation(Observation):
    def __init__(
        self,
        production_facilities: BasicProductionFacilities,
        consumption_facilities: BasicConsumptionFacilities,
    ):
        """Saves the state of the production and consumption facilities and calculates/saves their balance.

        Args:
            production_facilities (BasicProductionFacilities): Production facilities.
            consumption_facilities (BasicConsumptionFacilities): Consumption facilities.
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
        """Length of Observation state (number of producers + number of consumers)

        Returns:
            int: Length.
        """
        return (
            BasicProductionFacilities.__len__()
            + BasicConsumptionFacilities.__len__()
            + 1
        )

    def to_tensor(self) -> torch.Tensor:
        """Creates tensor of current state with producers, consumers and balance (concatenated into single dimension)

        Returns:
            torch.Tensor: Flat tensor with producers, consumers and balance.
        """
        return torch.cat(
            (
                self.production_facilities.to_tensor(),
                self.consumption_facilities.to_tensor(),
                torch.tensor([self.balance]),
            )
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "BasicObservation":
        """Creates BasicObservation from tensor representation.

        Args:
            tensor (torch.Tensor): Flat tensor with [producers, consumers, balance]

        Returns:
            BasicObservation: New BasicObservation object.
        """
        production_facilities = BasicProductionFacilities.from_tensor(
            tensor[: ProductionFacilities.__len__()]
        )
        consumption_facilities = BasicConsumptionFacilities.from_tensor(
            tensor[ProductionFacilities.__len__() : -1]
        )
        return BasicObservation(production_facilities, consumption_facilities)


class BasicElectricityGridEnv(gym.Env):
    def __init__(
        self,
        config: dict,
        alpha: float = 1.0,
        beta: float = 0.01,
        max_steps: int = 1000,
    ):
        """

        Args:
            config (dict): Config from environment.toml.
            alpha (float, optional): Coefficient of production loss (squared obsolute power balance). Defaults to 1.0.
            beta (float, optional): Coefficient of carbon emission loss (squared carbon emissions in kg/h). Defaults to 0.01.
            max_steps (int, optional): Maximum episode length. Defaults to 1000.
        """
        self.config = config
        self.production = BasicProductionFacilities(config["Production"])
        self.consumption = BasicConsumptionFacilities(config["Consumption"])
        self.alpha = alpha  # weight of electricity balance loss
        self.beta = beta  # weight of carbon emission loss
        self.steps = 0
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Dict(
            {
                "production_facilities": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(BasicProductionFacilities.__len__(),),
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
            shape=(BasicProductionFacilities.__len__(),),
            dtype=np.float32,
        )
        return

    def generate_observation(self, obs: BasicObservation) -> dict[str, np.ndarray]:
        """Creates a dictionary representing the current observation state.

        Args:
            obs (BasicObservation)

        Returns:
            dict: Observation with the following structure: {"production_facilities": np.ndarray, "consumption_facilities": np.ndarray, "balance": np.ndarray}
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
            seed (int | None, optional): Seed for resetting the environment. Defaults to None.
            options (dict | None, optional): Unused. Defaults to None.

        Returns:
            tuple[dict, dict[str, Any]]: Observation with the following structure {"production_facilities": np.ndarray, "consumption_facilities": np.ndarray, "balance": np.ndarray} and an empty info dict.
        """
        super().reset(seed=seed)
        self.steps = 0

        self.production = BasicProductionFacilities(
            self.config["Production"], self.np_random.integers(np.iinfo(np.int16).max)
        )
        self.consumption = BasicConsumptionFacilities(
            self.config["Consumption"]
        )  # just use total values of power usage multiplied by 1

        return (
            self.generate_observation(
                BasicObservation(self.production, self.consumption)
            ),
            {},
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Takes an action in the environment and updates its internal state appropiately.

        Format of the observation_dict: {"production_facilities": np.ndarray, "consumption_facilities": np.ndarray, "balance": np.ndarray}
        The reward is calculated by the negative weighted sum of the squared balance of produced and consumed energy and squared emissions.
        Info is always empty

        Args:
            action (np.ndarray): The action that should be taken for each production facility.

        Returns:
            tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]: observation_dict, reward, terminated, truncated, info.
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
        carbon_emissions = float(self.production.carbon_emission_costs())
        loss = (
            self.alpha * abs(power_produced - power_consumed) ** 2
            + self.beta * carbon_emissions**2
        )

        observation = BasicObservation(self.production, self.consumption)
        reward = -loss
        terminated = False
        truncated = self.steps >= self.max_steps
        info: dict = {}
        if truncated:
            print(self.production.to_tensor())
            print(observation.balance / (power_produced + power_consumed) * 100)
        return (
            self.generate_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )
