from abc import abstractmethod, ABC
import torch
import numpy as np
from typing import SupportsFloat
import tomllib

with open("environment.toml", "rb") as config_file:
    env_config = tomllib.load(config_file)


class ConsumptionFacilities:
    facility_idxs = {
        facility: idx for idx, facility in enumerate(["homes", "industrial"])
    }

    def __init__(self, consumption_config: dict):
        self.config = consumption_config
        self.facilities = torch.zeros(len(self.facility_idxs), dtype=torch.float32)
        self["homes"] = consumption_config["power"]["homes"]
        self["industrial"] = consumption_config["power"]["industrial"]
        return

    @classmethod
    def __len__(cls):
        return len(cls.facility_idxs)

    def __getitem__(self, key: str) -> int:
        return int(self.facilities[self.facility_idxs[key]].item())

    def __setitem__(self, key: str, value: int):
        self.facilities[self.facility_idxs[key]] = value
        return

    @abstractmethod
    def power_consumption(self) -> SupportsFloat:
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        return self.facilities

    @staticmethod
    @abstractmethod
    def from_tensor(tensor: torch.Tensor) -> "ConsumptionFacilities":
        pass


class ProductionFacilities:
    facility_idxs = {
        facility: idx
        for idx, facility in enumerate(env_config["Production"]["power"].keys())
    }

    def __init__(
        self,
        production_config: dict,
        seed: int | None = None,
        power_goal: float | None = None,
        init_technique: str = "zeros",
    ):
        """Initializes the production facilities with the provided config

        Args:
            production_config (dict): Production part of the environment.toml config file.
            seed (int | None, optional): Seed for random facility initialization. Defaults to None.
        """
        self.config = production_config

        if init_technique == "zeros":
            self.facilities = torch.zeros(len(self.facility_idxs))
            return
        if init_technique == "random":
            gen = np.random.default_rng(seed=seed)
            self.facilities = (
                torch.tensor(
                    gen.integers(
                        low=1,
                        high=self.config["BIGGEST_STATE_NUM"],
                        size=len(self.facility_idxs),
                    )
                )
                / self.config["BIGGEST_STATE_NUM"]
            )
        elif init_technique == "contemporary":
            self.facilities = torch.tensor(
                list(self.config["init"]["values"].values()), dtype=torch.float32
            )
        # normalize values to match power_goal
        self.facilities *= (
            power_goal / float(self.power_output({})) if power_goal else 1
        )
        return

    @classmethod
    def __len__(cls) -> int:
        """Gets the number of different production facility types.

        Returns:
            int: Number of production facility types.
        """
        return len(cls.facility_idxs)

    @abstractmethod
    def carbon_emission_costs(self, *args) -> SupportsFloat:
        """Getter for carbon emission costs in EUR/h of all production facilities.

        Returns:
            SupportsFloat: Sum of carbon emission costs in EUR/h.
        """
        raise NotImplementedError

    @abstractmethod
    def power_output(self, *args) -> SupportsFloat:
        """Getter for power output in MW of all production facilities.

        Args:
            factors (dict | None): Factors for actually produced electricity of solar and wind facilities.

        Returns:
            SupportsFloat: Sum of real power output in MW.
        """
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        """Creates tensor from current state of production facilities (same order as in config)

        Returns:
            torch.Tensor: Tensor representation of production facilities.
        """
        return self.facilities

    @staticmethod
    @abstractmethod
    def from_tensor(tensor: torch.Tensor) -> "ProductionFacilities":
        """Creates ProductionFacilities object from given tensor.

        Args:
            tensor (torch.Tensor): Tensor representation of state of ProductionFacilities.

        Returns:
            ProductionFacilities: New ProductionFacilities object.
        """
        raise NotImplementedError


class Observation:
    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        """Creates a tensor representation of the observation.

        Returns:
            torch.Tensor: Tensor representation of the observation.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_tensor(tensor: torch.Tensor) -> "Observation":
        """Creates Observation object from given tensor.

        Args:
            tensor (torch.Tensor): Tensor representing a Observation object.

        Returns:
            Observation: Observation object from given tensor.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def __len__(cls) -> int:
        """Getter for length of observation.

        Returns:
            int: Length of observation.
        """
        raise NotImplementedError
