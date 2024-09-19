import gymnasium as gym
import numpy as np
import torch
import pandas as pd
from typing import Any, SupportsFloat
from .basis import ProductionFacilities
from .util import huber_loss
import datetime
import pytz
from pathlib import Path


class Dataset:
    """Dataset which models the environmental given data.

    Contains data about wind speed, solar speed and how much electricity is used.

    The calculated factors conceptually are percentages for how much solar or wind is generated right now.
    To normalize the action space of the agent, these factors are scaled in such a way that the means are equal to 1.

    Attributes:
        config: contains names of datasets
        data_dir: path where the datasets are stored
        dataset: the actual dataset after it was constructed
        offset: index 0 may be at any point of the (chronological) data. Wraps around when end is reached
    """

    def __init__(self, dataset_config: dict, offset: float = 0.0):
        self.config = dataset_config
        self.data_dir = Path(dataset_config["data_dir"])

        def consumption_german_to_utc(row) -> datetime.datetime:
            """Convert german time in consumption set to utc

            Args:
                row (): row from pandas dataframe

            Returns:
                datetime object in utc
            """
            german_tz = pytz.timezone("Europe/Berlin")
            utc_time = german_tz.normalize(
                german_tz.localize(
                    datetime.datetime.strptime(
                        f"{row['Datum']}T{row['Anfang']}", "%d.%m.%YT%H:%M"
                    )
                )
            ).astimezone(pytz.utc)
            return utc_time

        def windandsolarset_to_utc(row) -> datetime.datetime:
            """Convert german time in wind- and solar-sets to utc

            Args:
                row (): row from pandas dataframe

            Returns:
                datetime object in utc
            """
            return datetime.datetime.strptime(
                f"{row['MESS_DATUM']}", "%Y%m%d%H"
            ).replace(tzinfo=pytz.utc)

        consumption_set = pd.read_csv(
            self.data_dir / self.config["consumptiondata_path"],
            delimiter=";",
            usecols=[
                "Datum",
                "Anfang",
                "Gesamt (Netzlast) [MWh] Berechnete Auflösungen",
            ],
        )
        consumption_set["consumption [MW]"] = (
            consumption_set["Gesamt (Netzlast) [MWh] Berechnete Auflösungen"]
            .str.replace(".", "")
            .str.replace(",", ".")
            .astype(float)
        )
        consumption_set["timestamp"] = consumption_set.apply(
            consumption_german_to_utc, axis=1
        )
        consumption_set = consumption_set.drop(
            ["Datum", "Anfang", "Gesamt (Netzlast) [MWh] Berechnete Auflösungen"],
            axis=1,
        ).set_index("timestamp", drop=True)

        wind_sets = [
            pd.read_csv(
                self.data_dir / self.config["winddata_paths"][location],
                delimiter=";",
                usecols=["MESS_DATUM", "   F"],
                dtype={"MESS_DATUM": str},
                na_values=("-999"),
            )
            for location in self.config["winddata_paths"].keys()
        ]
        for i in range(len(wind_sets)):
            wind_sets[i]["wind_speed [m/s]"] = wind_sets[i]["   F"].astype(float)
            # replace missing values with mean of neighbors
            wind_sets[i]["wind_speed [m/s]"] = wind_sets[i][
                "wind_speed [m/s]"
            ].interpolate()
            wind_sets[i]["timestamp"] = wind_sets[i].apply(
                windandsolarset_to_utc, axis=1
            )
            wind_sets[i] = (
                wind_sets[i]
                .drop(["MESS_DATUM", "   F"], axis=1)
                .set_index("timestamp", drop=True)
            )
        wind_mean_set = pd.concat(wind_sets).groupby("timestamp").mean()

        solar_sets = [
            pd.read_csv(
                self.data_dir / self.config["solardata_paths"][location],
                delimiter=";",
                usecols=["MESS_DATUM", "FG_LBERG"],
                dtype={"MESS_DATUM": str},
                na_values=("-999"),
            )
            for location in self.config["solardata_paths"].keys()
        ]
        for i in range(len(solar_sets)):
            # replace missing values with mean of neighbors
            solar_sets[i]["FG_LBERG"] = (
                solar_sets[i]["FG_LBERG"].astype(float).interpolate()
            )
            solar_sets[i]["solar_irradiance"] = solar_sets[i]["FG_LBERG"]
            solar_sets[i]["MESS_DATUM"] = solar_sets[i]["MESS_DATUM"].str.replace(
                r":\d+$", "", regex=True
            )
            solar_sets[i]["timestamp"] = solar_sets[i].apply(
                windandsolarset_to_utc, axis=1
            )
            solar_sets[i] = (
                solar_sets[i]
                .drop(["FG_LBERG", "MESS_DATUM"], axis=1)
                .set_index("timestamp", drop=True)
            )
        solar_mean_set = pd.concat(solar_sets).groupby("timestamp").mean()

        self.dataset = consumption_set.join([wind_mean_set, solar_mean_set], how="inner")  # type: ignore
        # scales production factors to interval [0, x] so that its mean is 1
        # this normalizes the actions for the agent, so that building "1" solar, wind, etc. is the same amount of electricity production as building "1" coal, nuclear etc.
        self.dataset["wind_percentage"] = self.dataset["wind_speed [m/s]"].apply(
            self._windspeed_to_percentage
        )
        self.dataset["wind_factor"] = (
            self.dataset["wind_percentage"] / self.dataset["wind_percentage"].mean()
        )
        self.dataset["solar_factor"] = (
            self.dataset["solar_irradiance"] / self.dataset["solar_irradiance"].mean()
        )

        self.offset = round(offset * len(self.dataset))
        return

    def __len__(self) -> int:
        """Length of dataset.
        One item represents an interval of one hour

        Returns:
            length (amount of hours) of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[float, dict[str, float]]:
        """Return one hour interval from dataset.
        Wraps around when end is reached

        Args:
            idx: index to get

        Returns:
            consumption in MW, wind factor in [0, 1], solar factor in [0, 1]
        """
        real_idx = (idx + self.offset) % len(self)
        row = self.dataset.iloc[real_idx]
        return float(row["consumption [MW]"]), {
            "wind": float(row["wind_factor"]),
            "solar": float(row["solar_factor"]),
        }

    def set_offset_factor(self, offset: float = 0.0):
        """Sets the timestamp/position at which the episode starts.

        Args:
            offset (float, optional): Relative positioning in the time series. Defaults to 0.0.
        """
        self.offset = round(offset * len(self))
        return

    def _windspeed_to_percentage(self, windspeed_ground: float) -> float:
        """Models a typical power curve of wind turbines and computes the expected relative power output at the given ground wind speed.

        https://en.wind-turbine-models.com/powercurves e.g. Enercon E-175 EP5
        Computes percentage based on linear spline approximation of power curve.

        Args:
            windspeed_ground (float): Wind speed at ground (10m) in m/s.

        Returns:
            float: Percentage of power output [0,1].
        """
        windspeed = self._windspeed_at_proper_height(windspeed_ground)
        if windspeed < self.config["wind"]["start"]:
            return 0
        if windspeed > self.config["wind"]["cutoff"]:
            return 0
        if windspeed > self.config["wind"]["saturation_start"]:
            return 1

        # linearly approximate if not in one of the other zones
        return (windspeed - self.config["wind"]["start"]) / (
            self.config["wind"]["saturation_start"] - self.config["wind"]["start"]
        )

    def _windspeed_at_proper_height(self, windspeed_ground: float) -> float:
        """Calculates an approximated wind speed at wind turbine hub heights from ground level wind speeds (10m)

        Wind profile power law: https://en.wikipedia.org/wiki/Wind_profile_power_law

        Args:
            windspeed_ground (float): Wind speed at ground (10m) in m/s

        Returns:
            float: Wind speed at wind turbine hub height.
        """
        return (
            windspeed_ground
            * (self.config["wind"]["turbine_height"] / 10)
            ** self.config["wind"]["hellman_exponent"]
        )


class ComplexProductionFacilities(ProductionFacilities):
    """Production Facility wrapper which calculates power output, carbon emissions and power costs.
    Uses the values given in config.
    """

    def carbon_emission_costs(self, factors: dict) -> SupportsFloat:
        """Calculates the carbon emission costs of the currently produced electricity (rescaled to real values) based on the following formula
        and the carbon emission values from the config for each production type:

        Sum{facilities} ( carbon [kg of CO2/MWh] / 1000 * power{facility}[MW] * carbon_price [EUR/ton]

        Returns:
            SupportsFloat: Sum of carbon emission costs in EUR/h
        """
        power_outputs = self._power_output_per_facility(factors)
        emissions_kg_per_facility = [
            self.config["carbon"][facility_name] * power_outputs[facility_name]
            for facility_name in self.facility_idxs.keys()
        ]
        assert np.all(np.array(emissions_kg_per_facility) >= 0)
        return (
            sum(emissions_kg_per_facility) / 1000 * self.config["carbon_costs"]
        )  # in EUR/h

    def production_costs(self, factors) -> SupportsFloat:
        """Computes the levelized cost of electricity (lcoe) of the currently produced electricity (rescaled to real values) based on the following formula
        and the lcoe values from the config for each production type:

        Sum{facilities} ( lcoe [Cent/kWh] * 1000 [->MWh] / 100 [->EUR] * power{facility}[MW] * count{facility}))

        Returns:
            SupportsFloat: Sum of lcoe in EUR/h.
        """
        power_outputs = self._power_output_per_facility(factors)
        eur_per_facility = {
            facility: self.config["lcoe"][facility]
            * (1000 / 100)
            * power_outputs[facility]
            for facility in power_outputs.keys()
        }
        assert np.all(np.array(list(eur_per_facility.values())) >= 0)
        return sum(eur_per_facility.values())  # EUR/h

    def _power_output_per_facility(self, factors: dict[str, float]) -> dict[str, float]:
        """Computes the currently produced electricity (rescaled to real values) of all production facilities with the following formula:

        power{facility} [MW] * count{facility} * factor{facility}

        Args:
            factors (dict): Dictionary with the keys ["solar", "wind'] and the factor [float] of real power output to installed capacity.

        Returns:
            dict: Real power outputs per facility type in the format {"facility": float}
        """
        potential_power = {
            facility_name: self.config["power"][facility_name]
            * facility_count.item()
            * self.config["BIGGEST_STATE_NUM"]
            for facility_name, facility_count in zip(
                self.facility_idxs.keys(), self.facilities
            )
        }
        real_power = potential_power
        for key, factor in factors.items():
            real_power[key] *= factor
        return real_power

    def power_output(self, factors: dict[str, float]) -> SupportsFloat:  # type: ignore # does not match super class
        """Computes the sum of the currently produced electricity (rescaled to real values) of all production facilities with the following formula:

        Sum{facilities} ( power{facility} [MW] * count{facility} * factor{facility})

        The power output of solar and wind fluctuates because of weather influences.

        Args:
            factors (dict | None, optional): Dictionary with the keys ["solar", "wind'] and the factor [float] of real power output to installed capacity. Defaults to None.

        Returns:
            SupportsFloat: Sum of power output in MW.
        """
        return sum(self._power_output_per_facility(factors).values())

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ComplexProductionFacilities":
        """Creates ComplexProductionFacilities object from its tensor representation:

        By BIGGEST_STATE_NUM normalized values of all currently installed production facilities (1 value per type) in the same order as in the config file
        with the following shape: (num_facility_types).

        Args:
            tensor (torch.Tensor): Tensor with shape (num_facility_types)

        Returns:
            ComplexProductionFacilities: ComplexProductionFacilities object.
        """
        new_obj = cls.__new__(cls)  # skip __init__ to reuse tensor
        new_obj.facilities = tensor
        return new_obj


class StorageFacilities:
    """Can store electricity and feed it back into the network

    Attributes:
        config: given constant values
        n: how many storage facilities exist right now
        max_power_throughput_per_facility: how much power one storage facility can output in MW
        capacity_per_facility: how much electricity one storage facility is able to store in MWh
        stored_electricity: actual stored electricity in MWh
    """

    def __init__(self, storage_config: dict, n: int):
        self.config = storage_config
        self.n = n
        self.capacity_per_facility = storage_config["capacity"]
        self.stored_electricity = (
            storage_config["initial_load"] * self.capacity_per_facility * self.n
        )

    def discharge(self, amount: float) -> tuple[float, float]:
        """Discharge power from storage facilities.

        Args:
            amount: amount to discharge if enough stored in MWh

        Returns:
            (amount_withdrawn, carbon_emissions); the amount actually withdrawn in MWh and the corresponding carbon costs in Euro
        """
        amount_withdrawn = min(self.stored_electricity, amount)
        self.stored_electricity -= amount_withdrawn
        return (
            amount_withdrawn,
            amount_withdrawn
            * (self.config["carbon"] / 1000)
            * self.config["carbon_costs"],
        )

    def charge(self, amount: float) -> float:
        """Charge up storage facilities

        Args:
            amount: amount to charge in MWh. If more than storable, throw away rest

        Returns:
            Amount charged
        """
        capacity = self.capacity_per_facility * self.n
        new_stored = min(capacity, self.stored_electricity + amount)
        diff = new_stored - self.stored_electricity
        self.stored_electricity = new_stored
        return diff

    def capacity(self) -> float:
        return self.capacity_per_facility * self.n

    def stored_ratio(self) -> float:
        """How full the storage facilities are.

        Returns:
            factor in [0, 1]
        """
        if self.n == 0:
            return 0
        return self.stored_electricity / (self.capacity_per_facility * self.n)

    def add_action(self, action: float):
        """Execute action to build or demolish storage facilities

        Args:
            action: float in [-1, 1] -> will be scaled up according to configuration
        """
        self.n = min(
            self.config["BIGGEST_STATE_NUM"],
            max(0, self.n + round(action * self.config["BIGGEST_ACTION_NUM"])),
        )
        self.stored_electricity = min(
            self.capacity_per_facility * self.n, self.stored_electricity
        )
        return

    def storage_costs(self) -> float:
        """Storage costs in Euro per hour per MWh

        Returns:
            Storage costs in Euro per hour per MWh
        """
        return (
            self.config["cent_per_day_per_kWh"]
            * (1000 / 100 / 24)
            * self.stored_electricity
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            np.array([self.n / self.config["BIGGEST_STATE_NUM"], self.stored_ratio()]),
            dtype=torch.float32,
        )


class ComplexElectricityGridEnv(gym.Env):
    def __init__(
        self,
        config: dict,
        dataset: Dataset,
        alpha: float = 1.0e-4,
        beta: float = 1.0e-7,
        max_steps: int = 10000,
        production_init_method: str = "zeros",
        use_storage: bool = True,
    ):
        """

        Args:
            config (dict): Config from environment.toml file.
            dataset (Dataset): Time series dataset.
            alpha (float, optional): Coefficient of production loss (squared obsolute power balance). Defaults to 1.0e-3.
            beta (float, optional): Coefficient of carbon emission loss (squared carbon emissions in kg/h). Defaults to 1.0e-6.
            max_steps (int, optional): Maximum episode length. Defaults to 1000.
            production_init_method (str, optional): Initialization method from ['zeros', 'random', 'contemporary]. Defaults to "zeros".
        """
        self.config = config
        self.config["Production"]["carbon_costs"] = config["carbon_costs"]
        self.config["Storage"]["carbon_costs"] = config["carbon_costs"]
        self.dataset = dataset
        self.production_init_method = production_init_method
        self.production = ComplexProductionFacilities(
            self.config["Production"], init_technique=production_init_method
        )
        self.storage = StorageFacilities(self.config["Storage"], n=100)
        self.use_storage = use_storage
        self.alpha = alpha  # weight of grid stability loss
        self.beta = beta  # weight of electricity costs
        self.steps = 0
        self.max_steps = max_steps
        observation_space_dict = {
            "production_facilities": gym.spaces.Box(  # production facilities are in [0, 1] and scaled up according to config
                low=0,
                high=1,
                shape=(ComplexProductionFacilities.__len__(),),
                dtype=np.float32,
            ),
            "potential_power_produced": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "power_produced": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "power_consumed": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "balance": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
        }
        if self.use_storage:
            observation_space_dict["storage_facilities"] = gym.spaces.Box(
                low=0, high=1, shape=(2,), dtype=np.float32
            )  # one for charge percentage and one for built facilities
        self.observation_space = gym.spaces.Dict(observation_space_dict)  # type: ignore
        self.action_space = (
            gym.spaces.Box(  # actions are also scaled by constant in config
                low=-1,
                high=1,
                shape=(
                    (
                        ComplexProductionFacilities.__len__() + 1
                        if self.use_storage
                        else ComplexProductionFacilities.__len__()
                    ),
                ),
                dtype=np.float32,  # + 1 for storage facilities
            )
        )
        self.last_loss = 0.0
        return

    def generate_observation(
        self,
        production: ComplexProductionFacilities,
        storage: StorageFacilities,
        power_produced: float,
        power_consumed: float,
    ) -> dict:
        """Generate observation object on which the agent operates

        Args:
            production: current production facilities
            storage: current storage facilities
            power_produced: total power_produced
            power_consumed: total power_consumed

        Returns:
            observation dictionary
        """
        observation_dict = {
            "production_facilities": production.to_tensor().cpu().numpy(),
            "potential_power_produced": np.array([production.power_output({})], dtype=np.float32),
            "power_produced": np.array([power_produced], dtype=np.float32),
            "power_consumed": np.array([power_consumed], dtype=np.float32),
            "balance": np.array([power_produced - power_consumed], dtype=np.float32),
        }
        if self.use_storage:
            observation_dict["storage_facilities"] = storage.to_tensor().cpu().numpy()
        return observation_dict

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[str, Any]]:
        """Resets the environment

        Args:
            seed (int | None, optional): Seed for determinism. Defaults to None.
            options (dict | None, optional): Unused. Defaults to None.

        Returns:
            tuple[dict, dict[str, Any]]: returns observation dictionary and empty info_dict
        """
        super().reset(seed=seed)
        self.steps = 0

        self.dataset.set_offset_factor(self.np_random.random())

        self.production = ComplexProductionFacilities(
            self.config["Production"],
            seed=self.np_random.integers(np.iinfo(np.int16).max),
            power_goal=self.dataset[0][0],
            init_technique=self.production_init_method,
        )
        self.storage = StorageFacilities(
            self.config["Storage"],
            n=self.np_random.random() * self.config["Storage"]["BIGGEST_STATE_NUM"],
        )
        power_consumed, power_factors = self.dataset[self.steps]
        power_produced = float(self.production.power_output(power_factors))
        power_balance = power_produced - power_consumed
        if power_balance < 0:  # try to get electricity from storage
            storage_power_diff, _ = self.storage.discharge(abs(power_balance))
        else:
            storage_power_diff = -self.storage.charge(power_balance)
        power_produced += storage_power_diff
        power_balance += storage_power_diff

        return (
            self.generate_observation(
                self.production, self.storage, power_produced, power_consumed
            ),
            {},
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        One step/hour in environment

        Args:
            action: build/demolish production buildings ([-1, +1]^n)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        production_action = action[:-1] if self.use_storage else action
        self.production.facilities = torch.clamp(
            self.production.facilities
            + (
                production_action
                * (
                    self.config["Production"]["BIGGEST_ACTION_NUM"]
                    / self.config["Production"]["BIGGEST_STATE_NUM"]
                )
            ),
            min=0,
            max=1,
        )  # scale actions according to config and keep result in [0, 1]
        if self.use_storage:
            self.storage.add_action(action[-1])

        power_consumed, factors = self.dataset[self.steps]
        power_produced = float(self.production.power_output(factors))
        production_carbon_costs = float(self.production.carbon_emission_costs(factors))
        production_costs = float(self.production.production_costs(factors))
        power_balance = power_produced - power_consumed
        if self.use_storage:
            if power_balance < 0:  # try to get electricity from storage
                storage_power_diff, storage_carbon_costs = self.storage.discharge(
                    abs(power_balance)
                )
            else:
                storage_carbon_costs = 0
                storage_power_diff = -self.storage.charge(power_balance)
            power_produced += storage_power_diff
            power_balance += storage_power_diff
            storage_cost = self.storage.storage_costs()
        else:
            storage_carbon_costs = 0
            storage_power_diff = 0
            storage_cost = 0
        carbon_costs = production_carbon_costs + storage_carbon_costs
        # min -> only apply loss if not enough power
        power_loss = (self.alpha * min(power_balance, 0)) ** 2
        cost_loss = (
            self.beta * (carbon_costs + production_costs + storage_cost)
        ) ** 2
        loss = power_loss + cost_loss

        reward = self.last_loss - loss
        self.last_loss = loss
        terminated = False
        truncated = self.steps >= self.max_steps
        observation = self.generate_observation(
            self.production, self.storage, power_produced, power_consumed
        )
        info = {
            "obs": observation,
            "reward": reward,
            "power_loss": power_loss,
            "cost_loss": cost_loss,
            "potential_power_produced": self.production.power_output({}),
            "power_produced": power_produced,
            "power_produced_per_facility": self.production._power_output_per_facility(
                factors
            )
            | {"storage": storage_power_diff},
            "power_consumed": power_consumed,
            "storage_capacity": self.storage.capacity(),
            "storage_store_ratio": self.storage.stored_ratio(),
            "production_cost": production_costs,
            "carbon_cost": carbon_costs,
            "storage_cost": storage_cost,
            "step": self.steps,
        }

        return observation, reward, terminated, truncated, info
