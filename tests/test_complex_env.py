import numpy as np
import tomllib
import copy
import pytest

from stable_baselines3.common.env_checker import check_env

from src.complex_env import *

with open("environment.toml", "rb") as config_file:
    config = tomllib.load(config_file)


class TestDataset:
    def test_dataset(self):
        dataset1 = Dataset(config["Dataset"])
        dataset2 = copy.deepcopy(dataset1)
        dataset2.set_offset_factor(0.5)

        # test samples
        for idx in range(100):
            sample = dataset1[idx]
            assert sample[0] >= 0
            assert sample[1]["wind"] >= 0
            assert sample[1]["solar"] >= 0

        # test offsets
        assert dataset1[0] != dataset2[0]

        # test windpercentage
        assert dataset1._windspeed_to_percentage(0) == 0
        assert dataset1._windspeed_to_percentage(1000) == 0
        for num in range(1000):
            assert (
                dataset1._windspeed_to_percentage(num) >= 0
                and dataset1._windspeed_to_percentage(num) <= 1
            )


class TestComplexProductionFacilities:
    def test_init(self):
        prod1 = ComplexProductionFacilities(
            config["Production"], init_technique="zeros"
        )
        prod2 = ComplexProductionFacilities(
            config["Production"], init_technique="random"
        )
        prod3 = ComplexProductionFacilities(
            config["Production"], init_technique="contemporary"
        )

    def test_power_output(self):
        prod1 = ComplexProductionFacilities(
            config["Production"], init_technique="zeros"
        )
        assert prod1.power_output({}) == 0
        prod1.facilities[prod1.facility_idxs["wind"]] = 1
        assert float(prod1.power_output({})) > 0
        assert float(prod1.power_output({"wind": 0})) == 0

    def test_power_goal(self):
        power_goal = 10000
        prod1 = ComplexProductionFacilities(
            config["Production"], init_technique="random", power_goal=power_goal
        )
        prod2 = ComplexProductionFacilities(
            config["Production"], init_technique="contemporary", power_goal=power_goal
        )
        assert prod1.power_output({}) == pytest.approx(power_goal)
        assert prod2.power_output({}) == pytest.approx(power_goal)

    def test_carbon_emission(self):
        config["Production"]["carbon_costs"] = 1
        prod1 = ComplexProductionFacilities(
            config["Production"], init_technique="zeros"
        )
        assert prod1.carbon_emission_costs({}) == 0
        prod1.facilities[prod1.facility_idxs["wind"]] = 1
        assert float(prod1.carbon_emission_costs({})) > 0
        assert float(prod1.carbon_emission_costs({"wind": 0})) == 0


class TestStorageFacilities:
    def test_discharge(self):
        config["Storage"]["carbon_costs"] = 1
        config["Storage"]["carbon"] = 1 * config["Storage"]["capacity"]
        stor1 = StorageFacilities(config["Storage"], 10)
        stored_elec = stor1.stored_electricity
        assert stor1.discharge(stored_elec) == (stored_elec, stored_elec)
        stor1.stored_electricity = stored_elec
        assert stor1.discharge(stored_elec + 1) == (stored_elec, stored_elec)
        stor1.stored_electricity = stored_elec
        assert stor1.discharge(1) == (1, 1)

    def test_charge(self):
        config["Storage"]["carbon_costs"] = 1
        stor1 = StorageFacilities(config["Storage"], 1)
        stor1.capacity_per_facility = 1000
        stor1.stored_electricity = 0
        assert stor1.charge(1000) == 1000
        assert stor1.stored_electricity == 1000
        assert stor1.charge(1000) == 0
        assert stor1.stored_electricity == 1000
        stor1.stored_electricity = 500
        assert stor1.charge(1000) == 500
        assert stor1.stored_electricity == 1000
        stor1.stored_electricity = 0
        assert stor1.charge(5000) == 1000
        assert stor1.stored_electricity == 1000

    def test_stored_ratio(self):
        stor1 = StorageFacilities(config["Storage"], 10)
        stor1.stored_electricity = 0
        assert stor1.stored_ratio() == 0
        stor1.stored_electricity = stor1.capacity_per_facility * stor1.n
        assert stor1.stored_ratio() == 1
        stor1.stored_electricity = stor1.capacity_per_facility * stor1.n * (1 / 2)
        assert stor1.stored_ratio() == 0.5

    def test_add_action(self):
        stor1 = StorageFacilities(config["Storage"], 10)
        rand_actions = (np.random.random(size=100) - 0.5) * 2
        for action in rand_actions:
            stor2 = StorageFacilities(config["Storage"], 10)
            stor1.add_action(action)
            assert stor1.n >= 0 and stor1.n <= config["Storage"]["BIGGEST_STATE_NUM"]
            stor2.add_action(action)
            assert stor2.n >= 0 and stor2.n <= config["Storage"]["BIGGEST_STATE_NUM"]


class TestComplexElectricityGridEnv:
    def test_stable_baselines_check_env(self):
        dataset = Dataset(config["Dataset"])
        check_env(ComplexElectricityGridEnv(config, dataset))

    def test_check_max_steps(self):
        """
        max_steps is always steps, as we do not have a termination condition
        """
        dataset = Dataset(config["Dataset"])
        env1 = ComplexElectricityGridEnv(config, dataset, max_steps=10000)

        i = 0
        done = False
        env1.reset()
        while not done:
            _, _, terminated, truncated, _ = env1.step(env1.action_space.sample())
            done = terminated or truncated
            i += 1
        assert i == 10000

        # test if it works properly after reset
        i = 0
        done = False
        env1.reset()
        while not done:
            _, _, terminated, truncated, _ = env1.step(env1.action_space.sample())
            done = terminated or truncated
            i += 1
        assert i == 10000

        # test with other max_steps
        env2 = ComplexElectricityGridEnv(config, dataset, max_steps=12345)
        i = 0
        done = False
        env2.reset()
        while not done:
            _, _, terminated, truncated, _ = env2.step(env2.action_space.sample())
            done = terminated or truncated
            i += 1
        assert i == 12345
