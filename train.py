import argparse
from pathlib import Path
from typing import SupportsFloat
import datetime
import numpy as np
import gymnasium as gym
import tomllib
import pandas as pd

from src.basic_env import BasicElectricityGridEnv
from src.advanced_env import AdvancedElectricityGridEnv
from src.complex_env import ComplexElectricityGridEnv, Dataset
from src.util import LoggingEnvWrapper, EvalLogCallback

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm

info_keys = [
    "obs",
    "reward",
    "power_loss",
    "cost_loss",
    "power_produced",
    "power_produced_per_facility",
    "power_consumed",
    "storage_capacity",
    "storage_store_ratio",
    "production_cost",
    "carbon_cost",
    "storage_cost",
    "step",
]


def get_environment(
    name: str,
    env_config: dict,
    dataset: Dataset,
    complex_production_init_method: str = "zeros",
    use_storage=True,
) -> gym.Env:
    """Creates an environment instance with the given configurations.

    Args:
        name (str): Name of the environment class.
        env_config (dict): Config from environment.toml file.
        dataset (Dataset): Dataset for ComplexElectricityGridEnv.
        complex_production_init_method (str, optional): Production facility intialization method for ComplexElectricityGridEnv from ["zeros", "random", "contemporary"]. Defaults to "zeros".

    Returns:
        gym.Env: New gym environment.
    """
    env_class = eval(name)
    if "Complex" in name:
        return env_class(
            env_config,
            dataset,
            production_init_method=complex_production_init_method,
            use_storage=use_storage,
        )
    return env_class(env_config)


def train(args: argparse.Namespace) -> SupportsFloat:
    """Trains an agent.

    Args:
        args (argparse.Namespace): Training configurations.

    Returns:
        SupportsFloat: Mean final reward of evaluations.
    """
    with open("environment.toml", "rb") as config_file:
        env_config = tomllib.load(config_file)

    env_config["carbon_costs"] = args.carbon_costs

    dataset = Dataset(env_config["Dataset"])
    check_env(
        get_environment(args.env, env_config, dataset, use_storage=not args.no_storage)
    )

    # evaluation environment for training
    eval_log_training_env = get_environment(
        args.env,
        env_config,
        dataset,
        complex_production_init_method="contemporary",
        use_storage=not args.no_storage,
    )

    if args.logname:
        logpath = Path(f"./logs/{args.logname}")
    else:
        logpath = Path(
            f"./logs/{args.algorithm}_{args.env}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        )
    if not logpath.exists():
        logpath.mkdir(parents=True)

    for i in range(args.runs):
        # Log during training via callback
        training_logger = EvalLogCallback(
            args.eval_every_n_steps, eval_log_training_env, evaluate_training
        )

        vec_env = DummyVecEnv(
            [
                lambda: Monitor(
                    get_environment(
                        args.env,
                        env_config,
                        dataset,
                        complex_production_init_method="random",
                        use_storage=not args.no_storage,
                    )
                )
            ]
            * 4
        )

        # take seed times 10, because stable baselines actually uses [seed, seed+n_envs-1] as seeds
        model = eval(args.algorithm)(
            "MultiInputPolicy",
            vec_env,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            seed=i * 10,
        )

        model = model.learn(
            total_timesteps=args.steps, progress_bar=True, callback=training_logger
        )

        # log evaluation in logs folder
        eval_env = LoggingEnvWrapper(
            get_environment(
                args.env,
                env_config,
                dataset,
                complex_production_init_method="contemporary",
                use_storage=not args.no_storage,
            ),
            filepath=logpath / f"evaluation_{i}.json",
            info_keys=tuple(info_keys),
            args=args,
        )
        reward_mean, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
        eval_env.save()

        agent_dir = Path("./agents")
        if not agent_dir.exists():
            agent_dir.mkdir()
        model.save(agent_dir / logpath.name)

        df = training_logger.merge_logs()
        df.to_json(logpath / f"training_log_{i}.json")

    return np.mean(reward_mean)


def evaluate_training(
    env: gym.Env, model: BaseAlgorithm, n_steps: int = 10000, n_eval_episodes: int = 5
) -> pd.DataFrame:
    """Evaluate while training.
    To be used in callback

    Args:
        env: environment which is wrapped in LoggingEnvWrapper
        model: model to evaluate
        n_steps: how_many steps one environment should be evaluated at max
        n_eval_episodes: how many episodes to evaluate

    Returns:
        list of accumulated info dicts given by LoggingEnvWrapper
    """
    log_env = LoggingEnvWrapper(
        env, None, info_keys=tuple(info_keys + ["training_step"])
    )
    for episode in range(n_eval_episodes):
        state, _ = log_env.reset(seed=episode)
        for _ in range(n_steps):
            action, _ = model.predict(state, deterministic=True)
            state, _, terminated, truncated, _ = log_env.step(action)
            if terminated or truncated:
                break
    return log_env.get_dataframe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script trains an algorithm on the provided environment and saves a log file in the logs directory. environment.toml stores basic configurations."
    )
    parser.add_argument(
        "--env",
        "-e",
        default="ComplexElectricityGridEnv",
        choices=[
            "ComplexElectricityGridEnv",
            "AdvancedElectricityGridEnv",
            "BasicElectricityGridEnv",
        ],
        help="The environment on which the agent is trained.",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        default="PPO",
        choices=["PPO", "A2C"],
        help="The training algorithm that should be used for optimizing the agent.",
    )
    parser.add_argument(
        "--logname",
        help="If not specified the default log file naming is used: '[algorithm]_[env]_[timestamp]'",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Defines the number of training runs"
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=10000,
        help="Determines the frequency of the monitoring of the training process. If None no evaluation is done during training.",
    )

    parser.add_argument(
        "--carbon_costs", type=float, default=237, help="EUR per ton of CO2"
    )
    parser.add_argument(
        "--no_storage", action="store_true", help="disable storage facilities if wanted"
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=500000,
        help="Number of steps the agent takes during training.",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="The entropy coefficient for the training algorithm.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate for the training algorithm. stable_baselines3 PPO: 0.0003, stable_baselines3 A2C: 0.0007",
    )

    args = parser.parse_args()

    train(args)
