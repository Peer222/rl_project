from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
import gymnasium as gym
import argparse
import json
import gymnasium as gym
from collections import defaultdict
from typing import Any, SupportsFloat, Callable
import pandas as pd
from pathlib import Path


def huber_loss(x: SupportsFloat, delta: float = 1.0) -> float:
    """Computes the smoothed Huber loss
    https://en.wikipedia.org/wiki/Huber_loss

    Args:
        x (SupportsFloat): Input
        delta (float, optional): Threshold for smoothing/flatten MSE loss. Defaults to 1.0.

    Returns:
        float: Huber loss value
    """
    x = float(x)
    return (1 / 2) * x**2 if abs(x) <= delta else delta * (abs(x) - (1 / 2) * delta)


def dataframe_from_accinfos(acc_infos: list[dict], info_keys: tuple) -> pd.DataFrame:
    matching_keys = [
        key for key in acc_infos[0].keys() if key.split(".")[0] in info_keys
    ]
    dfs = [
        pd.DataFrame({key: acc_info[key] for key in matching_keys})
        for acc_info in acc_infos
    ]
    for i, df in enumerate(dfs):
        df["env_num"] = i
    return pd.concat(dfs, ignore_index=True)


class LoggingEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        filepath: Path | None,
        info_keys: tuple = (),
        args: argparse.Namespace | None = None,
    ):
        """

        Args:
            env (gym.Env): Gymnasium environment that should be wrapped.
            filepath (str): filepath of the saved log file.
            info_keys (tuple, optional): Keys that should be saved from the information dict returned by the environment step method. Defaults to ().
            args (argparse.Namespace | None, optional): Script arguments/hyperparameters that are stored with the log. Defaults to None.
        """
        self.env = env
        self.info_keys = info_keys
        self.filepath = filepath
        self.args = args

        self.accumulated_info: defaultdict = None  # type: ignore
        self.accumulated_infos: list = []

        self._action_space: gym.spaces.Space | None = None
        self._observation_space: gym.spaces.Space | None = None
        self._reward_range: tuple | None = None
        self._metadata: dict[str, Any] | None = None

        self._cached_spec = None

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[str, Any]]:
        """Resets the wrapped environment and clears the log without saving.

        Args:
            seed (int | None, optional): Seed for resetting the wrapped environment. Defaults to None.
            options (dict | None, optional): Unused. Defaults to None.

        Returns:
            tuple[dict, dict[str, Any]]: _description_
        """
        if self.accumulated_info:
            self.accumulated_infos.append(self.accumulated_info)
        self.accumulated_info = defaultdict(lambda: [], {})
        return self.env.reset(seed=seed)

    def step(self, action):
        """Performs a step in the wrapped environment and stores the returned information from the info_dict.

        Args:
            action (Any): Action that should be performed in the environment.

        Returns:
            Any: env.step() return values.
        """
        observation, reward, terminated, truncated, newinfo = self.env.step(action)
        for key, value in newinfo.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    try:  # isinstance Sequence is not working, as numpy array is not a member
                        if len(subval) == 1:
                            subval = subval[0]
                    except:
                        pass
                    self.accumulated_info[f"{key}.{subkey}"].append(subval)
            else:
                self.accumulated_info[key].append(value)
        info = self.accumulated_info if terminated or truncated else newinfo
        return observation, reward, terminated, truncated, info

    def get_dataframe(self) -> pd.DataFrame:
        return dataframe_from_accinfos(
            self.accumulated_infos + [self.accumulated_info], self.info_keys
        )

    def save(self):
        if not self.filepath:
            return
        df = dataframe_from_accinfos(self.accumulated_infos, self.info_keys)

        df.to_json(self.filepath)
        with open(self.filepath.parent / "config.json", "w") as f:
            json.dump(self.env.config, f)
        with open(self.filepath.parent / "args.json", "w") as f:
            json.dump(vars(self.args), f)

    def __del__(self):
        """Saves the log, arguments and the environment config in a json file on deletion of the wrapper.
        {
            "config": Config_dict,
            "args": Args_dict[optional],
            "log": Log in pd.to_json() default format
        }
        """
        self.save()


class EvalLogCallback(BaseCallback):
    """Log information during training
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        eval_every_n_steps: int,
        evaluation_env: gym.Env,
        evaluation_function: Callable[[gym.Env, BaseAlgorithm], pd.DataFrame],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.eval_every_n_steps = eval_every_n_steps
        self.last_eval_step = 0
        self.evaluation_env = evaluation_env
        self.evaluation_function = evaluation_function
        self.info: dict = {}

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.info[self.num_timesteps] = self.evaluation_function(
            self.evaluation_env, self.model
        )
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.num_timesteps - self.last_eval_step >= self.eval_every_n_steps:
            self.last_eval_step = self.num_timesteps
            self.info[self.num_timesteps] = self.evaluation_function(
                self.evaluation_env, self.model
            )
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.info[self.num_timesteps] = self.evaluation_function(
            self.evaluation_env, self.model
        )
        return

    def merge_logs(self) -> pd.DataFrame:
        """Merges evaluation logs from different timesteps of the training.

        Returns:
            pd.DataFrame: Merged log.
        """
        dfs_to_concat = []
        for steps, df in self.info.items():
            df_to_concat = df
            df_to_concat["training_step"] = steps
            dfs_to_concat.append(df_to_concat)
        merged_df = pd.concat(dfs_to_concat)
        return merged_df.reset_index()
