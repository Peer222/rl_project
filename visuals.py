from enum import Enum
from typing import Optional
from pathlib import Path
from argparse import ArgumentParser
import tomllib  # type: ignore
import json
from math import floor

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns  # type: ignore
import pandas as pd
import numpy as np

from src.complex_env import Dataset


class Color(Enum):
    BLACK = (0, 0, 0)
    ORANGE = (217 / 256, 130 / 256, 30 / 256)
    YELLOW = (227 / 256, 193 / 256, 0)
    BLUE = (55 / 256, 88 / 256, 136 / 256)
    GREY = (180 / 256, 180 / 256, 180 / 256)
    RED = (161 / 256, 34 / 256, 0)
    GREEN = (0, 124 / 256, 6 / 256)
    LIGHT_GREY = (240 / 256, 240 / 256, 240 / 256)


COLORS: list = [color.value for color in Color]


class Time(Enum):
    YEAR = {"name": "years", "scaling": 365}
    DAY = {"name": "days", "scaling": 1}

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def fromstring(s: str) -> "Time":
        try:
            return Time[s]
        except:
            raise ValueError()


class Unit(Enum):
    UNITLESS = {"name": "unitless", "abbreviation": "", "description": ""}
    MEGAWATT = {"name": "megawatt", "abbreviation": "MW", "description": "(in MW)"}
    MEGAWATTHOURS = {
        "name": "megawatt hours",
        "abbreviation": "MWh",
        "description": "(in MWh)",
    }
    EURO = {"name": "euro", "abbreviation": "€", "description": "(in EUR)"}
    METERPERSECOND = {
        "name": "meter per second",
        "abbreviation": "m/s",
        "description": "(m/s)",
    }


def plot(plt, file: Optional[Path] = None) -> None:
    """Saves or shows given plot.

    Args:
        plt (_type_): Plot
        file (Path, optional): Path of saved plot. If None the plot is only shown. Defaults to None.
    """
    # plt.tight_layout()
    if not file:
        plt.show()
    else:
        if not file.parent.is_dir():
            file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=file, dpi=300, bbox_inches="tight")
        plt.close()


def add_x_ticks(ax: Axes, steps: int, timeformat: Time, num_ticks: int = 10) -> None:
    """Adds x ticks to the given axes.

    Args:
        ax (Axes): Axes that should contain the x ticks.
        steps (int): Number of data entries in the plotted time series.
        timeformat (Time): Time specification (HOUR, DAY, YEAR).
        num_ticks (int, optional): Number of ticks. Defaults to 10.
    """
    positions = np.arange(0, num_ticks, dtype=float) / (num_ticks - 1)
    ax.set_xticks(positions * steps)
    ax.set_xticklabels((positions * steps // timeformat.value["scaling"]).astype(int))  # type: ignore


def show_multiline_plot(
    data_series: pd.DataFrame,
    colors: list[Color] | None,
    x_label: str = "",
    y_labels: list[str] = [],
    unit: Unit = Unit.UNITLESS,
    filename: Optional[Path] = None,
    timeformat: Time = Time.DAY,
    hue: str | None = None,
    errorbar: tuple[str, int] | None = ("ci", 95)
) -> None:
    """Plots multiple line plots into single axes.

    Args:
        data_series (pd.DataFrame): errorbars are plotted if multiple values exist per x.
        colors (list[Color]): List of colors for each line plot.
        x_label (str): If data_series is of type pd.DataFrame, x_label determines x column.
        y_labels (list[str]): If data_series is of type pd.Dataframe, labels provide the used column names.
        unit (Unit): Unit of the shown data.
        filename (Path, optional): Name of the file in that the plot is saved. If None plot is shown instead of saved. Defaults to None.
        timeformat (Time, optional): Determines at what scale the xticks are presented (HOUR, DAY, YEAR). Defaults to Time.DAY.
    """
    assert len(y_labels) > 0

    _, ax = plt.subplots(figsize=(15, 7))

    data_series[x_label] = data_series[x_label] - data_series[x_label].min()
    if hue:
        sns.lineplot(
            data=data_series,
            x=x_label,
            y=y_labels[0],
            hue=hue,
            ax=ax,
            errorbar=errorbar,
            dashes=False,
        )
    else:
        sns.lineplot(data=data_series.set_index('Day')[y_labels], c=None, ax=ax, errorbar=errorbar, dashes=False)

    plt.xlabel(f"Time (in {timeformat.value['name']})")
    plt.ylabel(f"Amount {unit.value['description']}")

    ax.grid(True, color=Color.LIGHT_GREY.value)
    sns.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)
    return


def show_line_plot(
    data_series: pd.Series | pd.DataFrame,
    y_label: str = "",
    x_label: str = "",
    color: Color = Color.BLUE,
    unit: Unit = Unit.UNITLESS,
    filename: Optional[Path] = None,
    timeformat: Time = Time.DAY,
) -> None:
    """Plots a single data series with a blue line plot.

    Args:
        data_series (pd.Series | pd.DataFrame): Data that should be plotted.
        y_label (str | None): Label of data column in DataFrame. Not needed for pd.Series.
        x_label (str | None): Label of column in DataFrame that is used for x axis. Not needed for pd.Series.
        unit (Unit, optional): Unit of the shown data.
        filename (Path, optional): Name of the file in that the plot is saved. If None plot is shown instead of saved. Defaults to None.
        timeformat (Time, optional): Determines at what scale the xticks are presented (HOUR, DAY, YEAR). Defaults to Time.DAY.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    if isinstance(data_series, pd.Series):
        steps = np.arange(0, len(data_series), 1)
        ax.plot(steps, list(data_series), c=color.value, label=data_series.name.split(".")[-1].replace("_", " "))  # type: ignore
    elif isinstance(data_series, pd.DataFrame):
        steps = np.arange(0, len(data_series[x_label].unique()), 1)
        data_series[x_label] = data_series[x_label] - data_series[x_label].min()
        sns.lineplot(
            data_series,
            x=x_label,
            y=y_label,
            errorbar=("ci", 95),
            c=color.value,
            label=y_label.split(".")[-1].replace("_", " "),
        )
    else:
        raise TypeError(f"{type(data_series)} is not supported.")

    plt.xlabel(f"Time (in {timeformat.value['name']})")
    plt.ylabel(f"Amount {unit.value['description']}")
    plt.legend()

    ax.grid(True, color=Color.LIGHT_GREY.value)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    ax.set_xlim(xmin=0)

    add_x_ticks(ax, steps[-1], timeformat)

    sns.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)
    return


def show_production(
    production_df: pd.DataFrame,
    x_label: str,
    consumption: Optional[pd.DataFrame] = None,
    unit: Unit = Unit.MEGAWATTHOURS,
    filename: Optional[Path] = None,
    title: str = "",
    timeformat: Time = Time.DAY,
) -> None:
    """Plots the amount of produced power seperated by the used technology.

    Args:
        production_df (pd.DataFrame): Amount of produced energy that should be plotted.
        x_label (str): Label of index column.
        consumption (pd.Series): Amount of consumed energy that should be plotted. (Not Implemented)
        unit (Unit): Unit of the shown data.
        filename (Path, optional): Name of the file in that the plot is saved. If None plot is shown instead of saved. Defaults to None.
        timeformat (Time, optional): Determines at what scale the xticks are presented (HOUR, DAY, YEAR). Defaults to Time.DAY.

    Raises:
        NotImplementedError: Raised when consumption parameter is given.
    """
    steps = np.arange(0, len(production_df[x_label].unique()), 1)

    ax: Axes
    _, ax = plt.subplots(figsize=(12, 5))

    production_df = production_df.reset_index(drop=True)
    production_df = production_df.rename(lambda x: x.split(".")[-1].replace("_", " "), axis=1)
    production_df_mean = production_df.groupby("Day").mean()
    labels = [col for col in production_df.columns if col != x_label]
    steps = np.arange(0, len(production_df[x_label].unique()), 1)
    ax.stackplot(
        steps,
        *[production_df_mean[label] for label in labels],
        labels=labels,
        colors=COLORS,
    )
    # production_df.plot.area(x=x_label, ax=ax, alpha=1.0, color=COLORS)
    if consumption is not None:
        raise NotImplementedError  # does not work (line plot not visible)
        consumption.plot(ax=ax, c=Color.RED, label="consumed power", zorder=50)

    if title:
        plt.title(title)
    plt.xlabel(f"Time (in {timeformat.value['name']})")
    plt.ylabel(f"Amount {unit.value['description']}")
    plt.legend(loc="upper left")

    ax.grid(True, color=Color.LIGHT_GREY.value)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)

    add_x_ticks(ax, steps[-1], timeformat)

    sns.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)
    return


def show_storage(
    data_series: pd.Series,
    capacity: float,
    unit: Unit = Unit.UNITLESS,
    filename: Optional[Path] = None,
    timeformat: Time = Time.DAY,
) -> None:
    """Plots the storage utilitzation with upper and lower bounds

    Args:
        data_series (pd.Series): Data that should be plotted.
        capacity(float): Total capacity of the storage facilities of the environment
        unit (Unit, optional): Unit of the shown data.
        filename (Path, optional): Name of the file in that the plot is saved. If None plot is shown instead of saved. Defaults to None.
        timeformat (Time, optional): Determines at what scale the xticks are presented (HOUR, DAY, YEAR). Defaults to Time.DAY.
    """
    steps = np.arange(0, len(data_series), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, list(data_series), c=Color.BLUE.value, label=data_series.name.split(".")[-1])  # type: ignore

    ax.hlines(y=0, xmin=0, xmax=len(steps), linewidth=3, color=Color.LIGHT_GREY.value)
    ax.hlines(
        y=capacity,
        xmin=0,
        xmax=len(steps),
        linewidth=3,
        color=Color.GREY.value,
        label="capacity",
    )

    plt.xlabel(f"Time (in {timeformat.value['name']})")
    plt.ylabel(f"Amount {unit.value['description']}")
    plt.legend()

    ax.grid(True, color=Color.LIGHT_GREY.value)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=-0.05 * capacity, ymax=capacity + 0.05 * capacity)

    add_x_ticks(ax, steps[-1], timeformat)

    sns.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)
    return


def drop_warmup(log_df: pd.DataFrame, proportion: float = 0.01) -> pd.DataFrame:
    """Drops the first rows of each episode depending on the proportion or the maximum length of 1 year.

    Args:
        log_df (pd.DataFrame): The log dataframe that should be cut.
        proportion (float, optional): Proportion of the dataframes episode length that should be cut off at the beginning. Defaults to 0.01.

    Returns:
        pd.DataFrame: Pruned dataframe.
    """
    start_day = floor(log_df["day"].max() * proportion)
    return log_df[log_df["day"] >= start_day]


def select_days(log_df: pd.DataFrame, num_days: int = 365) -> pd.DataFrame:
    """Selects the number of days that should be displayed in the dataframe.

    Args:
        log_df (pd.DataFrame):  The log dataframe that should be cut.
        num_days (int, optional): Number of days in the resulting dataframe. Defaults to 365.

    Returns:
        pd.DataFrame: Cutted dataframe.
    """
    return log_df[log_df["day"] <= num_days]


def concat_dataframes(df_paths: list[Path]) -> pd.DataFrame:
    """Loads dataframes from filepaths and concatenates them.
    'agent' column is added with index of training run.

    Args:
        df_paths (list[Path]): paths to stored log dataframes.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    assert len(df_paths) > 0
    logs: list[pd.DataFrame] = []
    for i, path in enumerate(df_paths):
        df = pd.read_json(path)
        df["agent"] = i
        logs.append(df)

    return pd.concat(logs).reset_index()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create plots from logfiles specified by option --log_dir"
    )
    parser.add_argument(
        "--log_dir",
        "-d",
        type=str,
        required=True,
        help="Path of the log directory (logs/log_dir)",
    )
    parser.add_argument(
        "--timeformat",
        type=Time.fromstring,
        choices=list(Time),
        default=Time.DAY,
        help="time format of the x axis for time dependent plots",
    )
    parser.add_argument(
        "--env_num",
        type=int,
        default=0,
        help="The number of the environment/ evaluation run to plot (Has no effect when accumulate flag is set)",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="If set the mean over all environments/ evaluation runs is used for plotting (Not implemented)",
    )
    parser.add_argument(
        "--drop_warmup",
        action="store_true",
        help="If set the first percent of the data entries of the episode are dropped",
    )
    parser.add_argument(
        "--num_days",
        type=int,
        default=365,
        help="The number of days that should be plotted. 'Starts' from the end.",
    )
    parser.add_argument(
        "--plot_weather",
        action="store_true",
        help="If set a sample year of the weather data is plotted.",
    )
    args = parser.parse_args()

    # sns.set_style("whitegrid")
    sns.set_context("poster")

    logdir_path = Path(args.log_dir)
    eval_logpaths = list(logdir_path.glob(r"evaluation_*.json"))
    train_logpaths = list(logdir_path.glob(r"training_*.json"))
    with open(logdir_path / "config.json", "r") as config_file:
        config = json.load(config_file)

    columns_to_aggregate = {
        "obs.power_produced": "sum",
        "obs.power_consumed": "sum",
        "obs.balance": "sum",
        "balance_min": "min",
        "reward": "sum",
        "power_loss": "sum",
        "cost_loss": "sum",
        "power_produced": "sum",
        "power_produced_per_facility.coal": "sum",
        "power_produced_per_facility.nuclear": "sum",
        "power_produced_per_facility.solar": "sum",
        "power_produced_per_facility.wind": "sum",
        "power_produced_per_facility.gas": "sum",
        "power_produced_per_facility.storage": "sum",
        "power_consumed": "sum",
        "storage_capacity": "mean",
        "storage_store_ratio": "mean",
        "carbon_cost": "sum",
        "production_cost": "sum",
        "storage_cost": "sum",
    }
    column_renamer = {
        "obs.balance": "Power Balance",
        "balance": "Power Balance",
        "balance_min": "Min Power Balance over day",
        "reward": "Reward",
        "power_loss": "Power Loss",
        "cost_loss": "Cost Loss",
        "power_produced": "Power produced",
        "power_produced_per_facility.coal": "Coal",
        "power_produced_per_facility.nuclear": "Nuclear",
        "power_produced_per_facility.solar": "Solar",
        "power_produced_per_facility.wind": "Wind",
        "power_produced_per_facility.gas": "Gas",
        "power_produced_per_facility.storage": "Storage",
        "power_consumed": "Power consumed",
        "storage_capacity": "Capacity",
        "storage_store_ratio": "Store Ratio",
        "carbon_cost": "Carbon Cost",
        "production_cost": "Production Cost",
        "storage_cost": "Storage Cost",
        "storage_stored": "Stored Electricity",
        "day": "Day",
        "training_step": "Training Step"
    }
    # concatenate all logs and bin per 24 hours
    evaluation_log = concat_dataframes(eval_logpaths).drop(
        ["obs.production_facilities", "obs.storage_facilities", "index"],
        axis=1,
        errors="ignore",
    )
    evaluation_log["day"] = evaluation_log["step"] // 24
    evaluation_log["balance_min"] = evaluation_log["obs.balance"]  # will be aggregated to min in next step
    evaluation_log = (
        evaluation_log.drop("step", axis=1)
        .groupby(["agent", "env_num", "day"])
        .agg(columns_to_aggregate)
        .reset_index()
    )
    evaluation_log["storage_stored"] = (
        evaluation_log["storage_store_ratio"] * evaluation_log["storage_capacity"]
    )
    evaluation_log["storage_stored"] = evaluation_log["storage_store_ratio"] * evaluation_log["storage_capacity"]

    training_log = concat_dataframes(train_logpaths).drop(
        ["obs.production_facilities", "obs.storage_facilities", "index"],
        axis=1,
        errors="ignore",
    )
    training_log["day"] = training_log["step"] // 24
    training_log["balance_min"] = training_log["obs.balance"]  # will be aggregated to min in next step
    training_log = (
        training_log.drop("step", axis=1)
        .groupby(["training_step", "agent", "env_num", "day"])
        .agg(columns_to_aggregate)
        .reset_index()
    )
    training_log["storage_stored"] = training_log["storage_store_ratio"] * training_log["storage_capacity"]

    evaluation_log["power_produced_per_facility.storage"] = evaluation_log[
        "power_produced_per_facility.storage"
    ].apply(lambda x: max(0, x))

    production_facilities = [
        f"power_produced_per_facility.{facility}"
        for facility in config["Production"]["power"].keys()
    ]

    if args.drop_warmup:
        evaluation_log = drop_warmup(evaluation_log)

    evaluation_log = select_days(evaluation_log, args.num_days)
    training_log = select_days(training_log, args.num_days)

    result_dir = Path("visuals") / logdir_path.name
    if not result_dir.exists():
        result_dir.mkdir(parents=True)

    import re
    print("ratios")
    print(evaluation_log.filter(regex=r"^power_produced_per").sum() / evaluation_log.filter(regex=r"^power_produced_per").sum().sum())
    show_production(
        evaluation_log[production_facilities + ["day"]].rename(mapper=column_renamer, axis=1),
        column_renamer["day"],
        None,
        Unit.MEGAWATTHOURS,
        result_dir / "production_avg.png",
        timeformat=args.timeformat,
    )
    show_production(
        evaluation_log[production_facilities + ["power_produced_per_facility.storage", "day"]].rename(mapper=column_renamer, axis=1),
        column_renamer["day"],
        None,
        Unit.MEGAWATTHOURS,
        result_dir / "production_storage_avg.png",
        timeformat=args.timeformat,
    )
    show_multiline_plot(
        evaluation_log.rename(mapper=column_renamer, axis=1),
        [Color.BLACK, Color.RED],
        x_label=column_renamer["day"],
        y_labels=[column_renamer["power_produced"], column_renamer["power_consumed"]],
        unit=Unit.MEGAWATTHOURS,
        filename=result_dir / "production_vs_consumption_avg.png",
        timeformat=args.timeformat,
    )
    show_multiline_plot(
        evaluation_log.rename(mapper=column_renamer, axis=1),
        [Color.BLUE, Color.RED],
        x_label=column_renamer["day"],
        y_labels=[column_renamer["cost_loss"], column_renamer["power_loss"]],
        filename=result_dir / "cost_and_power_loss_avg.png",
        timeformat=args.timeformat,
    )
    show_multiline_plot(
        evaluation_log.rename(mapper=column_renamer, axis=1),
        [Color.BLUE, Color.GREY],
        x_label=column_renamer["day"],
        y_labels=[column_renamer["storage_stored"], column_renamer["storage_capacity"]],
        unit=Unit.MEGAWATTHOURS,
        filename=result_dir / "storage_avg.png",
        timeformat=args.timeformat,
    )
    show_line_plot(
        evaluation_log.rename(mapper=column_renamer, axis=1),
        y_label=column_renamer["balance"],
        x_label=column_renamer["day"],
        unit=Unit.MEGAWATTHOURS,
        filename=result_dir / "balance_avg.png",
        timeformat=args.timeformat,
    )
    show_line_plot(
        evaluation_log.rename(mapper=column_renamer, axis=1),
        y_label=column_renamer["reward"],
        x_label=column_renamer["day"],
        filename=result_dir / "reward_avg.png",
        timeformat=args.timeformat,
    )
    show_multiline_plot(
        evaluation_log.rename(mapper=column_renamer, axis=1),
        None,
        x_label=column_renamer["day"],
        unit=Unit.EURO,
        y_labels=list(map(lambda x: column_renamer[x], ["production_cost", "storage_cost", "carbon_cost"])),
        filename=result_dir / "costs_avg.png",
        timeformat=args.timeformat,
    )

    # training evaluation
    # filter out training_log to only keep a few steps
    training_steps_wanted = [0, 10000, 20000, 30000, 50000, 100000, 200000]
    training_steps_next_equal = []  # find actual training steps to plot, above values are not the ones actually recorded
    for step_wanted in training_steps_wanted:
        training_steps_next_equal.append(training_log['training_step'].iloc[(training_log['training_step']-step_wanted).abs().argsort()[0]])
    training_log = training_log[training_log['training_step'].isin(training_steps_next_equal)]


    # plot balance line for different training-steps
    show_multiline_plot(
        training_log.rename(mapper=column_renamer, axis=1),
        None,
        x_label=column_renamer["day"],
        y_labels=[column_renamer['balance_min']],
        unit=Unit.MEGAWATTHOURS,
        hue=column_renamer["training_step"],
        filename=result_dir / "training_balance.png",
        errorbar=None
    )

    # plot mean costs over episode for different training-steps
    training_log['Cost'] = training_log['production_cost'] + training_log['storage_cost'] + training_log['carbon_cost']
    show_multiline_plot(
        training_log.rename(mapper=column_renamer, axis=1),
        None,
        x_label=column_renamer["day"],
        y_labels=['Cost'],
        unit=Unit.EURO,
        hue=column_renamer['training_step'],
        filename=result_dir / "training_cost.png",
        errorbar=None,
    )

    training_log['Loss'] = training_log['power_loss'] + training_log['cost_loss']
    # plot losses over episode for different training-steps
    show_multiline_plot(
        training_log.rename(mapper=column_renamer, axis=1),
        None,
        x_label=column_renamer["day"],
        y_labels=['Loss'],
        hue=column_renamer['training_step'],
        filename=result_dir / "training_loss.png",
        errorbar=None,
    )

    # plots weather (at 12) (does not match period of evaluations)
    if args.plot_weather:
        dataset = Dataset(config["Dataset"], offset=0)

        show_line_plot(
            dataset.dataset["wind_speed [m/s]"].iloc[np.arange(12, 365 * 24 + 13, 24)],
            unit=Unit.METERPERSECOND,
            filename=result_dir / "wind.png",
            timeformat=Time.DAY,
        )
        show_line_plot(
            dataset.dataset["solar_irradiance"].iloc[np.arange(12, 365 * 24 + 13, 24)],
            color=Color.YELLOW,
            unit=Unit.UNITLESS,
            filename=result_dir / "solar.png",
            timeformat=Time.DAY,
        )
