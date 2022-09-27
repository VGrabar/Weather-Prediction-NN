from typing import List

import numpy as np
import pandas as pd
import pathlib
import torch

import tqdm


def create_celled_data(
    data_path,
    dataset_name,
    left_border: int = 0,
    down_border: int = 0,
    right_border: int = 2000,
    up_border: int = 2500,
    time_col: str = "time",
    event_col: str = "val",
    x_col: str = "x",
    y_col: str = "y",
):

    data_path = pathlib.Path(
        data_path,
        dataset_name,
    )

    df = pd.read_csv(data_path)
    df.sort_values(by=[time_col], inplace=True)
    df = df[[event_col, x_col, y_col, time_col]]

    indicies = range(df.shape[0])
    start_date = int(df[time_col][indicies[0]])
    finish_date = int(df[time_col][indicies[-1]])
    n_cells_hor = df[x_col].max() - df[x_col].min() + 1
    n_cells_ver = df[y_col].max() - df[y_col].min() + 1
    celled_data = torch.zeros(
        [finish_date - start_date + 1, 1, n_cells_hor, n_cells_ver]
    )

    for i in tqdm.tqdm(indicies):
        if (
            (df[x_col][i] > left_border)
            and (df[x_col][i] < right_border)
            and (df[y_col][i] > down_border)
            and (df[y_col][i] < up_border)
        ):

            x = int(df[x_col][i])
            y = int(df[y_col][i])
            celled_data[int(df[time_col][i]) - start_date, 0, x, y] = df[event_col][i]


    return celled_data
