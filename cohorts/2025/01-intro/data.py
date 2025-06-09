import pandas as pd
import os
from tqdm import tqdm
import requests
import threading
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer


def get_data(month: int, year: int, color: str) -> pd.DataFrame:
    """
    Downloads a Parquet trip data file for a specified month, year, and taxi color from a remote server,
    saving it locally in the 'data' directory with a progress bar.

    Args:
        month (int): The month of the trip data to download (1-12).
        year (int): The year of the trip data to download (e.g., 2023).
        color (str): The color of the taxi (e.g., 'yellow', 'green').

    Returns:
        str: The local file path to the downloaded Parquet file.
    """

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"
    os.makedirs("data", exist_ok=True)
    local_filename = f"data/{color}_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    # Use a larger chunk size for faster download
    chunk_size = 1024 * 1024  # 1MB

    with open(local_filename, "wb") as file, tqdm(
        desc=local_filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    return local_filename


def optimize_dtypes(df):
    """
    Otimiza os tipos de dados das colunas do DataFrame para formatos menores quando possível.

    Args:
        df (pd.DataFrame): DataFrame a ser otimizado.

    Returns:
        pd.DataFrame: DataFrame com tipos de dados otimizados.
    """
    for col in df.columns:
        col_data = df[col]
        if pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="integer")
        elif pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="float")
        elif pd.api.types.is_object_dtype(col_data):
            num_unique_values = col_data.nunique()
            num_total_values = len(col_data)
            if num_unique_values / num_total_values < 0.5:
                df[col] = col_data.astype("category")
    return df


def read_data(file_path: str, file_name: str) -> pd.DataFrame:
    """
    Reads a Parquet file from the specified directory and returns its contents as a pandas DataFrame.

    Parameters:
        file_path (str): The directory path where the file is located.
        file_name (str): The name of the file to read (with or without .parquet extension).

    Returns:
        pandas.DataFrame: The contents of the Parquet file.

    Raises:
        ValueError: If the file format is not supported (i.e., not a .parquet file).
        FileNotFoundError: If the specified file does not exist.
    """
    if not file_name.lower().endswith(".parquet"):
        file_name += ".parquet"
    full_path = os.path.join(file_path, file_name)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    if file_name.lower().endswith(".parquet"):
        return pd.read_parquet(full_path)
    else:
        raise ValueError("Unsupported file format: {}".format(file_name))


def filter_vect_data(df: pd.DataFrame):
    """
    Filters and vectorizes the input DataFrame for model training.

    Args:
        df (pd.DataFrame): Input DataFrame with taxi trip data.
        output_format (str): Output format for the vectorized data.
                             Options: "array" (default), "dataframe".

    Returns:
        X_train: Vectorized feature matrix in the specified format.
    """

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype("str")

    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    train_data = dv.fit_transform(train_dicts)

    return train_data


def training_data(X, y):
    """
    Trains a Linear Regression model on the provided training data and returns predictions.

    Args:
        X (array-like or pd.DataFrame): Feature matrix for training.
        y (array-like or pd.Series): Target values for training.

    Returns:
        tuple:
            - y_pred (np.ndarray): Predicted target values for the training data.
            - None: The function prints the RMSE to stdout and returns None as the second element.

    Side Effects:
        Prints the Root Mean Squared Error (RMSE) of the model on the training data.
    """
    lr = LinearRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    return y_hat, print(f"Train RMSE: {root_mean_squared_error(y, y_hat)}")
