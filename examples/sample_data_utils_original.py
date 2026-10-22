import os
import sys
import numpy as np
import pandas as pd

sys.path.append("..")

from mtsa import files_train_test_split

def get_acoustic_sample_data():
    path_input_1 = os.path.join(os.getcwd(),  "sample_data", "machine_type_1", "id_00")
    X_train, X_test, y_train, y_test = files_train_test_split(path_input_1)
    return X_train, y_train, X_test, y_test


def get_tabular_sample_data():
    data_path = os.path.join(os.getcwd()) + "/linear_ts_with_confounder_n5000_vars6_lag2_gaussian.csv"
    np.random.seed(42)
    df = pd.read_csv(data_path)
    df = df.drop(columns=["U"])
    df = df.drop(columns=["time"])

    X = df.to_numpy()
    y = np.zeros(X.shape[0])

    def set_abnormal_data(x, step, y_test, features=None, magnitude=2, start_point=None):
        anomaly = magnitude * np.sin(np.pi / 4)

        if features is None:
            features = slice(None)

        intervals = [
            (start_point or step, step),
            (2 * step, (2 * step) + step),
            (4 * step, (4 * step) + step),
            (x.shape[0] - (100 + step), x.shape[0] - 100),
        ]

        for start, end in intervals:
            x[start:end, features] += anomaly
            y_test[start:end] = np.maximum(y_test[start:end], 1)

        return x, y_test

    X, y = set_abnormal_data(X, 200, y, features=[4, 5], magnitude=2, start_point=150)
    X, y = set_abnormal_data(X, 13, y, features=[1, 0], magnitude=7, start_point=200)
    mask_normal = y == 0
    mask_abnormal = y == 1

    X_normal = X[mask_normal]
    X_abnormal = X[mask_abnormal]

    anomaly_quantity = len(X_abnormal)

    X_test = np.vstack([X_normal[:anomaly_quantity], X_abnormal])

    X_train = X_normal[anomaly_quantity:]

    y_test = np.concatenate([np.zeros(anomaly_quantity, dtype=int), np.ones(anomaly_quantity, dtype=int)])

    y_train = np.zeros_like(X_train[:, 0])

    print("Test normals :", np.sum(y_test == 0))
    print("Test anomalies:", np.sum(y_test == 1))
    print("Train normals :", np.sum(y_train == 0))
    print("Train anomalies:", np.sum(y_train == 1))

    return X_train, y_train, X_test, y_test
