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

    n_train = int(0.7 * X.shape[0])
    X_train, X_test = X[:n_train].copy(), X[n_train:].copy()
    y_train, y_test = np.zeros(X_train.shape[0], dtype=int), np.zeros(X_test.shape[0], dtype=int)

  
    def set_abnormal_data(x, y, starts, length, features=None, magnitude=2):
        anomaly = magnitude * np.sin(np.pi / 4)

        if features is None:
            features = slice(None)

        # CORRIGIDO: todo intervalo é (início, início + duração), então nunca fica menor nem vazio.
        for start in starts:
            x[start:start + length, features] += anomaly
            y[start:start + length] = 1

        return x, y


    X_test, y_test = set_abnormal_data(X_test, y_test, starts=[150, 600, 1100], length=50, features=[4, 5], magnitude=2)
    X_test, y_test = set_abnormal_data(X_test, y_test, starts=[400, 900, 1350], length=13, features=[1, 0], magnitude=7)

    print("Test normals :", np.sum(y_test == 0))
    print("Test anomalies:", np.sum(y_test == 1))
    print("Train normals :", np.sum(y_train == 0))
    print("Train anomalies:", np.sum(y_train == 1))

    return X_train, y_train, X_test, y_test
