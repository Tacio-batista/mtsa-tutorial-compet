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

    # REMOVIDO: o rótulo era criado para a série inteira porque as anomalias eram injetadas nas 5000 linhas;
    # agora cada conjunto (treino e teste) tem o seu próprio rótulo, criado logo abaixo.
    # y = np.zeros(X.shape[0])

    # CORRIGIDO: a série é dividida pelo tempo ANTES de injetar anomalias (70% iniciais = treino, 30% finais = teste), sem reordenar nenhuma linha.
    n_train = int(0.7 * X.shape[0])
    X_train, X_test = X[:n_train].copy(), X[n_train:].copy()
    y_train, y_test = np.zeros(X_train.shape[0], dtype=int), np.zeros(X_test.shape[0], dtype=int)

    # REMOVIDO: a função antiga montava os intervalos a partir de "step" e o primeiro deles era
    # (start_point or step, step) em vez de (start_point, start_point + step). Com start_point=150, step=200
    # a anomalia ia de 150 a 200 (50 linhas, não 200); com start_point=200, step=13 o intervalo (200, 13)
    # ficava VAZIO (início > fim), sem nenhum aviso.
    # def set_abnormal_data(x, step, y_test, features=None, magnitude=2, start_point=None):
    #     anomaly = magnitude * np.sin(np.pi / 4)
    #
    #     if features is None:
    #         features = slice(None)
    #
    #     intervals = [
    #         (start_point or step, step),
    #         (2 * step, (2 * step) + step),
    #         (4 * step, (4 * step) + step),
    #         (x.shape[0] - (100 + step), x.shape[0] - 100),
    #     ]
    #
    #     for start, end in intervals:
    #         x[start:end, features] += anomaly
    #         y_test[start:end] = np.maximum(y_test[start:end], 1)
    #
    #     return x, y_test

    def set_abnormal_data(x, y, starts, length, features=None, magnitude=2):
        anomaly = magnitude * np.sin(np.pi / 4)

        if features is None:
            features = slice(None)

        # CORRIGIDO: todo intervalo é (início, início + duração), então nunca fica menor nem vazio.
        for start in starts:
            x[start:start + length, features] += anomaly
            y[start:start + length] = 1

        return x, y

    # REMOVIDO: as anomalias eram injetadas na série inteira (inclusive na parte que viraria treino).
    # X, y = set_abnormal_data(X, 200, y, features=[4, 5], magnitude=2, start_point=150)
    # X, y = set_abnormal_data(X, 13, y, features=[1, 0], magnitude=7, start_point=200)

    # CORRIGIDO: mesmas colunas e magnitudes do original, mas as anomalias entram só no teste, no próprio lugar no tempo.
    X_test, y_test = set_abnormal_data(X_test, y_test, starts=[150, 600, 1100], length=50, features=[4, 5], magnitude=2)
    X_test, y_test = set_abnormal_data(X_test, y_test, starts=[400, 900, 1350], length=13, features=[1, 0], magnitude=7)

    # REMOVIDO: este bloco era o erro principal. X[mask] tira as linhas anômalas e EMENDA os pedaços normais
    # (instantes que nunca foram vizinhos passam a ser vizinhos), e o vstack monta o teste com as primeiras
    # linhas normais seguidas de TODAS as anomalias empilhadas no fim (rótulos 0...0 1...1). Como o modelo lê
    # janelas de 12 linhas seguidas, as janelas nas emendas misturam momentos sem relação e a ordem temporal
    # (dependência de lag 2 da série) é destruída.
    # mask_normal = y == 0
    # mask_abnormal = y == 1
    #
    # X_normal = X[mask_normal]
    # X_abnormal = X[mask_abnormal]
    #
    # anomaly_quantity = len(X_abnormal)
    #
    # X_test = np.vstack([X_normal[:anomaly_quantity], X_abnormal])
    #
    # X_train = X_normal[anomaly_quantity:]
    #
    # y_test = np.concatenate([np.zeros(anomaly_quantity, dtype=int), np.ones(anomaly_quantity, dtype=int)])
    #
    # y_train = np.zeros_like(X_train[:, 0])

    print("Test normals :", np.sum(y_test == 0))
    print("Test anomalies:", np.sum(y_test == 1))
    print("Train normals :", np.sum(y_train == 0))
    print("Train anomalies:", np.sum(y_train == 1))

    return X_train, y_train, X_test, y_test
