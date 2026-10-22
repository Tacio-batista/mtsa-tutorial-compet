"""Ferramentas do Desafio 1: diagnóstico do erro e comparação dos cenários A, B, C e D (GANF e Hitachi).

- sample_data_utils_original.py -> código ORIGINAL da aula, intacto
- sample_data_utils.py          -> código CORRIGIDO
Este arquivo só USA as duas versões; não altera nenhuma delas.
"""
import contextlib
import io
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from examples import sample_data_utils as corrigida
from examples import sample_data_utils_original as original
from mtsa.losses.forward_loss import ForwardLoss
from mtsa.losses.lagrange_loss import LagrangeLoss
from mtsa.losses.mse_loss import MSELoss
from mtsa.losses.trace_exponential_matrix_loss import TraceExponentialMatrixLoss
from mtsa.models.hitachi import ReconstructionError, WindowedHitachi
from mtsa.models.hybrid.ganf import GANF
from mtsa.predictors.standard_predictor import StandardPredictor
from mtsa.trainers.lagrange_trainer import LagrangeTrainer
from mtsa.trainers.unsupervised_trainer import UnsupervisedTrainer
from mtsa.utils.data import get_dataloader
from mtsa.utils.plots import plot_roc_pr_curves

CSV = "linear_ts_with_confounder_n5000_vars6_lag2_gaussian.csv"
WINDOW = 12
VERSOES = {"original": original.get_tabular_sample_data, "corrigida": corrigida.get_tabular_sample_data}


def carregar(versao):
    """Chama a função de dados da versão escolhida ('original' ou 'corrigida') sem os prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return VERSOES[versao]()


# ============================================================================
# 1) Por que a versão original está errada: rastrear o instante de cada linha
# ============================================================================

def _rastrear_tempo(funcao):
    """Roda a função de dados REAL trocando a coluna X3 pelo instante original (coluna time).

    X3 nunca recebe anomalia em nenhuma das versões, então depois da divisão ela diz exatamente
    de que instante veio cada linha. Serve só para o diagnóstico; o modelo nunca vê isso.
    """
    leitura_real = pd.read_csv

    def leitura_rastreada(*args, **kwargs):
        df = leitura_real(*args, **kwargs)
        df["X3"] = df["time"].astype(float)
        return df

    with mock.patch.object(pd, "read_csv", leitura_rastreada), contextlib.redirect_stdout(io.StringIO()):
        X_train, y_train, X_test, y_test = funcao()

    t_train = np.rint(X_train[:, 2]).astype(int)
    t_test = np.rint(X_test[:, 2]).astype(int)
    return t_train, t_test, np.asarray(y_test).astype(int)


def diagnostico_ordem_temporal(janela=WINDOW):
    """Mostra, para as duas versões, onde a ordem do tempo quebra e o que o modelo recebe numa janela."""
    rastreio = {}
    for versao in ["original", "corrigida"]:
        t_train, t_test, y_test = _rastrear_tempo(VERSOES[versao])
        rastreio[versao] = (t_train, t_test, y_test)

        print(f"\n================ versão {versao.upper()} ================")
        for nome, t in [("TREINO", t_train), ("TESTE", t_test)]:
            emendas = np.where(np.diff(t) != 1)[0]
            print(f"{nome}: {len(t)} linhas, {len(emendas)} emenda(s) no tempo")
            for i in emendas:
                print(f"    posição {i} -> {i + 1}: instante {t[i]} -> {t[i + 1]}")

        volta = np.where(np.diff(t_test) < 0)[0]
        if len(volta):
            inicio = max(volta[0] - janela // 2 + 1, 0)
            descricao = "que atravessa a emenda onde o tempo VOLTA"
        else:
            inicio = max(int(np.argmax(y_test == 1)) - janela // 2, 0)
            descricao = "no início da primeira anomalia"
        print(f"\nExemplo: janela de {janela} linhas do teste {descricao} (o que o modelo recebe):")
        print("    instantes:", t_test[inicio:inicio + janela].tolist())
        print("    rótulos:  ", y_test[inicio:inicio + janela].tolist())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for linha, versao in enumerate(["original", "corrigida"]):
        t_train, t_test, y_test = rastreio[versao]
        ax = axes[linha, 0]
        ax.scatter(np.arange(len(t_train)), t_train, s=2, color="tab:blue", label="normal")
        ax.set(title=f"{versao.upper()} - treino")
        ax = axes[linha, 1]
        pos = np.arange(len(t_test))
        ax.scatter(pos[y_test == 0], t_test[y_test == 0], s=2, color="tab:blue", label="normal")
        ax.scatter(pos[y_test == 1], t_test[y_test == 1], s=2, color="tab:red", label="anomalia")
        ax.set(title=f"{versao.upper()} - teste")
    for ax in axes.ravel():
        ax.set(xlabel="posição da linha no array entregue ao modelo", ylabel="instante original (time)")
        ax.grid(alpha=0.3)
        ax.legend(markerscale=5)
    fig.suptitle("Série íntegra = reta contínua subindo. Degraus e quedas = série cortada e emendada",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================================
# 2) Treino e avaliação do GANF (mesma configuração do notebook da aula)
# ============================================================================

def _loader(X, y):
    return get_dataloader(X, y, window_size=WINDOW, stride_size=1, batch_size=32, shuffle=False,
                          num_workers=0, persistent_workers=False, drop_last=False)


def treinar_e_avaliar_ganf(X_train, y_train, X_test, y_test, seed=0, epochs=2, interactions=2):
    """Treina o GANF como na aula e devolve (y, y_hat): gabarito e nota de cada ponto do teste."""
    torch.manual_seed(seed)
    model = GANF(n_blocks=6, input_features=1, hidden_size=32, n_hidden=1, model="MAF",
                 device="cpu", time_series_amount=X_train.shape[1])
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "weight_decay": 0.1}, {"params": [model.adjacency_matrix]}],
        lr=1e-4, weight_decay=0.0,
    )
    loss = LagrangeLoss(primal_loss=ForwardLoss(), dual_losses=[TraceExponentialMatrixLoss()])
    LagrangeTrainer().fit(model=model, optimizer=optimizer, loss=loss, train_loader=_loader(X_train, y_train),
                          epochs=epochs, interactions=interactions)
    model.eval()
    y_hat, y = StandardPredictor().predict(function=model, dataloader=_loader(X_test, y_test),
                                           pointwise=True, aggregation="mean", device="cpu")
    return np.ravel(y), np.ravel(y_hat)


def treinar_e_avaliar_hitachi(X_train, y_train, X_test, y_test, seed=0, epochs=30):
    """Treina o Hitachi como na seção do notebook e devolve (y, y_hat): gabarito e nota de cada ponto do teste."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_train)            # média e desvio do TREINO
    X_train = scaler.transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")
    model = WindowedHitachi(window_size=WINDOW, n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    UnsupervisedTrainer().fit(model=model, optimizer=optimizer, loss=MSELoss(),
                              train_loader=_loader(X_train, y_train), epochs=epochs)
    model.eval()
    y_hat, y = StandardPredictor().predict(function=ReconstructionError(model), dataloader=_loader(X_test, y_test),
                                           pointwise=True, aggregation="mean", device="cpu")
    return np.ravel(y), np.ravel(y_hat)


# ============================================================================
# 3) Cenários A, B, C e D
# ============================================================================

# Intervalos que o código ORIGINAL realmente gera (inclusive o (200, 13) vazio e o (150, 200) encurtado)
_INTERVALOS_ORIGINAL = {(4, 5): [(150, 200), (400, 600), (800, 1000), (4700, 4900)],
                        (1, 0): [(200, 13), (26, 39), (52, 65), (4887, 4900)]}
# Posições que o código CORRIGIDO usa no teste (iguais às de sample_data_utils.py)
_ANOMALIAS_CORRIGIDA = [([4, 5], [150, 600, 1100], 50), ([1, 0], [400, 900, 1350], 13)]


def _dados_com_k(versao, k):
    """Réplica de cada versão com anomalias de k desvios-padrão (usada só nos cenários C e D).

    A lógica de divisão é a mesma de cada arquivo; muda apenas o tamanho somado nas anomalias.
    """
    X = pd.read_csv(CSV).drop(columns=["U", "time"]).to_numpy()

    if versao == "corrigida":
        n_train = int(0.7 * X.shape[0])
        X_train, X_test = X[:n_train].copy(), X[n_train:].copy()
        y_train, y_test = np.zeros(len(X_train), dtype=int), np.zeros(len(X_test), dtype=int)
        std = X_train.std(axis=0)
        for features, starts, length in _ANOMALIAS_CORRIGIDA:
            for start in starts:
                X_test[start:start + length, features] += k * std[features]
                y_test[start:start + length] = 1
        return X_train, y_train, X_test, y_test

    y = np.zeros(X.shape[0])
    std = X.std(axis=0)
    for features, intervalos in _INTERVALOS_ORIGINAL.items():
        features = list(features)
        for start, end in intervalos:
            X[start:end, features] += k * std[features]
            y[start:end] = 1
    X_normal, X_abnormal = X[y == 0], X[y == 1]
    n = len(X_abnormal)
    X_test = np.vstack([X_normal[:n], X_abnormal])
    X_train = X_normal[n:]
    y_test = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    y_train = np.zeros_like(X_train[:, 0])
    return X_train, y_train, X_test, y_test


def dados(versao, k=None):
    """k=None usa o arquivo de verdade da versão; k=número usa anomalias de k desvios-padrão."""
    return carregar(versao) if k is None else _dados_com_k(versao, k)


def rodar_cenarios(treinar_e_avaliar, modelo, k=2, seed=0, graficos_individuais=True):
    """Treina o modelo nos 4 cenários e devolve os resultados de cada um."""
    cenarios = {
        "A: original + anomalias originais": ("original", None),
        "B: corrigida + anomalias originais": ("corrigida", None),
        f"C: original + anomalias de {k} desvios": ("original", k),
        f"D: corrigida + anomalias de {k} desvios": ("corrigida", k),
    }
    resultados = {}
    for nome, (versao, k_cenario) in cenarios.items():
        print(f"\n>>> {modelo} | {nome}")
        X_train, y_train, X_test, y_test = dados(versao, k_cenario)
        y, y_hat = treinar_e_avaliar(X_train, y_train, X_test, y_test, seed=seed)
        resultados[nome] = {
            "versao": versao, "k": k_cenario, "y": y, "y_hat": y_hat,
            "auc": roc_auc_score(y, y_hat), "ap": average_precision_score(y, y_hat),
            "prevalencia": float(np.mean(y)),
        }
        if graficos_individuais:
            plot_roc_pr_curves(y, y_hat, title=f"{modelo} | {nome}")
    return resultados


def rodar_cenarios_ganf(k=2, seed=0, graficos_individuais=True):
    return rodar_cenarios(treinar_e_avaliar_ganf, "GANF", k=k, seed=seed, graficos_individuais=graficos_individuais)


def rodar_cenarios_hitachi(k=2, seed=0, graficos_individuais=True):
    return rodar_cenarios(treinar_e_avaliar_hitachi, "Hitachi", k=k, seed=seed,
                          graficos_individuais=graficos_individuais)


def comparar_modelos(resultados_por_modelo):
    """Tabela com os 4 cenários nas linhas e AUC/AP de cada modelo nas colunas."""
    linhas = []
    for modelo, resultados in resultados_por_modelo.items():
        for nome, r in resultados.items():
            linhas.append({"cenário": nome, "modelo": modelo, "AUC-ROC": round(r["auc"], 4), "AP": round(r["ap"], 4)})
    return pd.DataFrame(linhas).pivot(index="cenário", columns="modelo", values=["AUC-ROC", "AP"])


def tabela_resultados(resultados):
    return pd.DataFrame([
        {"cenário": nome, "AUC-ROC": round(r["auc"], 4), "AP": round(r["ap"], 4),
         "AP do acaso (% anomalias)": round(r["prevalencia"], 3)}
        for nome, r in resultados.items()
    ])


def _estilo(r):
    """Cor = tamanho da anomalia; traço = divisão (tracejado = original, contínuo = corrigida)."""
    cor = "tab:orange" if r["k"] is None else "tab:blue"
    traco = "--" if r["versao"] == "original" else "-"
    return cor, traco


def graficos_comparativos(resultados, modelo=None):
    """As 4 curvas ROC juntas, as 4 PR juntas e as barras de AUC e AP lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    ax = axes[0]
    for nome, r in resultados.items():
        cor, traco = _estilo(r)
        fpr, tpr, _ = roc_curve(r["y"], r["y_hat"])
        ax.plot(fpr, tpr, traco, color=cor, lw=2.2, label=f"{nome[0]} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="acaso (0,50)")
    ax.set(title="ROC dos 4 cenários", xlabel="Taxa de falsos positivos (FPR)",
           ylabel="Taxa de verdadeiros positivos (TPR)", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))

    ax = axes[1]
    for nome, r in resultados.items():
        cor, traco = _estilo(r)
        precision, recall, _ = precision_recall_curve(r["y"], r["y_hat"])
        ax.plot(recall, precision, traco, color=cor, lw=2.2, drawstyle="steps-post",
                label=f"{nome[0]} (AP = {r['ap']:.3f}; acaso = {r['prevalencia']:.2f})")
    ax.set(title="Precision-Recall dos 4 cenários", xlabel="Recall (anomalias detectadas)",
           ylabel="Precisão (acusações corretas)", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))

    ax = axes[2]
    letras = [nome[0] for nome in resultados]
    x = np.arange(len(letras))
    aucs = [r["auc"] for r in resultados.values()]
    aps = [r["ap"] for r in resultados.values()]
    barras_auc = ax.bar(x - 0.2, aucs, width=0.4, color="tab:green", label="AUC-ROC")
    barras_ap = ax.bar(x + 0.2, aps, width=0.4, color="tab:purple", label="AP")
    ax.scatter(x + 0.2, [r["prevalencia"] for r in resultados.values()], marker="_", s=600,
               color="black", zorder=3, label="AP do acaso")
    ax.bar_label(barras_auc, fmt="%.3f", fontsize=8)
    ax.bar_label(barras_ap, fmt="%.3f", fontsize=8)
    ax.set(title="Resumo", xticks=x, xticklabels=letras, ylim=(0, 1.1), ylabel="valor")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right" if ax is not axes[1] else "lower left")
    prefixo = f"{modelo} | " if modelo else ""
    fig.suptitle(prefixo + "Laranja = anomalias originais | Azul = anomalias de k desvios | "
                 "Tracejado = divisão original | Contínuo = divisão corrigida", fontsize=12)
    plt.tight_layout()
    plt.show()


def notas_ao_longo_do_tempo(resultados, modelo=None):
    """Nota do modelo em cada instante do teste, com as anomalias do gabarito em vermelho."""
    fig, axes = plt.subplots(len(resultados), 1, figsize=(15, 3 * len(resultados)))
    for ax, (nome, r) in zip(np.atleast_1d(axes), resultados.items()):
        nota, gabarito = r["y_hat"], r["y"]
        ax.plot(nota, lw=0.8, color="tab:blue", label="nota do modelo")
        ax.fill_between(np.arange(len(nota)), nota.min(), nota.max(), where=gabarito == 1,
                        color="red", alpha=0.2, label="anomalia (gabarito)")
        ax.set(title=f"{modelo} | {nome}" if modelo else nome, ylabel="nota")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("posição no teste")
    plt.tight_layout()
    plt.show()
