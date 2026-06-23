import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = "figures"
CSV_SPEED = "dataset_entregable/resultados_benchmark.csv"
CSV_ROB = "dataset_entregable/resultados_robustez_final.csv"

BACKEND_ORDER = ["CPU FP32", "CPU INT8 (ORT)", "CPU INT8 (OV)",
                 "GPU FP32 (CUDA)", "GPU FP16 (TRT)"]
COND_ORDER = ["clean", "rain", "fog"]
COND_LABELS = {"clean": "Clean", "rain": "Rain", "fog": "Fog"}
COLORS = ["#94a3b8", "#60a5fa", "#2563eb", "#16a34a", "#dc2626"]


def label_bars(ax, fmt="{:.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h and not np.isnan(h):
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=7.5)


def plot_speed(df):
    piv = df.pivot_table(index="modelo_dispositivo", columns="condicion",
                         values="fps", aggfunc="mean")
    piv = piv.reindex([b for b in BACKEND_ORDER if b in piv.index])
    piv = piv.reindex(columns=[c for c in COND_ORDER if c in piv.columns])
    piv = piv.rename(columns=COND_LABELS)

    fig, ax = plt.subplots(figsize=(11, 6))
    piv.plot(kind="bar", ax=ax, width=0.78,
             color=["#3b82f6", "#22c55e", "#f59e0b"][:piv.shape[1]])
    ax.set_title("Inference speed by backend and weather condition",
                 fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("FPS (frames per second)")
    ax.legend(title="Condition")
    ax.tick_params(axis="x", rotation=20)
    label_bars(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_speed.png")
    fig.savefig(out, dpi=700, bbox_inches="tight")
    print(f" -> {out}")

    fps_mean = df.groupby("modelo_dispositivo")["fps"].mean()
    if "CPU FP32" in fps_mean.index and fps_mean["CPU FP32"] > 0:
        base = fps_mean["CPU FP32"]
        speedup = (fps_mean / base).reindex(
            [b for b in BACKEND_ORDER if b in fps_mean.index])

        fig2, ax2 = plt.subplots(figsize=(9, 5.5))
        bars = ax2.bar(range(len(speedup)), speedup.values,
                       color=COLORS[:len(speedup)])
        ax2.set_xticks(range(len(speedup)))
        ax2.set_xticklabels(speedup.index, rotation=20, ha="right")
        ax2.axhline(1.0, color="gray", ls="--", alpha=0.6)
        ax2.set_title("Average speedup vs CPU FP32", fontweight="bold")
        ax2.set_ylabel("Speedup (x)")
        for b, v in zip(bars, speedup.values):
            ax2.annotate(f"{v:.2f}x", (b.get_x() + b.get_width() / 2, v),
                         ha="center", va="bottom", fontsize=9)
        fig2.tight_layout()
        out2 = os.path.join(FIG_DIR, "fig_speedup.png")
        fig2.savefig(out2, dpi=700, bbox_inches="tight")
        print(f" -> {out2}")


def plot_size(df):
    sub = df.dropna(subset=["peso_mb"])
    sub = sub[sub["peso_mb"] > 0]
    if sub.empty:
        return
    sizes = sub.groupby("modelo_dispositivo")["peso_mb"].first()
    sizes = sizes.reindex([b for b in BACKEND_ORDER if b in sizes.index]).dropna()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(range(len(sizes)), sizes.values,
                  color=["#94a3b8", "#60a5fa", "#2563eb"][:len(sizes)])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes.index, rotation=15, ha="right")
    ax.set_title("Model size on disk", fontweight="bold")
    ax.set_ylabel("MB")
    for b, v in zip(bars, sizes.values):
        ax.annotate(f"{v:.2f} MB", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_size.png")
    fig.savefig(out, dpi=700, bbox_inches="tight")
    print(f" -> {out}")


def plot_robustness(df):
    df = df.copy()
    df["modelo"] = df["modelo"].replace({"INT8_OV": "INT8 (OpenVINO)", "FP32": "FP32"})
    piv = df.pivot_table(index="condicion", columns="modelo",
                         values="map50", aggfunc="mean")
    piv = piv.reindex([c for c in COND_ORDER if c in piv.index])
    piv.index = [COND_LABELS.get(c, c) for c in piv.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    piv.plot(kind="bar", ax=ax, width=0.7, color=["#1e40af", "#f59e0b"])
    ax.set_title("Robustness: FP32 vs INT8 mAP@50 by weather condition",
                 fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("mAP@50")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Model")
    label_bars(ax, fmt="{:.3f}")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_robustness.png")
    fig.savefig(out, dpi=700, bbox_inches="tight")
    print(f" -> {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})

    if os.path.exists(CSV_SPEED):
        print(f" -> Reading {CSV_SPEED}")
        ds = pd.read_csv(CSV_SPEED)
        plot_speed(ds)
        plot_size(ds)
    else:
        print(f" [!] {CSV_SPEED} not found (run benchmark.py).")

    if os.path.exists(CSV_ROB):
        print(f" -> Reading {CSV_ROB}")
        dr = pd.read_csv(CSV_ROB)
        plot_robustness(dr)
    else:
        print(f" [!] {CSV_ROB} not found (run eval_robustez_final.py).")

    print(" -> Done. Open the 'figures/' folder.")


if __name__ == "__main__":
    main()