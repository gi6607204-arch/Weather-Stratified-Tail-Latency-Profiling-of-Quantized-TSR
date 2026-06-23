import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = "figures"


def find_results_csv():
    candidates = glob.glob("runs/detect/train*/results.csv")
    candidates = [c for c in candidates if os.path.exists(c)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def col(df, *names):
    """Return the first column that exists (names vary across versions)."""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def main():
    path = find_results_csv()
    if path is None:
        print("Error: 'runs/detect/train*/results.csv' not found.")
        print("       Run train.py first.")
        return

    print(f" -> Reading: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()          

    epoch = col(df, "epoch")
    map50 = col(df, "metrics/mAP50(B)", "metrics/mAP_0.5", "metrics/mAP50")
    map5095 = col(df, "metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "metrics/mAP50-95")
    prec = col(df, "metrics/precision(B)", "metrics/precision")
    rec = col(df, "metrics/recall(B)", "metrics/recall")

    tr_box = col(df, "train/box_loss")
    tr_cls = col(df, "train/cls_loss")
    tr_dfl = col(df, "train/dfl_loss")
    va_box = col(df, "val/box_loss")
    va_cls = col(df, "val/cls_loss")
    va_dfl = col(df, "val/dfl_loss")
    lr = col(df, "lr/pg0")

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    os.makedirs(FIG_DIR, exist_ok=True)

    if map50 is not None:
        best_idx = map50.idxmax()
        best_ep = int(epoch.iloc[best_idx]) if epoch is not None else best_idx
        best_val = map50.iloc[best_idx]
    else:
        best_ep, best_val = None, None

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Training dynamics — YOLOv8n", fontsize=14, fontweight="bold")

    # (a) mAP
    if map50 is not None:
        ax[0, 0].plot(epoch, map50, label="mAP@50", color="#2563eb", lw=2)
    if map5095 is not None:
        ax[0, 0].plot(epoch, map5095, label="mAP@50-95", color="#16a34a", lw=2)
    if best_ep is not None:
        ax[0, 0].axvline(best_ep, color="#dc2626", ls="--", alpha=0.7,
                         label=f"Best epoch ({best_ep})")
        ax[0, 0].scatter([best_ep], [best_val], color="#dc2626", zorder=5)
    ax[0, 0].set_title("(a) Accuracy (mAP)")
    ax[0, 0].set_xlabel("Epoch"); ax[0, 0].set_ylabel("mAP")
    ax[0, 0].legend()

    if tr_box is not None:
        ax[0, 1].plot(epoch, tr_box, label="train box", color="#2563eb", lw=1.5)
        ax[0, 1].plot(epoch, tr_cls, label="train cls", color="#16a34a", lw=1.5)
        ax[0, 1].plot(epoch, tr_dfl, label="train dfl", color="#d97706", lw=1.5)
    if va_box is not None:
        ax[0, 1].plot(epoch, va_box, label="val box", color="#2563eb", ls="--", lw=1.5)
        ax[0, 1].plot(epoch, va_cls, label="val cls", color="#16a34a", ls="--", lw=1.5)
        ax[0, 1].plot(epoch, va_dfl, label="val dfl", color="#d97706", ls="--", lw=1.5)
    ax[0, 1].set_title("(b) Losses (solid = train, dashed = val)")
    ax[0, 1].set_xlabel("Epoch"); ax[0, 1].set_ylabel("Loss")
    ax[0, 1].legend(fontsize=8, ncol=2)

    if prec is not None:
        ax[1, 0].plot(epoch, prec, label="Precision", color="#7c3aed", lw=2)
    if rec is not None:
        ax[1, 0].plot(epoch, rec, label="Recall", color="#db2777", lw=2)
    ax[1, 0].set_title("(c) Precision and Recall")
    ax[1, 0].set_xlabel("Epoch"); ax[1, 0].set_ylabel("Value")
    ax[1, 0].legend()

    if lr is not None:
        ax[1, 1].plot(epoch, lr, color="#0891b2", lw=2)
    ax[1, 1].set_title("(d) Learning rate (cosine annealing)")
    ax[1, 1].set_xlabel("Epoch"); ax[1, 1].set_ylabel("LR")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = os.path.join(FIG_DIR, "training_curves.png")
    fig.savefig(out1, dpi=700, bbox_inches="tight")
    print(f" -> Figure saved: {out1}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if map50 is not None:
        ax2.plot(epoch, map50, label="mAP@50", color="#2563eb", lw=2.2)
    if map5095 is not None:
        ax2.plot(epoch, map5095, label="mAP@50-95", color="#16a34a", lw=2.2)
    if best_ep is not None:
        ax2.axvline(best_ep, color="#dc2626", ls="--", alpha=0.7,
                    label=f"Convergence (epoch {best_ep}, mAP@50={best_val:.3f})")
        ax2.scatter([best_ep], [best_val], color="#dc2626", zorder=5)
    ax2.set_title("Training convergence (YOLOv8n)", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP")
    ax2.legend()
    fig2.tight_layout()
    out2 = os.path.join(FIG_DIR, "mAP_curve.png")
    fig2.savefig(out2, dpi=700, bbox_inches="tight")
    print(f" -> Figure saved: {out2}")

    if best_ep is not None:
        print(f"\n -> Best mAP@50 = {best_val:.4f} at epoch {best_ep}.")
    print(" -> Done. Open the 'figures/' folder.")


if __name__ == "__main__":
    main()