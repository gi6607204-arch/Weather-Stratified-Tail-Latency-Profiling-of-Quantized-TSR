"""
Tabla ÚNICA y limpia de robustez para el paper.
Mide FP32 (ONNX) e INT8 (OpenVINO) sobre los MISMOS datos, en una sola corrida,
para clean / rain / fog. Evita mezclar números de corridas distintas.

Uso:  py eval_robustez_final.py
"""
import os
import glob
from ultralytics import YOLO


# ----------------------------------------------------------------------
# Localización automática de los modelos
# ----------------------------------------------------------------------
def buscar_fp32_onnx():
    candidatos = [
        "dataset_entregable/best.onnx",
    ]
    candidatos += glob.glob("runs/detect/train*/weights/best.onnx")
    candidatos = [c for c in candidatos if os.path.exists(c)]
    return max(candidatos, key=os.path.getmtime) if candidatos else None


def buscar_fp32_pt():
    candidatos = glob.glob("runs/detect/train*/weights/best.pt")
    candidatos = [c for c in candidatos if os.path.exists(c)]
    return max(candidatos, key=os.path.getmtime) if candidatos else None


def buscar_openvino():
    patrones = [
        "runs/detect/train*/weights/*openvino_model",
        "dataset_entregable/**/*openvino_model",
        "**/*openvino_model",
    ]
    candidatos = []
    for p in patrones:
        candidatos.extend(glob.glob(p, recursive=True))
    candidatos = [c for c in candidatos if os.path.isdir(c)]
    return max(candidatos, key=os.path.getmtime) if candidatos else None


# ----------------------------------------------------------------------
# Evaluación de mAP por condición
# ----------------------------------------------------------------------
def evaluar(modelo_path, etiqueta, device):
    """Devuelve {cond: (map50, map50_95)} evaluando en clean/rain/fog."""
    resultados = {}
    if modelo_path is None or not os.path.exists(modelo_path):
        print(f"  [!] No se encontró el modelo {etiqueta}; se omite.")
        return resultados
    try:
        modelo = YOLO(modelo_path, task="detect")
    except Exception as e:
        print(f"  [!] No se pudo cargar {etiqueta}: {e}")
        return resultados

    for cond in ["clean", "rain", "fog"]:
        yaml_path = f"dataset_entregable/dataset_{cond}.yaml"
        if not os.path.exists(yaml_path):
            print(f"  [!] Falta {yaml_path}; se omite {cond}.")
            continue
        try:
            r = modelo.val(data=yaml_path, split="val", imgsz=640,
                           device=device, verbose=False)
            resultados[cond] = (float(r.box.map50), float(r.box.map))
            print(f"  [{etiqueta:>4}] {cond:>5}: "
                  f"mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}")
        except Exception as e:
            print(f"  [!] Falló {etiqueta}/{cond}: {e}")
    return resultados


# ----------------------------------------------------------------------
# Tabla + CSV
# ----------------------------------------------------------------------
def imprimir_tabla(rob_fp32, rob_int8):
    ancho = 80
    print("\n" + "=" * ancho)
    print(f"{'ROBUSTEZ POR CONDICIÓN (mAP) — FP32 vs INT8 (OpenVINO)':^{ancho}}")
    print("=" * ancho)
    print(f"  {'Condición':<12}{'FP32 mAP50':>12}{'FP32 mAP':>12}"
          f"{'INT8 mAP50':>12}{'INT8 mAP':>12}{'Δ mAP50':>12}{'Caída %':>8}")
    print("-" * ancho)
    for cond in ["clean", "rain", "fog"]:
        f50, f = rob_fp32.get(cond, (None, None))
        i50, i = rob_int8.get(cond, (None, None))

        def s(x):
            return f"{x:.4f}" if x is not None else "—"

        if f50 is not None and i50 is not None:
            delta = f"{(i50 - f50):+.4f}"
            caida = f"{((f50 - i50) / f50 * 100):.1f}%" if f50 > 0 else "—"
        else:
            delta, caida = "—", "—"
        print(f"  {cond:<12}{s(f50):>12}{s(f):>12}{s(i50):>12}{s(i):>12}"
              f"{delta:>12}{caida:>8}")
    print("=" * ancho)
    print("  Δ mAP50 = INT8 − FP32 (negativo = el INT8 pierde precisión).")
    print("  Ambos modelos evaluados en la MISMA corrida y sobre los mismos datos.")


def exportar_csv(rob_fp32, rob_int8, path):
    import csv
    with open(path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["condicion", "modelo", "map50", "map50_95"])
        for cond in ["clean", "rain", "fog"]:
            if cond in rob_fp32:
                w.writerow([cond, "FP32", f"{rob_fp32[cond][0]:.4f}",
                            f"{rob_fp32[cond][1]:.4f}"])
            if cond in rob_int8:
                w.writerow([cond, "INT8_OV", f"{rob_int8[cond][0]:.4f}",
                            f"{rob_int8[cond][1]:.4f}"])
    print(f"\n -> CSV exportado: {path}")


def main():
    print("=== TABLA FINAL DE ROBUSTEZ (una sola corrida) ===\n")

    # FP32: usa el .pt si existe (más fiel a las métricas de train.py);
    # si no, cae al ONNX FP32.
    fp32_pt = buscar_fp32_pt()
    fp32_onnx = buscar_fp32_onnx()
    fp32_modelo = fp32_pt if fp32_pt else fp32_onnx
    int8_ov = buscar_openvino()

    print(f" -> FP32: {fp32_modelo}")
    print(f" -> INT8 (OpenVINO): {int8_ov}\n")

    print(" -> Evaluando FP32...")
    # .pt corre en GPU (device 0); ONNX también puede, pero cpu es más seguro
    dev_fp32 = "0" if (fp32_pt and fp32_modelo == fp32_pt) else "cpu"
    rob_fp32 = evaluar(fp32_modelo, "FP32", device=dev_fp32)

    print("\n -> Evaluando INT8 (OpenVINO, CPU)...")
    rob_int8 = evaluar(int8_ov, "INT8", device="cpu")

    imprimir_tabla(rob_fp32, rob_int8)
    exportar_csv(rob_fp32, rob_int8, "dataset_entregable/resultados_robustez_final.csv")


if __name__ == "__main__":
    main()