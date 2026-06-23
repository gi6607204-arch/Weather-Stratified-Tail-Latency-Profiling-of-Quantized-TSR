import os
import glob
import shutil
import numpy as np
import cv2
from ultralytics import YOLO


# ----------------------------------------------------------------------
# NUEVO: letterbox (conserva proporción + relleno, igual que YOLO).
# El resize directo deformaba la imagen y arruinaba la calibración.
# ----------------------------------------------------------------------
def _letterbox(img, size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img_r
    return canvas


# ----------------------------------------------------------------------
# Cuantización estática INT8 con calibración real (PTQ calibrada).
# ARREGLOS CLAVE vs versión anterior (que daba mAP = 0.0000):
#   - per_channel=True   -> imprescindible para las cabezas de YOLO
#   - QuantFormat.QDQ    -> mejor precisión y soporte que QOperator
#   - letterbox + RGB    -> calibrar con la MISMA distribución que ve YOLO
# ----------------------------------------------------------------------
def calibrar_y_cuantizar_int8(onnx_fp32_path, onnx_int8_path, calibration_dir, num_imagenes=100):
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader,
            QuantType, QuantFormat
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        print("  [!] onnxruntime.quantization no disponible. Usando fallback.")
        return False

    class YOLOCalibrationReader(CalibrationDataReader):
        def __init__(self, img_dir, num_imgs):
            self.imagenes = [
                os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ][:num_imgs]
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.imagenes):
                return None
            img = cv2.imread(self.imagenes[self.idx])
            img = _letterbox(img, 640)             # conserva proporción
            img = img[:, :, ::-1]                   # BGR -> RGB
            tensor = np.ascontiguousarray(
                img.transpose(2, 0, 1)).astype(np.float32) / 255.0
            tensor = np.expand_dims(tensor, axis=0)
            self.idx += 1
            return {"images": tensor}

    onnx_prep_path = onnx_fp32_path.replace(".onnx", "_prep.onnx")
    try:
        quant_pre_process(
            input_model_path=onnx_fp32_path,
            output_model_path=onnx_prep_path,
            skip_symbolic_shape=True,
        )
        modelo_a_cuantizar = onnx_prep_path
        print("  [+] Pre-procesamiento del grafo completado.")
    except Exception as e:
        print(f"  [!] quant_pre_process falló ({e}), cuantizando el modelo original.")
        modelo_a_cuantizar = onnx_fp32_path

    reader = YOLOCalibrationReader(calibration_dir, num_imagenes)

    quantize_static(
        model_input=modelo_a_cuantizar,
        model_output=onnx_int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,        # QDQ: mejor precisión y soporte que QOperator
        per_channel=True,                    # CRÍTICO para detectores YOLO
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )

    if os.path.exists(onnx_prep_path):
        os.remove(onnx_prep_path)

    return True


# ----------------------------------------------------------------------
# Generación de YAMLs (clean/rain/fog) por si no existen.
# ----------------------------------------------------------------------
def generar_yamls(entregable_dir, n_clases=43):
    path_absoluto = os.path.abspath(entregable_dir).replace("\\", "/")
    nombres = "[" + ", ".join([f"'class_{i}'" for i in range(n_clases)]) + "]"
    for escenario in ["clean", "rain", "fog"]:
        yaml_path = os.path.join(entregable_dir, f"dataset_{escenario}.yaml")
        if not os.path.exists(yaml_path):
            with open(yaml_path, 'w') as f:
                f.write(
                    f"path: {path_absoluto}\n"
                    f"train: train/images\n"
                    f"val: val_{escenario}/images\n"
                    f"nc: {n_clases}\n"
                    f"names: {nombres}\n"
                )


# ----------------------------------------------------------------------
# Evaluación de mAP por condición (clean / rain / fog).
# ----------------------------------------------------------------------
def evaluar_robustez(onnx_path, etiqueta, entregable_dir, device="cpu", imgsz=640):
    """Devuelve {condicion: (map50, map50_95)} para clean/rain/fog."""
    resultados = {}
    if not os.path.exists(onnx_path):
        print(f"  [!] No existe {onnx_path}, omitiendo {etiqueta}.")
        return resultados
    try:
        modelo = YOLO(onnx_path, task="detect")
    except Exception as e:
        print(f"  [!] No se pudo cargar {etiqueta}: {e}")
        return resultados

    for cond in ["clean", "rain", "fog"]:
        yaml_path = os.path.join(entregable_dir, f"dataset_{cond}.yaml")
        if not os.path.exists(yaml_path):
            print(f"  [!] Falta {yaml_path}; omitiendo {cond}.")
            continue
        try:
            m = modelo.val(data=yaml_path, split="val", imgsz=imgsz,
                           device=device, verbose=False)
            resultados[cond] = (float(m.box.map50), float(m.box.map))
            print(f"  [{etiqueta:>4}] {cond:>5}: "
                  f"mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")
        except Exception as e:
            print(f"  [!] Falló eval {etiqueta}/{cond}: {e}")
    return resultados


def imprimir_tabla_robustez(rob_fp32, rob_int8):
    ancho = 78
    print("\n" + "=" * ancho)
    print(f"{'ROBUSTEZ POR CONDICIÓN (mAP) — FP32 vs INT8':^{ancho}}")
    print("=" * ancho)
    print(f"  {'Condición':<12}{'FP32 mAP50':>12}{'FP32 mAP':>12}"
          f"{'INT8 mAP50':>12}{'INT8 mAP':>12}{'Δ mAP50':>12}")
    print("-" * ancho)
    for cond in ["clean", "rain", "fog"]:
        f50, f = rob_fp32.get(cond, (None, None))
        i50, i = rob_int8.get(cond, (None, None))
        def s(x): return f"{x:.4f}" if x is not None else "—"
        delta = f"{(i50 - f50):+.4f}" if (f50 is not None and i50 is not None) else "—"
        print(f"  {cond:<12}{s(f50):>12}{s(f):>12}{s(i50):>12}{s(i):>12}{delta:>12}")
    print("=" * ancho)
    print("  (Δ mAP50 = INT8 − FP32. Negativo = el INT8 pierde precisión en esa condición.)")


def exportar_csv_robustez(rob_fp32, rob_int8, path):
    import csv
    with open(path, 'w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["condicion", "modelo", "map50", "map50_95"])
        for cond in ["clean", "rain", "fog"]:
            if cond in rob_fp32:
                w.writerow([cond, "FP32", f"{rob_fp32[cond][0]:.4f}", f"{rob_fp32[cond][1]:.4f}"])
            if cond in rob_int8:
                w.writerow([cond, "INT8", f"{rob_int8[cond][0]:.4f}", f"{rob_int8[cond][1]:.4f}"])
    print(f" -> CSV de robustez exportado: {path}")


def main():
    print("=== FASE 2: EXPORTACIÓN Y CUANTIZACIÓN OPTIMIZADA ===")

    patron = os.path.join("runs", "detect", "train*", "weights", "best.pt")
    modelos = glob.glob(patron)
    if not modelos:
        print("Error: No se encontró ningún 'best.pt' en 'runs/detect/train*'.")
        return

    baseline_path = max(modelos, key=os.path.getmtime)
    print(f" -> Modelo detectado: {baseline_path}")

    model = YOLO(baseline_path)
    output_dir = "dataset_entregable"
    os.makedirs(output_dir, exist_ok=True)

    onnx_fp32_path = os.path.join(output_dir, "best.onnx")
    onnx_int8_path = os.path.join(output_dir, "yolov8n_int8.onnx")
    calibration_dir = os.path.join(output_dir, "train", "images")

    # ---- Exportar ONNX FP32 -----------------------------------------
    print("\n[1/3] Exportando a ONNX FP32 con simplificación de grafo...")
    onnx_fp32_temp = model.export(
        format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17,
    )
    if onnx_fp32_temp and os.path.exists(onnx_fp32_temp):
        shutil.move(onnx_fp32_temp, onnx_fp32_path)
    else:
        posible = baseline_path.replace("best.pt", "best.onnx")
        if os.path.exists(posible):
            shutil.move(posible, onnx_fp32_path)
    print(f" -> ONNX FP32 guardado en: {onnx_fp32_path}")

    # ---- Cuantización INT8 ------------------------------------------
    print("\n[2/3] Aplicando cuantización estática INT8 calibrada...")
    if os.path.exists(calibration_dir) and os.path.exists(onnx_fp32_path):
        exito = calibrar_y_cuantizar_int8(onnx_fp32_path, onnx_int8_path, calibration_dir)
        if not exito:
            _exportar_int8_ultralytics(model, baseline_path, onnx_int8_path, output_dir)
    else:
        print("  [!] Directorio de calibración no encontrado. Usando fallback de Ultralytics.")
        _exportar_int8_ultralytics(model, baseline_path, onnx_int8_path, output_dir)

    # ---- OpenVINO IR ------------------------------------------------
    print("\n[3/3] Exportando a OpenVINO IR...")
    try:
        openvino_path = model.export(
            format="openvino", imgsz=640, int8=True,
            data="dataset_entregable/dataset_clean.yaml",
        )
        print(f" -> Modelo OpenVINO guardado en: {openvino_path}")
    except Exception as e:
        print(f"  [!] OpenVINO no disponible: {e}")

    # ---- Diagnóstico de tamaño --------------------------------------
    peso_original = os.path.getsize(baseline_path) / (1024 * 1024)
    peso_fp32 = os.path.getsize(onnx_fp32_path) / (1024 * 1024) if os.path.exists(onnx_fp32_path) else 0
    peso_int8 = os.path.getsize(onnx_int8_path) / (1024 * 1024) if os.path.exists(onnx_int8_path) else 0
    print(f"\n -> Peso original (.pt): {peso_original:.2f} MB")
    print(f" -> Peso ONNX FP32:      {peso_fp32:.2f} MB")
    print(f" -> Peso ONNX INT8:      {peso_int8:.2f} MB")
    if peso_fp32 > 0 and peso_int8 > 0:
        print(f" -> Reducción FP32→INT8: {((peso_fp32 - peso_int8) / peso_fp32) * 100:.1f}%")

    # ==================================================================
    # FASE 4: VERIFICACIÓN DE ROBUSTEZ — mAP por condición
    # ==================================================================
    print("\n=== FASE 4: VERIFICACIÓN DE ROBUSTEZ (mAP en clean/rain/fog) ===")
    generar_yamls(output_dir)

    print("\n -> Evaluando FP32 (GPU)...")
    rob_fp32 = evaluar_robustez(onnx_fp32_path, "FP32", output_dir, device="0")

    print("\n -> Evaluando INT8 (CPU — el INT8 estático corre en CPU EP)...")
    rob_int8 = evaluar_robustez(onnx_int8_path, "INT8", output_dir, device="cpu")

    imprimir_tabla_robustez(rob_fp32, rob_int8)
    exportar_csv_robustez(rob_fp32, rob_int8,
                          os.path.join(output_dir, "resultados_robustez.csv"))


def _exportar_int8_ultralytics(model, baseline_path, onnx_int8_path, output_dir):
    """Fallback: cuantización dinámica de Ultralytics."""
    onnx_int8_temp = model.export(
        format="onnx", int8=True,
        data="dataset_entregable/dataset_clean.yaml",
        imgsz=640, simplify=True
    )
    if onnx_int8_temp and os.path.exists(onnx_int8_temp):
        shutil.move(onnx_int8_temp, onnx_int8_path)
    else:
        posible = baseline_path.replace("best.pt", "best_int8.onnx")
        if os.path.exists(posible):
            shutil.move(posible, onnx_int8_path)


if __name__ == "__main__":
    main()