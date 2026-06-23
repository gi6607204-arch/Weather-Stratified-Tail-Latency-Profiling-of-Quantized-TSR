import os
import time
import glob
import csv
import cv2
import numpy as np
import onnxruntime as ort


# ======================================================================
# CONFIGURACIÓN
# ======================================================================
ONNX_FP32 = "dataset_entregable/best.onnx"
ONNX_INT8 = "dataset_entregable/yolov8n_int8.onnx"
CSV_OUT   = "dataset_entregable/resultados_benchmark.csv"
TRT_CACHE = "./trt_cache"

# Si conoces la ruta exacta del modelo OpenVINO, ponla aquí (apunta al .xml).
OPENVINO_XML = None  # ej: "dataset_entregable/best_openvino_model/best.xml"

CONDICIONES = ["clean", "rain", "fog"]
N_IMAGENES  = 100
WARMUP_CPU  = 10
WARMUP_GPU  = 20


# ======================================================================
# PREPROCESAMIENTO
# ======================================================================
def preprocesar_imagen(path_img, size=(640, 640)):
    img = cv2.imread(path_img)
    if img is None:
        return None
    img_res = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    tensor = img_res.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)


def _num_threads_cpu():
    try:
        import psutil
        fisicos = psutil.cpu_count(logical=False)
        if fisicos:
            return fisicos
    except Exception:
        pass
    return os.cpu_count() or 4


# ======================================================================
# MOTOR DE MEDICIÓN
# ======================================================================
def _stats(latencias):
    if not latencias:
        return None
    arr = np.array(latencias)
    return {
        "fps":         1000.0 / np.mean(arr),
        "lat_mean_ms": float(np.mean(arr)),
        "lat_p50_ms":  float(np.percentile(arr, 50)),
        "lat_p95_ms":  float(np.percentile(arr, 95)),
    }


def medir(infer_fn, imagenes, warmup=10):
    """Preprocesamiento FUERA del cronómetro (medición justa)."""
    tensor_warmup = preprocesar_imagen(imagenes[0])
    for _ in range(warmup):
        infer_fn(tensor_warmup)

    latencias = []
    for path_img in imagenes:
        tensor = preprocesar_imagen(path_img)
        if tensor is None:
            continue
        t0 = time.perf_counter()
        infer_fn(tensor)
        latencias.append((time.perf_counter() - t0) * 1000)
    return _stats(latencias)


# ======================================================================
# BACKENDS
# ======================================================================
def backend_ort_cpu(model_path):
    opciones = ort.SessionOptions()
    opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opciones.intra_op_num_threads = _num_threads_cpu()
    opciones.inter_op_num_threads = 1
    opciones.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Sin 'optimized_model_filepath' -> evita los warnings de serialización.
    sesion = ort.InferenceSession(
        model_path, sess_options=opciones, providers=['CPUExecutionProvider']
    )
    nombre = sesion.get_inputs()[0].name
    return lambda t: sesion.run(None, {nombre: t})


def backend_ort_cuda(model_path):
    cuda_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }
    opciones = ort.SessionOptions()
    opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opciones.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sesion = ort.InferenceSession(
        model_path, sess_options=opciones,
        providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider'],
    )
    nombre = sesion.get_inputs()[0].name
    return lambda t: sesion.run(None, {nombre: t})


def backend_trt_fp16(model_path, cache_dir=TRT_CACHE):
    """GPU FP16 vía TensorRT. La 1ra corrida compila el engine (minutos)."""
    os.makedirs(cache_dir, exist_ok=True)
    trt_options = {
        'device_id': 0,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_dir,
        'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
    }
    opciones = ort.SessionOptions()
    opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sesion = ort.InferenceSession(
        model_path, sess_options=opciones,
        providers=[
            ('TensorrtExecutionProvider', trt_options),
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider',
        ],
    )
    nombre = sesion.get_inputs()[0].name
    return lambda t: sesion.run(None, {nombre: t})


def backend_openvino(xml_path):
    import openvino as ov
    core = ov.Core()
    modelo = core.read_model(xml_path)
    compilado = core.compile_model(modelo, "CPU")
    salida = compilado.output(0)
    return lambda t: compilado([t])[salida]


# ======================================================================
# UTILIDADES
# ======================================================================
def peso_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def peso_openvino_mb(xml_path):
    return peso_mb(xml_path) + peso_mb(xml_path.replace(".xml", ".bin"))


def buscar_openvino_xml():
    if OPENVINO_XML and os.path.exists(OPENVINO_XML):
        return OPENVINO_XML
    patrones = [
        "dataset_entregable/**/*openvino*model*/*.xml",
        "runs/detect/train*/weights/*openvino_model/*.xml",
        "**/*openvino_model/*.xml",
    ]
    candidatos = []
    for p in patrones:
        candidatos.extend(glob.glob(p, recursive=True))
    return max(candidatos, key=os.path.getmtime) if candidatos else None


def construir_backends():
    """Crea cada backend UNA sola vez (sesiones/engine reutilizados en las 3 condiciones)."""
    backends = []
    providers = ort.get_available_providers()

    try:
        backends.append({"etiqueta": "CPU FP32", "peso": peso_mb(ONNX_FP32),
                         "infer": backend_ort_cpu(ONNX_FP32)})
    except Exception as e:
        print(f"  [!] CPU FP32 no disponible: {e}")

    try:
        backends.append({"etiqueta": "CPU INT8 (ORT)", "peso": peso_mb(ONNX_INT8),
                         "infer": backend_ort_cpu(ONNX_INT8)})
    except Exception as e:
        print(f"  [!] CPU INT8 (ORT) no disponible: {e}")

    xml = buscar_openvino_xml()
    if xml:
        try:
            backends.append({"etiqueta": "CPU INT8 (OV)", "peso": peso_openvino_mb(xml),
                             "infer": backend_openvino(xml)})
            print(f"  [+] OpenVINO: {xml}")
        except ImportError:
            print("  [!] OpenVINO no instalado (pip install openvino).")
        except Exception as e:
            print(f"  [!] OpenVINO falló: {e}")
    else:
        print("  [!] OpenVINO: no se encontró modelo (exporta con qat_onnx.py o define OPENVINO_XML).")

    if 'CUDAExecutionProvider' in providers:
        try:
            backends.append({"etiqueta": "GPU FP32 (CUDA)", "peso": None,
                             "infer": backend_ort_cuda(ONNX_FP32)})
        except Exception as e:
            print(f"  [!] GPU FP32 (CUDA) falló: {e}")

    if 'TensorrtExecutionProvider' in providers:
        print("  -> Construyendo engine TensorRT...")
        try:
            backends.append({"etiqueta": "GPU FP16 (TRT)", "peso": None,
                             "infer": backend_trt_fp16(ONNX_FP32)})
        except Exception as e:
            print(f"  [!] TensorRT falló: {e}")

    return backends


# ======================================================================
# TABLA Y CSV
# ======================================================================
def imprimir_tabla(resultados, baseline_fps, titulo):
    etiquetas = [r["etiqueta"] for r in resultados]
    ancho_lbl, ancho_col = 24, 16
    ancho_total = ancho_lbl + 2 + ancho_col * len(etiquetas)

    def fila(nombre, valores):
        celdas = "".join(f"{v:>{ancho_col}}" for v in valores)
        print(f"  {nombre:<{ancho_lbl}}{celdas}")

    print("=" * ancho_total)
    print(f"{titulo:^{ancho_total}}")
    print("=" * ancho_total)
    fila("Métrica", etiquetas)
    print("-" * ancho_total)
    fila("Peso del modelo (MB)",
         [f"{r['peso']:.2f}" if r['peso'] else "—" for r in resultados])
    fila("FPS promedio", [f"{r['stats']['fps']:.2f}" for r in resultados])
    fila("Latencia media (ms)", [f"{r['stats']['lat_mean_ms']:.2f}" for r in resultados])
    fila("Latencia p50 (ms)", [f"{r['stats']['lat_p50_ms']:.2f}" for r in resultados])
    fila("Latencia p95 (ms)", [f"{r['stats']['lat_p95_ms']:.2f}" for r in resultados])
    fila("Speedup vs CPU FP32",
         [f"{r['stats']['fps'] / baseline_fps:.2f}x" if baseline_fps else "—"
          for r in resultados])
    print("=" * ancho_total)


def exportar_csv(filas, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["condicion", "modelo_dispositivo", "peso_mb", "fps",
                    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms"])
        for cond, r in filas:
            s = r["stats"]
            w.writerow([cond, r["etiqueta"],
                        f"{r['peso']:.2f}" if r['peso'] else "",
                        f"{s['fps']:.2f}", f"{s['lat_mean_ms']:.2f}",
                        f"{s['lat_p50_ms']:.2f}", f"{s['lat_p95_ms']:.2f}"])


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=== FASE 3: AUDITORÍA DE VELOCIDAD POR CONDICIÓN (clean/rain/fog) ===")

    for ruta, nombre in [(ONNX_FP32, "FP32"), (ONNX_INT8, "INT8")]:
        if not os.path.exists(ruta):
            print(f"Error: No se encontró '{ruta}' ({nombre})")
            return

    print(f" -> Providers ORT disponibles: {ort.get_available_providers()}")
    print("\n -> Construyendo backends...")
    backends = construir_backends()
    if not backends:
        print("Error: ningún backend disponible.")
        return

    filas_csv = []
    for cond in CONDICIONES:
        img_dir = f"dataset_entregable/val_{cond}/images"
        if not os.path.isdir(img_dir):
            print(f"\n[{cond}] No existe '{img_dir}', omitido.")
            continue
        imagenes = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:N_IMAGENES]
        if not imagenes:
            print(f"\n[{cond}] Sin imágenes en '{img_dir}', omitido.")
            continue

        print(f"\n--- Condición: {cond.upper()} ({len(imagenes)} imágenes) ---")
        resultados = []
        for b in backends:
            warm = WARMUP_GPU if "GPU" in b["etiqueta"] else WARMUP_CPU
            stats = medir(b["infer"], imagenes, warm)
            if stats:
                resultados.append({"etiqueta": b["etiqueta"], "peso": b["peso"], "stats": stats})

        if resultados:
            baseline = resultados[0]["stats"]["fps"]
            imprimir_tabla(resultados, baseline, f"VELOCIDAD — {cond.upper()}")
            mejor = max(resultados, key=lambda r: r["stats"]["fps"])
            print(f" -> Más rápido en {cond}: {mejor['etiqueta']} "
                  f"({mejor['stats']['fps'] / baseline:.2f}x vs CPU FP32)")
            for r in resultados:
                filas_csv.append((cond, r))

    if filas_csv:
        exportar_csv(filas_csv, CSV_OUT)
        print(f"\n -> CSV exportado para el paper: {CSV_OUT}")
    print("\nNota: la velocidad es casi identica entre clean/rain/fog "
          "(mismo modelo y mismo tamano de entrada; el clima solo cambia los "
          "pixeles, no el computo). La precision (mAP) si cambia, esa va en qat_onnx.py.")


if __name__ == "__main__":
    main()