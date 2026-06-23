import os
import time
import torch
from ultralytics import YOLO
from thop import profile

def main():
    # ----------------------------------------------------------------------
    # Generación Automática de Archivos .YAML
    # ----------------------------------------------------------------------
    entregable_dir = "dataset_entregable"
    path_absoluto = os.path.abspath(entregable_dir).replace("\\", "/")
    nombres_clases = "[" + ", ".join([f"'class_{i}'" for i in range(43)]) + "]"

    for escenario in ["clean", "rain", "fog"]:
        yaml_path = os.path.join(entregable_dir, f"dataset_{escenario}.yaml")
        if not os.path.exists(yaml_path):
            contenido = (
                f"path: {path_absoluto}\n"
                f"train: train/images\n"
                f"val: val_{escenario}/images\n"
                f"nc: 43\n"
                f"names: {nombres_clases}\n"
            )
            with open(yaml_path, 'w') as f:
                f.write(contenido)

    # ----------------------------------------------------------------------
    # MEJORA 1: Entrenamiento con hiperparámetros orientados a eficiencia
    # ----------------------------------------------------------------------
    print("=== FASE 1: ENTRENAMIENTO Y EVALUACIÓN DE CONTROL ===")

    model_base = YOLO("yolov8n.pt")

    model_base.train(
        data="dataset_entregable/dataset_clean.yaml",
        epochs=500,           # Epocas para hacer mas preciso al modelo
        patience=50,          # Early stopping: corta si no mejora en 50 épocas
        imgsz=640,
        device="0",
        # --- NUEVAS OPTIMIZACIONES ---
        cos_lr=True,          # Cosine LR annealing: mejor convergencia final
        dropout=0.0,          # YOLOv8n ya es pequeño; sin dropout para max precisión
        cache="ram",          # Cachear dataset en RAM para entrenamiento más rápido
        workers=4,            # Carga de datos paralela
        batch=16,             # Batch explícito para reproducibilidad
        optimizer="AdamW",    # AdamW converge mejor que SGD en pocas épocas
        close_mosaic=10,      # Desactivar mosaic en las últimas 10 épocas (mejor mAP final)
        augment=True,         # Augmentations de Ultralytics activadas
        amp=True,             # Entrenamiento en FP16 mixto (más rápido en RTX 3060)
    )

    baseline_path = os.path.join(model_base.trainer.save_dir, "weights", "best.pt")
    if not os.path.exists(baseline_path):
        print("Error: No se guardó el modelo base correctamente.")
        return

    model_eval = YOLO(baseline_path)

    print("\n Auditando caída de precisión (mAP) bajo clima adverso...")
    for escenario in ["clean", "rain", "fog"]:
        metrics = model_eval.val(
            data=f"dataset_entregable/dataset_{escenario}.yaml",
            split="val",
            verbose=False
        )
        print(f"  [-] mAP50 ({escenario}): {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")

    # ----------------------------------------------------------------------
    # Auditoría Computacional
    # ----------------------------------------------------------------------
    print("\n=== FASE 2: AUDITORÍA COMPUTACIONAL TEÓRICA ===")

    peso_fp32 = os.path.getsize(baseline_path) / (1024 * 1024)
    print(f"[-] Tamaño original en Disco: {peso_fp32:.2f} MB")

    device = next(model_eval.model.parameters()).device
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    flops, params = profile(model_eval.model, inputs=(dummy_input,), verbose=False)
    print(f"[-] GFLOPS Teóricos (FP32): {flops / 1e9:.3f}")
    print(f"[-] Cantidad Total de Parámetros: {params:,}")

    # Recarga limpia para benchmark CPU (elimina hooks de thop)
    model_eval = YOLO(baseline_path)
    model_eval.to("cpu")

    img_dir = "dataset_entregable/val_clean/images"
    imagenes_prueba = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir)[:50]
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if imagenes_prueba:
        _ = model_eval(imagenes_prueba[0], device="cpu", verbose=False)

        start_time = time.time()
        for img in imagenes_prueba:
            _ = model_eval(img, device="cpu", verbose=False)
        end_time = time.time()

        fps_base = len(imagenes_prueba) / (end_time - start_time)
        print(f"[-] Rendimiento CPU Baseline (.pt): {fps_base:.2f} FPS")

if __name__ == "__main__":
    main()