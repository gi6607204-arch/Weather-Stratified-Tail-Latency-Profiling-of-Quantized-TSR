# Foggy but Fast: Weather-Stratified Tail-Latency Profiling of Quantized Traffic-Sign Detectors

Code and data for a measurement study of a compact traffic-sign detector under
weather and under quantization. We train a single YOLOv8n model on the German
Traffic Sign Detection Benchmark (GTSDB), render a held-out partition into
matched clean, rain, and fog variants with shared labels, quantize the model to
INT8 with calibrated post-training quantization, and profile five CPU and GPU
backends. For every weather condition we report mean average precision (mAP)
next to the median (p50) and 95th-percentile (p95) latency, on the same images
and in one run.

The point of the project is not a new architecture. It is the missing
measurement: per-condition accuracy paired with tail latency, across the
quantization and hardware spectrum.

## Key results

Accuracy (matched evaluation images, FP32 vs INT8/OpenVINO):

| Condition | FP32 mAP@50 | FP32 mAP@50-95 | INT8 mAP@50 | INT8 mAP@50-95 | Drop vs clean |
|-----------|:-----------:|:--------------:|:-----------:|:--------------:|:-------------:|
| Clean     | 0.761       | 0.664          | 0.739       | 0.636          | --            |
| Rain      | 0.558       | 0.468          | 0.544       | 0.449          | -26.7%        |
| Fog       | 0.497       | 0.378          | 0.474       | 0.359          | -34.7%        |

Latency and speed (clean set, 100 images, 640 input):

| Backend              | FPS   | p50 (ms) | p95 (ms) | Tail (p95/p50) | Speedup |
|----------------------|:-----:|:--------:|:--------:|:--------------:|:-------:|
| CPU FP32             | 30.6  | 33.08    | 37.49    | 1.13           | 1.00x   |
| CPU INT8 (ORT)       | 36.6  | 26.84    | 31.22    | 1.16           | 1.21x   |
| CPU INT8 (OpenVINO)  | 66.7  | 14.71    | 19.64    | 1.34           | 2.39x   |
| GPU FP32 (CUDA)      | 113.8 | 7.72     | 11.23    | 1.46           | 4.13x   |
| GPU FP16 (TensorRT)  | 247.6 | 4.10     | 4.49     | 1.10           | 8.95x   |

Model size: FP32 ONNX 11.73 MB, INT8 3.31-3.45 MB (about 71% smaller).

Takeaways: weather is the dominant cost and falls entirely on accuracy;
INT8 quantization costs at most 4.6% mAP while cutting the model to a third of
its size and roughly doubling CPU throughput; latency is deterministic across
weather (weather moves pixels, not compute).

## Pipeline

The pipeline runs end-to-end from the public GTSDB benchmark.

1. **`main.py`** downloads GTSDB (FullIJCNN2013), converts the annotations to
   YOLO format, splits into 600 training and 81 held-out evaluation images, and
   renders the evaluation partition into `clean`, `rain`, and `fog` variants with
   Albumentations (rain = streaks + motion blur, fog = `RandomFog`). It writes
   `dataset_entregable/` and the three dataset YAMLs.

2. **Training** is run with Ultralytics YOLOv8n on the clean split:

   ```bash
   yolo detect train model=yolov8n.pt data=dataset_entregable/dataset_clean.yaml \
        epochs=500 imgsz=640 batch=16 optimizer=AdamW cos_lr=True patience=50
   ```

   The best checkpoint lands in `runs/detect/train/weights/best.pt` (early-stopped
   at epoch 212 in our run).

3. **`qat_onnx.py`** exports ONNX FP32 (simplified graph), applies static
   calibrated INT8 PTQ (per-channel weights, QDQ format, letterbox + RGB
   calibration over 100 training images), exports an INT8 OpenVINO IR, and
   evaluates per-condition mAP for FP32 and INT8.

4. **`eval_robusto.py`** produces the single clean robustness table (FP32 vs
   INT8/OpenVINO) over `clean/rain/fog` in one run.

5. **`benchmark.py`** measures speed and the latency distribution (FPS, p50, p95)
   for every available backend and every condition. Backends: CPU FP32, CPU INT8
   (ONNX Runtime), CPU INT8 (OpenVINO), GPU FP32 (CUDA), GPU FP16 (TensorRT).
   Preprocessing is kept outside the timed region.

6. **Figures**:
   - `graphtrain.py`, `graphbench.py` build the matplotlib figures.
   - `graphs_ggplot.R` rebuilds them in ggplot2 at 600 DPI.
   - `graphs_extra.R` adds the accuracy-vs-throughput plane, the p50-to-p95
     latency dumbbell, and the qualitative weather and detection panels.

## Repository structure

```
.
├── main.py                       # GTSDB download + YOLO conversion + weather splits
├── qat_onnx.py                   # ONNX export, INT8 PTQ, OpenVINO, mAP eval
├── eval_robusto.py               # clean robustness table (FP32 vs INT8)
├── benchmark.py                  # per-backend, per-condition latency (FPS/p50/p95)
├── graphtrain.py                 # training-curve figures (matplotlib)
├── graphbench.py                 # speed/size/speedup/robustness figures (matplotlib)
├── graphs_ggplot.R               # same figures in ggplot2 (600 DPI)
├── graphs_extra.R                # pareto, latency tail, qualitative panels
├── dataset_entregable/           # generated splits, ONNX/INT8 models, result CSVs
│   ├── train/  val_clean/  val_rain/  val_fog/
│   ├── best.onnx  yolov8n_int8.onnx  best_int8_openvino_model/
│   ├── dataset_{clean,rain,fog}.yaml
│   └── resultados_*.csv
├── runs/detect/train/weights/    # best.pt, last.pt
├── figures/                      # figures used in the paper
└── figures_R/                    # 600-DPI ggplot2 figures
```

## Requirements

Python (3.10 or newer is fine; the checked-in `.venv` points to another machine,
so create a fresh environment):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install ultralytics onnxruntime openvino opencv-python albumentations \
            numpy pandas matplotlib psutil
```

GPU backends are optional. CUDA and TensorRT need `onnxruntime-gpu` and the
matching NVIDIA stack; without them, `benchmark.py` runs the CPU backends only.

The ggplot2 figures need R (4.5 used here) with:

```r
install.packages(c("ggplot2","dplyr","tidyr","readr","scales",
                   "patchwork","ggrepel","jpeg"))
```

## Reproduce

```bash
python main.py            # build dataset_entregable/ from GTSDB
# train (see command above), then:
python qat_onnx.py        # ONNX FP32 + INT8 (ORT/OpenVINO) + mAP per condition
python eval_robusto.py    # clean robustness table
python benchmark.py       # latency p50/p95 per backend and condition
python graphbench.py      # or: Rscript graphs_ggplot.R && Rscript graphs_extra.R
```

Results are written as CSVs in `dataset_entregable/` and as figures in
`figures/` (matplotlib) or `figures_R/` (ggplot2).

## Dataset

GTSDB (FullIJCNN2013), 43 sign classes, downloaded automatically by `main.py`
from the public archive. The rain and fog variants are synthetic renderings of
the clean held-out partition, so the three conditions share one ground truth.
Validating on a real condition-stratified set (for example CURE-TSD) is left to
future work.

## Citation

If you use this code or the weather splits, please cite:

```bibtex
@inproceedings{foggybutfast2026,
  title     = {Foggy but Fast: Weather-Stratified Tail-Latency Profiling of
               Quantized Traffic-Sign Detectors},
  author    = {TODO: author list},
  booktitle = {Mexican International Conference on Artificial Intelligence (MICAI)},
  year      = {2026},
  note      = {to appear}
}
```

## License and acknowledgments

The detector and training rely on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
released under AGPL-3.0; downstream use of this repository should respect that
license. GTSDB is distributed under its own terms by the dataset authors. Choose
a license for the project code (MIT is a common choice) and add a `LICENSE` file
before publishing.
