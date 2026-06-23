import os
import urllib.request
import zipfile
import cv2
import shutil
import albumentations as A

# Configuraciones principales
GTSDB_URL = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"
ZIP_NAME, RAW_DIR = "FullIJCNN2013.zip", "dataset_raw"
YOLO_DIR, ENTREGABLE_DIR = "dataset_yolo", "dataset_entregable"

def descargar_y_extraer():
    # Descarga y extrae el dataset original si no existe
    if not os.path.exists(ZIP_NAME):
        print("Iniciando descarga del dataset (esto puede tardar unos minutos)...")
        # Configuramos un User-Agent de navegador real para evitar bloqueos del servidor
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(GTSDB_URL, ZIP_NAME)
        print("¡Descarga completada con éxito!")

    if not os.path.exists(RAW_DIR):
        print("Extrayendo el dataset original...")
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("Extracción completada.")

def convertir_a_yolo():
    # Parsea gt.txt a formato YOLO y convierte .ppm a .jpg
    base_raw = os.path.join(RAW_DIR, "FullIJCNN2013")
    gt_file = os.path.join(base_raw, "gt.txt")
    
    for sub in ["images", "labels"]: 
        os.makedirs(os.path.join(YOLO_DIR, sub), exist_ok=True)

    anotaciones = {}
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            for line in filter(str.strip, f):
                img, l, t, r, b, cls = line.strip().split(';')
                # Agrupa bounding boxes por imagen
                anotaciones.setdefault(img, []).append((cls, *map(int, [l, t, r, b])))

    # Procesamiento de imágenes y normalización de coordenadas
    for img_name in [f for f in os.listdir(base_raw) if f.endswith('.ppm')]:
        img = cv2.imread(os.path.join(base_raw, img_name))
        if img is None: continue
        
        h, w = img.shape[:2]
        base_name = img_name.replace('.ppm', '')
        cv2.imwrite(os.path.join(YOLO_DIR, "images", f"{base_name}.jpg"), img)
        
        with open(os.path.join(YOLO_DIR, "labels", f"{base_name}.txt"), 'w') as f:
            for cls, l, t, r, b in anotaciones.get(img_name, []):
                xc, yc = (l+r)/(2*w), (t+b)/(2*h)
                bw, bh = (r-l)/w, (b-t)/h
                # Clipping entre 0.0 y 1.0 para YOLO
                xc, yc, bw, bh = [max(0.0, min(1.0, v)) for v in (xc, yc, bw, bh)]
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

def generar_entregables():
    # Separa en train (0-599) y val (600+) e inyecta clima en validación
    carpetas = ['train', 'val_clean', 'val_rain', 'val_fog']
    for c in carpetas:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(ENTREGABLE_DIR, c, sub), exist_ok=True)

    t_rain = A.Compose([A.RandomRain(brightness_coefficient=0.8, drop_width=1, blur_value=3, p=1.0), A.MotionBlur(blur_limit=5, p=0.3)])
    t_fog = A.Compose([A.RandomFog(fog_coef_lower=0.4, fog_coef_upper=0.7, alpha_coef=0.08, p=1.0)])

    imgs = sorted(os.listdir(os.path.join(YOLO_DIR, "images")))
    
    for idx, img_name in enumerate(imgs):
        base_name = img_name.replace('.jpg', '')
        src_img = os.path.join(YOLO_DIR, "images", img_name)
        src_lbl = os.path.join(YOLO_DIR, "labels", f"{base_name}.txt")
        
        # Train split (Limpios)
        if idx < 600: 
            shutil.copy(src_img, os.path.join(ENTREGABLE_DIR, 'train', 'images', img_name))
            if os.path.exists(src_lbl): shutil.copy(src_lbl, os.path.join(ENTREGABLE_DIR, 'train', 'labels', f"{base_name}.txt"))
        
        # Validación split (Limpios, Lluvia, Niebla)
        else:
            img = cv2.imread(src_img)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generación en lote de las variantes para evitar repetir código
            variantes = {
                'val_clean': img,
                'val_rain': cv2.cvtColor(t_rain(image=img_rgb)['image'], cv2.COLOR_RGB2BGR),
                'val_fog': cv2.cvtColor(t_fog(image=img_rgb)['image'], cv2.COLOR_RGB2BGR)
            }

            for carpeta, img_aug in variantes.items():
                cv2.imwrite(os.path.join(ENTREGABLE_DIR, carpeta, 'images', img_name), img_aug)
                if os.path.exists(src_lbl): 
                    shutil.copy(src_lbl, os.path.join(ENTREGABLE_DIR, carpeta, 'labels', f"{base_name}.txt"))

def generar_yamls():
    # Crea los 3 archivos YAML de configuración para YOLOv8
    nombres = "[" + ", ".join([f"'class_{i}'" for i in range(43)]) + "]"
    for tipo in ['clean', 'rain', 'fog']:
        contenido = f"path: {os.path.abspath(ENTREGABLE_DIR)}\ntrain: train/images\nval: val_{tipo}/images\nnc: 43\nnames: {nombres}\n"
        with open(os.path.join(ENTREGABLE_DIR, f'dataset_{tipo}.yaml'), 'w') as f:
            f.write(contenido)

if __name__ == "__main__":
    descargar_y_extraer()
    convertir_a_yolo()
    generar_entregables()
    generar_yamls()