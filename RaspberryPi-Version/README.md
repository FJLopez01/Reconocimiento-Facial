Entiendo que deseas recrear el tutorial de reconocimiento facial en Raspberry Pi y presentar la información en un archivo `README.md` con formato de escritura para GitHub, incluyendo fórmulas en LaTeX y emojis para mejorar la presentación. A continuación, te proporciono una guía detallada para lograrlo.

---

# 😃 Reconocimiento Facial con Raspberry Pi utilizando Python y OpenCV

Este proyecto implementa un sistema de reconocimiento facial en tiempo real en una Raspberry Pi, empleando las bibliotecas **Python**, **OpenCV** y **face_recognition**.

---

## 📋 Tabla de Contenidos

1. [Introducción](#-introducción)
2. [Instalación de Dependencias](#-instalación-de-dependencias)
3. [Preparación del Conjunto de Datos](#-preparación-del-conjunto-de-datos)
4. [Codificación de Rostros](#-codificación-de-rostros)
5. [Reconocimiento Facial en Tiempo Real](#-reconocimiento-facial-en-tiempo-real)
6. [Conclusión](#-conclusión)

---

## 📖 Introducción

El reconocimiento facial es una tecnología que permite identificar o verificar a una persona a partir de una imagen o video. En este proyecto, implementaremos un sistema de reconocimiento facial en una Raspberry Pi, aprovechando su portabilidad y eficiencia energética.

---

## 🛠️ Instalación de Dependencias

Antes de comenzar, es esencial asegurarse de que tu Raspberry Pi esté actualizada y tenga instaladas las dependencias necesarias.

### 🔄 Actualización del Sistema

```bash
sudo apt-get update
sudo apt-get upgrade
```

### 📦 Instalación de Paquetes Necesarios

```bash
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python3-dev
```

### 🖼️ Instalación de OpenCV

Para instalar OpenCV en tu Raspberry Pi, puedes seguir las instrucciones detalladas en el siguiente enlace: [Instalación de OpenCV en Raspberry Pi](https://pyimagesearch.com/2018/09/19/pip-install-opencv/).

### 🧠 Instalación de dlib y face_recognition

```bash
pip3 install dlib
pip3 install face_recognition
```

Además, instala la biblioteca `imutils` para funciones adicionales:

```bash
pip3 install imutils
```

---

## 🗂️ Preparación del Conjunto de Datos

Organiza las imágenes de las personas que deseas reconocer en un directorio estructurado de la siguiente manera:

```
dataset/
├── persona1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── persona2/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── ...
```

---

## 🧬 Codificación de Rostros

Crea un script llamado `encode_faces.py` para procesar las imágenes y generar las codificaciones faciales:

```python
# importaciones necesarias
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# ruta al dataset
dataset_path = "dataset"
# ruta al archivo de salida donde se almacenarán las codificaciones
encodings_path = "encodings.pickle"

# obtener las rutas de las imágenes en el dataset
image_paths = list(paths.list_images(dataset_path))

# inicializar la lista de codificaciones y nombres
known_encodings = []
known_names = []

# recorrer las rutas de las imágenes
for image_path in image_paths:
    # extraer el nombre de la persona desde la ruta
    name = image_path.split(os.path.sep)[-2]

    # cargar la imagen y convertirla de BGR (OpenCV) a RGB (face_recognition)
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detectar las coordenadas de los cuadros delimitadores de los rostros
    boxes = face_recognition.face_locations(rgb, model="hog")

    # computar las codificaciones faciales para los rostros
    encodings = face_recognition.face_encodings(rgb, boxes)

    # agregar cada codificación + nombre a nuestras listas
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

# guardar las codificaciones + nombres en disco
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_path, "wb") as f:
    f.write(pickle.dumps(data))
```

Ejecuta este script para generar el archivo `encodings.pickle`:

```bash
python3 encode_faces.py
```

---

## 🎥 Reconocimiento Facial en Tiempo Real

Crea un script llamado `pi_face_recognition.py` para realizar el reconocimiento facial en tiempo real utilizando la cámara de la Raspberry Pi:

```python
# importaciones necesarias
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2

# ruta al archivo de codificaciones
encodings_path = "encodings.pickle"

# cargar las codificaciones faciales conocidas
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

# inicializar la transmisión de video y permitir que la cámara se caliente
print("[INFO] iniciando transmisión de video...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# recorrer los frames de la transmisión de video
while True:
    # capturar el frame de la transmisión y redimensionarlo a un ancho de 500 píxeles
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # convertir el frame de BGR a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detectar las coordenadas de los cuadros delimitadores de los rostros
    boxes = face_recognition.face_locations(rgb, model="hog")
    # computar las codificaciones faciales para los rostros
    encodings = face_recognition.face_encodings(rgb, boxes)

    # inicializar la lista de nombres para los rostros detectados
    names = []

    # recorrer las codificaciones faciales detectadas
    for encoding in encodings:
        # intentar hacer coincidir cada rostro en el frame con nuestras codificaciones conocidas
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Desconocido"

        # verificar si encontramos una coincidencia
        if True in matches:
            # encontrar los índices de todas las coincidencias y contar el número de veces que cada una ocurre
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determinar 
