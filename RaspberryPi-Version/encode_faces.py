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
