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
::contentReference[oaicite:0]{index=0}
 
