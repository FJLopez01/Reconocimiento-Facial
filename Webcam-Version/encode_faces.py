# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# obtener las rutas de las imágenes en el conjunto de datos
imagePaths = list(paths.list_images("dataset"))

# inicializar la lista de codificaciones conocidas y nombres conocidos
knownEncodings = []
knownNames = []

# recorrer las rutas de las imágenes
for (i, imagePath) in enumerate(imagePaths):
    # extraer el nombre de la persona del nombre de la carpeta
    name = imagePath.split(os.path.sep)[-2]

    # cargar la imagen y convertirla de BGR (canal de OpenCV) a RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detectar las coordenadas (x, y) de los cuadros delimitadores de cada cara
    boxes = face_recognition.face_locations(rgb, model="hog")

    # calcular las codificaciones faciales para las caras
    encodings = face_recognition.face_encodings(rgb, boxes)

    # recorrer las codificaciones
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# guardar las codificaciones faciales + los nombres en el disco
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))
