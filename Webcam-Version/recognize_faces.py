# import the necessary packages
import face_recognition
import imutils
import pickle
import cv2

# cargar las codificaciones faciales conocidas
data = pickle.loads(open("encodings.pickle", "rb").read())

# inicializar la cámara web
video_capture = cv2.VideoCapture(0)

while True:
    # capturar un solo frame de video
    ret, frame = video_capture.read()

    # redimensionar el frame para un procesamiento más rápido y convertirlo a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb_frame.shape[1])

    # detectar las coordenadas (x, y) de los cuadros delimitadores de cada cara
    boxes = face_recognition.face_locations(rgb_frame, model="hog")

    # calcular las codificaciones faciales para las caras
    encodings = face_recognition.face_encodings(rgb_frame, boxes)

    # inicializar la lista de nombres para las caras detectadas
    names = []

    # recorrer las codificaciones faciales
    for encoding in encodings:
        # intentar hacer coincidir cada cara en el frame con las codificaciones conocidas
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Desconocido"

        # verificar si se encontró una coincidencia
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # contar el número de veces que cada nombre aparece en las coincidencias
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determinar el nombre con el mayor número de coincidencias
            name = max(counts, key=counts.get)

        # agregar el nombre a la lista de nombres
        names.append(name)

    # mostrar los resultados
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # ajustar las coordenadas del cuadro delimitador al tamaño original del frame
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # dibujar el cuadro delimitador y el nombre de la persona
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # mostrar el frame resultante
    cv2.imshow("Video", frame)

    # salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar la cámara web y cerrar las ventanas
video_capture.release()
cv2.destroyAllWindows()
