# Reconocimiento Facial con OpenCV, Deep Learning y Python

Este repositorio contiene dos implementaciones de reconocimiento facial basadas en **OpenCV, dlib y face_recognition**:

1. **Versión para Raspberry Pi**: Diseñada para correr en hardware de bajo consumo como **Raspberry Pi**.
2. **Versión para Webcam en PC**: Implementación en un entorno de escritorio con una cámara web.

---

## 📂 Estructura del Proyecto

### 📁 RaspberryPi-Version/
- `encode_faces.py` → Codifica los rostros en imágenes.
- `pi_face_recognition.py` → Reconocimiento facial en Raspberry Pi.
- `encodings.pickle` → Archivo con codificaciones de rostros generadas.
- `haarcascade_frontalface_default.xml` → Modelo Haar Cascade para detección facial.
- `dataset/` → Contiene imágenes de entrenamiento.
- `Lab_Reconocimiento_Facial.pdf` → Documentación del proyecto.

### 📁 Webcam-Version/
- `encode_faces.py` → Codifica los rostros en imágenes.
- `face_recognition_webcam.py` → Reconocimiento facial en PC con webcam.
- `encodings_webcam.pickle` → Archivo con codificaciones de rostros generadas.
- `dataset_webcam/` → Contiene imágenes de entrenamiento.
- `Reconocimiento_Facial_OpenCV.pdf` → Documentación del proyecto.

---

## ⚙️ Instalación y Configuración

### 🔹 Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Reconocimiento-Facial.git
cd Reconocimiento-Facial
```

### 🔹 Instalar dependencias
```bash
pip install opencv-python dlib face_recognition imutils
```

### 🔹 Ejecutar el reconocimiento facial

✅ **Para Raspberry Pi:**
```bash
python RaspberryPi-Version/pi_face_recognition.py
```

✅ **Para Webcam en PC:**
```bash
python Webcam-Version/face_recognition_webcam.py
```

---

## 📝 Notas Adicionales

- Asegúrate de tener una cámara conectada si usas la versión para **PC/Webcam**.
- Para Raspberry Pi, se recomienda usar una **Raspberry Pi 3 o superior** para mejor rendimiento.
- Puedes mejorar la precisión del modelo usando un dataset más grande y optimizando las codificaciones faciales.

---

## 📌 Contribuciones
Si deseas contribuir a este proyecto, siéntete libre de hacer un fork y enviar un pull request con mejoras o correcciones.

---

## 📜 Licencia
Este proyecto se encuentra bajo la licencia MIT. Puedes usarlo y modificarlo libremente.

---

¡Gracias por visitar este repositorio! 😊

