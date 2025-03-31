import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pickle
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Actividad: Clasificador de Dígitos con SVM", layout="wide")

# Estilos personalizados
st.markdown(
    """
    <style>
    .big-header {
        font-size: 36px;
        font-weight: bold;
        color: #4B0082;
        text-align: left;
        margin-top: 20px;
    }
    .section-title {
        font-size: 24px;
        color: #006400;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .code-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-left: 4px solid #4B0082;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título principal
st.markdown("<div class='big-header'>Actividad: Clasificador de Dígitos Manuscritos con Streamlit y SVM</div>", unsafe_allow_html=True)
st.write("""
En esta actividad vas a desarrollar una aplicación web con **Streamlit** que permitirá a un usuario:
- Dibujar un número manuscrito.
- Subir una imagen con un número.
- Obtener la predicción de un modelo entrenado mediante **SVM** (Support Vector Machine).

El objetivo es aplicar procesamiento de imágenes y aprendizaje automático en una aplicación interactiva real.
""")

# Cargar el modelo desde el archivo
try:
    with open("svm_digits_model.pkl", "rb") as f:
        modelo = pickle.load(f)
    scaler = modelo["scaler"]
    clf = modelo["clf"]
except FileNotFoundError:
    st.error("No se encontró el archivo 'svm_digits_model.pkl'. Asegúrate de generarlo primero.")
    st.stop()

# Función para preprocesar la imagen
def preprocess_image(image):
    # # Convertir a escala de grises
    img = Image.fromarray(image.astype('uint8')).convert('L')
    
    # #Redimensionar, ya que SVM fue entrenado con imagenes 8x8
    img = img.resize((8, 8))
    # #Necesitamos convertirla en una matriz de números, y que 
    # # sean hexadecimales, no de 0 a 255, porque digits usa eso, por eso escalamos
    img = np.array(img) / 16.0
    # # Aplanar la imagen para que sea un vector de 64 elementos
    image = img.flatten().reshape(1, -1)

    # # Aplicar el mismo scaler que usaste al entrenar
    image = scaler.transform(image)
    

    return image

# Función para preprocesar la imagen del lienzo
def preprocesar_canvas_para_svm(image_data):
    """Preprocesa los datos del lienzo para el modelo SVM."""
    if image_data is None:
        return None
    # El lienzo tiene fondo negro y trazo blanco, así que no invertimos colores
    imagen_scaled = np.array(image_data, dtype=np.uint8)
    imagen_scaled = preprocess_image(image_data)
    return imagen_scaled

# Función para predecir el dígito
def predict(image, invert=True):
    """Realiza la predicción del dígito usando el modelo SVM."""
    processed_image = preprocess_image(image, invert=invert)
    if processed_image is not None:
        prediccion = clf.predict(processed_image)
        return prediccion[0]
    return None

# Interfaz interactiva
st.markdown("<div class='section-title'>Clasificador de Dígitos</div>", unsafe_allow_html=True)

# Sección 1: Dibujar un dígito
st.markdown("### Dibujar un dígito")
st.write("Dibuja un número en el lienzo:")
canvas = st_canvas(
    fill_color="black",      # Fondo negro
    stroke_width=20,         # Grosor del trazo
    stroke_color="white",    # Color del trazo
    background_color="black",# Fondo explícito
    width=200,               # Ancho del lienzo
    height=200,              # Alto del lienzo
    drawing_mode="freedraw", # Modo de dibujo libre
    key="canvas",
    update_streamlit=True,   # Actualización en tiempo real
)

if st.button("Predecir dibujo"):
    if canvas.image_data is not None:
        img_processed = preprocesar_canvas_para_svm(canvas.image_data)
        if img_processed is not None:
            prediction = clf.predict(img_processed)
            st.subheader("Predicción")
            st.write(f"El modelo predice que el número es: **{prediction}**")
            # Mostrar la imagen procesada
            resized_img = cv2.resize(canvas.image_data.astype(np.uint8), (8, 8), interpolation=cv2.INTER_AREA)
            st.image(resized_img, caption="Imagen procesada (8x8)", width=100)
    else:
        st.warning("Por favor, dibuja algo en el lienzo antes de predecir.")

# Sección 2: Subir una imagen
st.markdown("### Subir una imagen")
archivo_subido = st.file_uploader("Sube una imagen manuscrita (JPG o PNG)", type=["jpg", "png"])
if archivo_subido is not None:
    # Mostrar imagen con PIL
    image = Image.open(archivo_subido)
    st.image(image, caption='Imagen subida', width=150)
    st.write("")
    
    # Convertir a array de NumPy y predecir
    image_array = np.array(image)
    # La imagen subida podría tener fondo claro y trazo oscuro, así que invertimos colores
    prediction = predict(image_array, invert=True)
    if prediction is not None:
        st.subheader(f"✅ El modelo predice que el número es: **{prediction}**")
        # Mostrar la imagen procesada
        resized_img = cv2.resize(image_array, (8, 8), interpolation=cv2.INTER_AREA)
        st.image(resized_img, caption="Imagen procesada (8x8)", width=100)

# Pestaña para prueba con el dataset digits
tab1 = st.tabs(["Prueba con dataset"])

with tab1[0]:
    st.write("Prueba con un dígito del dataset `digits` para verificar el modelo:")
    digits = load_digits()
    sample_idx = st.slider("Selecciona un índice del dataset (0-1796):", 0, 1796, 0)
    sample = digits.data[sample_idx].reshape(1, -1)
    true_label = digits.target[sample_idx]
    prediction = clf.predict(sample)
    st.write(f"Valor real: **{true_label}**")
    st.write(f"Predicción: **{prediction}**")
    st.image(digits.images[sample_idx], caption="Imagen del dataset (8x8)", width=100)

# Pie de página
st.markdown("<div class='section-title'>Notas</div>", unsafe_allow_html=True)
st.write("Esta app usa OpenCV para procesar imágenes y Scikit-learn para predecir dígitos manuscritos.")