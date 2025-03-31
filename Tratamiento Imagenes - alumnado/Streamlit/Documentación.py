import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from PIL import Image

# Configuración inicial de la página
st.set_page_config(
    page_title="Inicio - Predicción de Dígitos",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Sección de documentación ajustada al formato exacto
st.title("Modelo SVM (Support Vector Machine)")
st.write("Las **Máquinas de Vectores de Soporte (SVM)** son modelos de aprendizaje supervisado utilizados principalmente para clasificación, aunque también pueden aplicarse a regresión.")

st.subheader("🧠 Idea Principal")
st.write("El objetivo de SVM es **encontrar el mejor límite (frontera)** que separe dos clases, maximizando el espacio (margen) entre ellas.")
st.image("https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png", 
         caption="El hiperplano separa las dos clases con el mayor margen posible.", width=400)

st.subheader("🔍 Componentes Clave")
st.write("""
- **Hiperplano**: Es la frontera de decisión que separa las clases.  
- **Vectores de soporte**: Son los puntos más cercanos al hiperplano. Solo estos puntos afectan directamente a la posición del hiperplano.  
- **Margen**: Es la distancia entre el hiperplano y los vectores de soporte. SVM maximiza este margen.
""")

st.subheader("⚙️ Ventajas del modelo SVM")
st.write("""
- ✅ Funciona bien en espacios de alta dimensión.  
- ✅ Eficaz cuando hay una clara separación entre clases.  
- ✅ Usa pocos puntos de datos (vectores de soporte) → eficiente.
""")

st.subheader("🧠 ¿Cómo se aplica SVM a imágenes?")
st.write("""
En problemas como la clasificación de dígitos (por ejemplo, el dataset de dígitos de Scikit-learn):  

- Cada imagen de 8x8 píxeles se convierte en un vector de 64 características.  
- SVM trata de encontrar un hiperplano en ese espacio que separe dígitos diferentes (por ejemplo, 3s de 5s).  
- Para múltiples clases (0 a 9), SVM entrena varios clasificadores binarios (uno contra uno o uno contra todos).  

**Vector de 64 características:**  
[[ 0.  0.  5. 13.  9.  1.  0.  0.]

[ 0.  0. 13. 15. 10. 15.  5.  0.]

[ 0.  3. 15.  2.  0. 11.  8.  0.]

[ 0.  4. 12.  0.  0.  8.  8.  0.]

[ 0.  5.  8.  0.  0.  9.  8.  0.]

[ 0.  4. 11.  0.  1. 12.  7.  0.]

[ 0.  2. 14.  5. 10. 12.  0.  0.]

[ 0.  0.  6. 13. 10.  0.  0.  0.]]

text

Contraer

Ajuste

Copiar

Cada número representa la intensidad del píxel, pero el rango está entre 0 y 16.
""")

st.subheader("🎯 ¿Cómo clasifica SVM varios dígitos?")
st.write("""
SVM, por defecto, **solo sabe distinguir entre dos clases** (por ejemplo, "¿es un 3 o no lo es?").  

Pero cuando queremos clasificar **dígitos del 0 al 9**, tenemos **10 clases distintas**.  

Para ello, entrena varios clasificadores.  
SVM entrena **varios modelos pequeños** que comparan solo **dos números a la vez**. Esto se llama:  

---

###### 1️⃣ Uno contra todos (OvR)  
- Crea un modelo para cada número.  
- Cada modelo aprende: **"¿Es un 3 o no lo es?"**, **"¿Es un 5 o no lo es?"**, etc.  
- Se hacen 10 modelos (uno por cada dígito).  
- El modelo que esté más seguro es el que da la predicción final.  

###### 2️⃣ Uno contra uno (OvO)  
- Crea un modelo por **cada par posible de números**:  
  "¿Es un 2 o un 3?", "¿Es un 7 o un 9?", etc.  
- En total se crean **45 modelos** para los 10 dígitos.  
- Todos los modelos votan, y el número con más votos gana.  

###### ⚙️ Scikit-learn (la librería que estamos usando) **usa por defecto la estrategia "uno contra uno"**.
""")

st.subheader("📝 Ejemplo práctico")
st.code("""
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar dataset
digits = load_digits()
X = digits.data
y = digits.target

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Entrenamiento del modelo SVM
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)
""", language="python")

st.subheader("📊 Tipos de kernel")
st.write("""
SVM puede usar diferentes funciones para separar los datos:  

- `linear`: Línea o plano recto.  
- `poly`: Polinómico (curvas).  
- `rbf`: Radial Basis Function (muy común para separaciones complejas).  
- `sigmoid`: Similar a una red neuronal.
""")

# Cargar el modelo y el scaler desde el archivo .pkl
@st.cache_resource
def load_model():
    with open("svm_digits_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["clf"], data["scaler"]

clf, scaler = load_model()

# Función para preprocesar la imagen
def preprocess_image(image):
    try:
        # Convertir a imagen de Pillow si no lo es
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convertir a escala de grises
        image = image.convert("L")

        # Redimensionar a 8x8 con antialiasing
        image = image.resize((8, 8), Image.Resampling.LANCZOS)

        # Convertir a array numpy y normalizar a [0, 16]
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array / img_array.max()) * 16  # Normalización sin alterar distribución

        # Aplanar
        flattened = img_array.flatten().reshape(1, -1)

        # Mostrar valores preprocesados
        st.write("Imagen preprocesada antes de predecir:", flattened)

        return flattened  # Sin aplicar StandardScaler

    except Exception as e:
        st.error(f"Error en el preprocesamiento: {e}")
        return None

# Interfaz interactiva
st.header("🎨 Clasificador de Dígitos Manuscritos")
st.markdown("Dibuja un número o sube una imagen para predecirlo con SVM.")

tab1, tab2 = st.tabs(["Dibujar", "Subir Imagen"])

with tab1:
    st.subheader("Dibuja un número")
    canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Fondo transparente (sin rellenar)
    stroke_width=20,
    stroke_color="white",  # Trazo blanco para que se vea
    background_color="black",  # Fondo negro
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
    )
    
    if st.button("Predecir dibujo"):
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype(np.uint8)
            processed_img = preprocess_image(img_array)
            if processed_img is not None:
                prediction = clf.predict(processed_img)[0]
                st.success(f"El número predicho es: **{prediction}**")
                resized_img = cv2.resize(img_array, (8, 8), interpolation=cv2.INTER_AREA)
                st.image(resized_img, caption="Imagen procesada (8x8)", width=100)
        else:
            st.warning("Por favor, dibuja algo antes de predecir.")

with tab2:
    st.subheader("Sube una imagen")
    uploaded_file = st.file_uploader("Carga una imagen (JPG, PNG)", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        processed_img = preprocess_image(img_array)
        if processed_img is not None:
            prediction = clf.predict(processed_img)[0]
            st.success(f"El número predicho es: **{prediction}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Imagen original", width=200)
            with col2:
                resized_img = cv2.resize(img_array, (8, 8), interpolation=cv2.INTER_AREA)
                st.image(resized_img, caption="Imagen procesada (8x8)", width=100)
    else:
        st.info("Sube una imagen para predecir.")

# Ejemplo práctico interactivo
st.header("📝 Ejemplo práctico: Entrenamiento del modelo")
st.markdown("Aquí puedes ver cómo se entrenó el modelo SVM con el dataset de dígitos de Scikit-learn.")
if st.button("Ejecutar entrenamiento y mostrar resultados"):
    digits = load_digits()
    X = digits.data
    y = digits.target
    scaler_example = StandardScaler()
    X_scaled = scaler_example.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf_example = SVC(kernel="linear")
    clf_example.fit(X_train, y_train)
    accuracy = clf_example.score(X_test, y_test)
    st.success(f"Precisión del modelo entrenado: **{accuracy:.2%}**")
    st.subheader("Ejemplos del dataset")
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(f"Dígito: {digits.target[i]}")
        ax.axis("off")
    st.pyplot(fig)

# Pie de página
st.markdown("---")
st.write("Hecho con ❤️ por Diego para la actividad de SVM.")