import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuración inicial de la página
st.set_page_config(
    page_title="Clasificador de Dígitos - SVM",
    page_icon="🔢",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Aplicar CSS personalizado para colores e iconos
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #FF6F61 0%, #6B728E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #FF6F61;
        border-bottom: 2px solid #6B728E;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    .concept-box {
        background-color: #FFF3E2;
        border-left: 4px solid #FF6F61;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E6E6FA;
        border: 1px solid #6B728E;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .canvas-container {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .prediction-result {
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D4F1F4;
        border: 2px solid #75E6DA;
    }
    .button-custom {
        background-color: #FF6F61;
        color: white;
        font-weight: bold;
        border-radius: 0.3rem;
    }
    .success-box {
        background-color: #D4F1F4;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.markdown('<h1 style="color: #FF6F61; text-align: center;">🧠 SVM Toolkit</h1>', unsafe_allow_html=True)
navigation = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "🎨 Clasificador", "📝 Entrenar Modelo"]
)

# Función para preprocesar la imagen
def preprocess_image(image):
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("L")
        image = image.resize((8, 8), Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array / img_array.max()) * 16
        flattened = img_array.flatten().reshape(1, -1)
        return flattened, img_array
    except Exception as e:
        st.error(f"Error en el preprocesamiento: {e}")
        return None, None

# Cargar o entrenar modelo
@st.cache_resource
def load_or_train_model():
    try:
        with open("svm_digits_model.pkl", "rb") as f:
            data = pickle.load(f)
        return data["clf"], data["scaler"]
    except:
        digits = load_digits()
        X = digits.data
        y = digits.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)
        return clf, scaler

clf, scaler = load_or_train_model()

# Página de inicio
if navigation == "🏠 Inicio":
    st.markdown('<h1 class="main-header">Bienvenido al Clasificador SVM</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">🔢 Usa esta app para dibujar dígitos, clasificarlos con SVM o entrenar tu propio modelo.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>¿Qué es SVM?</h3>
        <p>Un algoritmo que encuentra la mejor línea o plano para separar clases, maximizando el margen entre ellas.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png", caption="Hiperplano SVM", width=200)

# Página del clasificador
elif navigation == "🎨 Clasificador":
    st.markdown('<h1 class="main-header">🎨 Clasificador de Dígitos</h1>', unsafe_allow_html=True)
    st.markdown('<div class="concept-box">✏️ Dibuja un número o sube una imagen para clasificar.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✏️ Dibujar", "📷 Subir Imagen"])
    
    with tab1:
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=300,
            height=300,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_btn = st.button("🔍 Predecir", key="predict", help="Clasifica tu dibujo")
        with col_btn2:
            if st.button("🧹 Limpiar", key="clear", help="Borra el lienzo"):
                st.experimental_rerun()

        if predict_btn and canvas_result.image_data is not None:
            with st.spinner("🔄 Analizando..."):
                img_array = canvas_result.image_data.astype(np.uint8)
                processed_img, img_8x8 = preprocess_image(img_array)
                if processed_img is not None:
                    prediction = clf.predict(processed_img)[0]
                    st.markdown(f'<div class="prediction-result">Predicción: <b>{prediction}</b></div>', unsafe_allow_html=True)
                    st.image(cv2.resize(img_8x8, (100, 100), interpolation=cv2.INTER_NEAREST), caption="Imagen 8x8", width=100)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        uploaded_file = st.file_uploader("📁 Sube una imagen", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            processed_img, img_8x8 = preprocess_image(img)
            if processed_img is not None:
                prediction = clf.predict(processed_img)[0]
                st.markdown(f'<div class="prediction-result">Predicción: <b>{prediction}</b></div>', unsafe_allow_html=True)
                st.image(cv2.resize(img_8x8, (100, 100), interpolation=cv2.INTER_NEAREST), caption="Imagen 8x8", width=100)

# Página de entrenamiento
elif navigation == "📝 Entrenar Modelo":
    st.markdown('<h1 class="main-header">📝 Entrenar Modelo SVM</h1>', unsafe_allow_html=True)
    st.markdown('<div class="concept-box">🚀 Entrena un modelo SVM para clasificar dígitos.</div>', unsafe_allow_html=True)

    if st.button("🚀 Entrenar", key="train_btn", help="Inicia el entrenamiento"):
        with st.spinner("⏳ Preparando datos..."):
            digits = load_digits()
            X = digits.data
            y = digits.target
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        with st.spinner("🔧 Entrenando modelo..."):
            clf = SVC(kernel="linear")
            clf.fit(X_train, y_train)
        
        with st.spinner("📊 Evaluando..."):
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        with open("svm_digits_model.pkl", "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler}, f)
        
        st.markdown(f'<div class="success-box">✅ Modelo entrenado con precisión: <b>{accuracy:.2f}</b></div>', unsafe_allow_html=True)
        st.balloons()