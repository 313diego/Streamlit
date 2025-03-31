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

# Configuración inicial de la página con tema personalizado
st.set_page_config(
    page_title="Predicción de Dígitos con SVM",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.streamlit.io/community',
        'Report a bug': None,
        'About': "Aplicación de demostración de SVM para reconocimiento de dígitos"
    }
)

# Estilos CSS personalizados para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #6c5ce7;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: #f8f9fa;
        border-left: 5px solid #6c5ce7;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        color: #6c5ce7;
        font-weight: 600;
    }
    .tab-content {
        padding: 1.5rem;
        border: 1px solid #e9ecef;
        border-radius: 0 0 10px 10px;
        margin-top: -1rem;
        background: white;
    }
    .prediction-result {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        color: white;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #e9ecef;
        font-size: 0.9rem;
    }
    .component-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .key-point {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #6c5ce7;
    }
    .canvas-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .btn-predict {
        background-color: #6c5ce7;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .btn-predict:hover {
        background-color: #5649c0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Función para crear secciones con estilo de tarjeta
def section_card(title, content):
    st.markdown(f"""
    <div class="card">
        <h3>{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Encabezado principal con animación
st.markdown('<h1 class="main-header">🧠 Modelo SVM para Reconocimiento de Dígitos</h1>', unsafe_allow_html=True)

# Introducción mejorada
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div class="component-container">
        <p>Las <span class="highlight">Máquinas de Vectores de Soporte (SVM)</span> son algoritmos de aprendizaje supervisado 
        utilizados principalmente para clasificación y regresión. En esta aplicación, exploramos cómo SVM 
        puede reconocer dígitos manuscritos con alta precisión.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png", 
             caption="Hiperplano separador con margen máximo", width=250)

# Pestañas principales para organizar el contenido
tab_teoria, tab_demo, tab_avanzado = st.tabs(["📚 Teoría SVM", "🎨 Demostración", "📊 Análisis Avanzado"])

with tab_teoria:
    st.markdown('<h2 class="subheader">🧠 Fundamentos de SVM</h2>', unsafe_allow_html=True)
    
    # Idea principal en formato de tarjeta
    section_card("💡 Idea Principal", """
    <p>El objetivo de SVM es <span class="highlight">encontrar el mejor límite (hiperplano)</span> 
    que separe dos clases, maximizando el espacio (margen) entre ellas.</p>
    """)
    
    # Componentes clave con mejor formato
    st.markdown('<h3 class="subheader">🔍 Componentes Clave</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="key-point">
            <h4>Hiperplano</h4>
            <p>Frontera de decisión que separa las clases en el espacio de características.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="key-point">
            <h4>Vectores de Soporte</h4>
            <p>Puntos más cercanos al hiperplano que determinan su posición óptima.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="key-point">
            <h4>Margen</h4>
            <p>Distancia entre el hiperplano y los vectores de soporte, que SVM busca maximizar.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Ventajas con iconos y mejor formato
    st.markdown('<h3 class="subheader">⚙️ Ventajas del modelo SVM</h3>', unsafe_allow_html=True)
    
    ventajas = [
        "✅ Funciona bien en espacios de alta dimensión",
        "✅ Eficaz cuando hay una clara separación entre clases",
        "✅ Usa pocos puntos de datos (vectores de soporte) → eficiente",
        "✅ Robusto contra el sobreajuste en espacios de alta dimensión",
        "✅ Versátil a través de diferentes funciones kernel"
    ]
    
    for ventaja in ventajas:
        st.markdown(f"<div class='key-point'>{ventaja}</div>", unsafe_allow_html=True)
    
    # Aplicación a imágenes con visualización
    st.markdown('<h3 class="subheader">🖼️ Aplicación a Imágenes</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="component-container">
            <p>En problemas como la clasificación de dígitos:</p>
            <ul>
                <li>Cada imagen de 8x8 píxeles se convierte en un vector de 64 características</li>
                <li>SVM encuentra un hiperplano en ese espacio que separe dígitos diferentes</li>
                <li>Para múltiples clases (0-9), SVM entrena varios clasificadores binarios</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # Mostrar un ejemplo de dígito y su representación
        digits = load_digits()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.imshow(digits.images[0], cmap='binary')
        ax1.set_title('Imagen de dígito')
        ax1.axis('off')
        
        # Representación matricial
        ax2.matshow(digits.images[0], cmap='binary')
        ax2.set_title('Matriz 8x8')
        st.pyplot(fig)
    
    # Estrategias multiclase con visualización
    st.markdown('<h3 class="subheader">🎯 Estrategias Multiclase</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="component-container">
            <h4>1️⃣ Uno contra todos (OvR)</h4>
            <ul>
                <li>Crea un modelo para cada número</li>
                <li>Cada modelo aprende: "¿Es un 3 o no lo es?"</li>
                <li>Se hacen 10 modelos (uno por cada dígito)</li>
                <li>El modelo más seguro da la predicción final</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="component-container">
            <h4>2️⃣ Uno contra uno (OvO)</h4>
            <ul>
                <li>Crea un modelo por cada par posible de números</li>
                <li>"¿Es un 2 o un 3?", "¿Es un 7 o un 9?", etc.</li>
                <li>En total se crean 45 modelos para los 10 dígitos</li>
                <li>Todos los modelos votan, y el número con más votos gana</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Tipos de kernel con visualización
    st.markdown('<h3 class="subheader">📊 Tipos de Kernel</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="component-container" style="height: 200px;">
            <h4 style="text-align: center;">Linear</h4>
            <p style="text-align: center;">Línea o plano recto</p>
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <div style="width: 100px; height: 100px; background: linear-gradient(45deg, #e0e0e0 50%, #6c5ce7 50%);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="component-container" style="height: 200px;">
            <h4 style="text-align: center;">Poly</h4>
            <p style="text-align: center;">Curvas polinómicas</p>
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <div style="width: 100px; height: 100px; background: radial-gradient(circle at 30% 70%, #e0e0e0 30%, #6c5ce7 30%);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="component-container" style="height: 200px;">
            <h4 style="text-align: center;">RBF</h4>
            <p style="text-align: center;">Función de base radial</p>
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <div style="width: 100px; height: 100px; background: radial-gradient(circle at center, #e0e0e0 40%, #6c5ce7 40%);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="component-container" style="height: 200px;">
            <h4 style="text-align: center;">Sigmoid</h4>
            <p style="text-align: center;">Similar a red neuronal</p>
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <div style="width: 100px; height: 100px; background: linear-gradient(to right, #e0e0e0 0%, #6c5ce7 100%);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Código de ejemplo con sintaxis resaltada
    st.markdown('<h3 class="subheader">📝 Código de Ejemplo</h3>', unsafe_allow_html=True)
    
    st.code("""
    # Importar librerías necesarias
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Cargar dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    # Entrenamiento del modelo SVM
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)

    # Evaluación del modelo
    accuracy = clf.score(X_test, y_test)
    print(f"Precisión del modelo: {accuracy:.2%}")
    """, language="python")

# Cargar el modelo y el scaler desde el archivo .pkl
@st.cache_resource
def load_model():
    try:
        with open("svm_digits_model.pkl", "rb") as f:
            data = pickle.load(f)
        return data["clf"], data["scaler"]
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo del modelo. Usando un modelo de ejemplo.")
        # Crear un modelo de ejemplo si no se encuentra el archivo
        digits = load_digits()
        X = digits.data
        y = digits.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)
        return clf, scaler

# Función para preprocesar la imagen con visualización mejorada
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

        # Invertir colores si es necesario (asumiendo que los dígitos son claros sobre fondo oscuro)
        if np.mean(img_array) > 8:  # Si la imagen es mayormente clara
            img_array = 16 - img_array  # Invertir

        # Aplanar
        flattened = img_array.flatten().reshape(1, -1)
        
        # Crear visualizaciones para el proceso
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        
        # Imagen original
        if isinstance(image, Image.Image):
            axes[0].imshow(np.array(image.convert("L")), cmap="gray")
        else:
            axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        # Imagen redimensionada
        axes[1].imshow(img_array.reshape(8, 8), cmap="gray")
        axes[1].set_title("Redimensionada (8x8)")
        axes[1].axis("off")
        
        # Representación numérica
        im = axes[2].matshow(img_array.reshape(8, 8), cmap="gray")
        axes[2].set_title("Valores numéricos")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        st.pyplot(fig)
        
        return flattened

    except Exception as e:
        st.error(f"Error en el preprocesamiento: {e}")
        return None

# Pestaña de demostración
with tab_demo:
    st.markdown('<h2 class="subheader">🎨 Clasificador de Dígitos Manuscritos</h2>', unsafe_allow_html=True)
    
    # Cargar el modelo
    clf, scaler = load_model()
    
    # Crear pestañas para los métodos de entrada
    demo_tab1, demo_tab2 = st.tabs(["✏️ Dibujar Dígito", "📷 Subir Imagen"])
    
    with demo_tab1:
        st.markdown("""
        <div class="component-container">
            <p>Dibuja un número del 0 al 9 en el lienzo y haz clic en "Predecir" para ver qué dígito reconoce el modelo SVM.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Contenedor centrado para el canvas
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón de predicción con estilo
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 Predecir", key="predict_draw", use_container_width=True)
        
        if predict_button:
            if canvas_result.image_data is not None:
                with st.spinner("Analizando el dígito..."):
                    img_array = canvas_result.image_data.astype(np.uint8)
                    processed_img = preprocess_image(img_array)
                    
                    if processed_img is not None:
                        prediction = clf.predict(processed_img)[0]
                        
                        # Mostrar predicción con animación
                        st.balloons()
                        st.markdown(f"""
                        <div class="prediction-result">
                            El número predicho es: <span style="font-size: 3rem; font-weight: bold;">{prediction}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mostrar probabilidades para cada clase
                        if hasattr(clf, "predict_proba"):
                            probs = clf.predict_proba(processed_img)[0]
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.bar(range(10), probs, color='#6c5ce7')
                            ax.set_xticks(range(10))
                            ax.set_xticklabels(range(10))
                            ax.set_xlabel('Dígito')
                            ax.set_ylabel('Probabilidad')
                            ax.set_title('Probabilidades por clase')
                            st.pyplot(fig)
                        elif hasattr(clf, "decision_function"):
                            # Para SVM que no tiene predict_proba directamente
                            decisions = clf.decision_function(processed_img)[0]
                            # Normalizar para visualización
                            decisions = (decisions - decisions.min()) / (decisions.max() - decisions.min())
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.bar(range(10), decisions, color='#6c5ce7')
                            ax.set_xticks(range(10))
                            ax.set_xticklabels(range(10))
                            ax.set_xlabel('Dígito')
                            ax.set_ylabel('Puntuación de decisión (normalizada)')
                            ax.set_title('Puntuaciones de decisión por clase')
                            st.pyplot(fig)
            else:
                st.warning("Por favor, dibuja algo antes de predecir.")
    
    with demo_tab2:
        st.markdown("""
        <div class="component-container">
            <p>Sube una imagen de un dígito manuscrito para que el modelo lo reconozca. La imagen debe tener un dígito claro sobre fondo contrastante.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Carga una imagen (JPG, PNG)", type=["jpg", "png"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            
            # Mostrar imagen original
            st.image(img, caption="Imagen original", width=200)
            
            # Botón de predicción
            if st.button("🔍 Predecir", key="predict_upload"):
                with st.spinner("Procesando imagen..."):
                    img_array = np.array(img)
                    processed_img = preprocess_image(img_array)
                    
                    if processed_img is not None:
                        prediction = clf.predict(processed_img)[0]
                        
                        # Mostrar predicción con animación
                        st.balloons()
                        st.markdown(f"""
                        <div class="prediction-result">
                            El número predicho es: <span style="font-size: 3rem; font-weight: bold;">{prediction}</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Sube una imagen para predecir.")

# Pestaña de análisis avanzado
with tab_avanzado:
    st.markdown('<h2 class="subheader">📊 Análisis del Modelo SVM</h2>', unsafe_allow_html=True)
    
    # Sección de entrenamiento interactivo
    st.markdown("""
    <div class="component-container">
        <h3>🔄 Entrenamiento Interactivo</h3>
        <p>Explora cómo diferentes parámetros afectan el rendimiento del modelo SVM en el dataset de dígitos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parámetros ajustables
    col1, col2 = st.columns(2)
    with col1:
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
        C = st.slider("Parámetro C (regularización)", 0.1, 10.0, 1.0, 0.1)
    with col2:
        gamma = st.selectbox("Gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1.0], index=0)
        test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.2, 0.05)
    
    # Botón para entrenar
    if st.button("🚀 Entrenar y Evaluar Modelo", use_container_width=True):
        with st.spinner("Entrenando modelo SVM..."):
            # Cargar datos
            digits = load_digits()
            X = digits.data
            y = digits.target
            
            # Escalar características
            scaler_example = StandardScaler()
            X_scaled = scaler_example.fit_transform(X)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            
            # Convertir gamma a float si es necesario
            if gamma not in ["scale", "auto"]:
                gamma = float(gamma)
            
            # Entrenar modelo
            clf_example = SVC(kernel=kernel, C=C, gamma=gamma)
            clf_example.fit(X_train, y_train)
            
            # Evaluar modelo
            train_accuracy = clf_example.score(X_train, y_train)
            test_accuracy = clf_example.score(X_test, y_test)
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <h3>Precisión en Entrenamiento</h3>
                    <p style="font-size: 2rem; color: #6c5ce7;">{train_accuracy:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <h3>Precisión en Prueba</h3>
                    <p style="font-size: 2rem; color: #6c5ce7;">{test_accuracy:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizar ejemplos del dataset
            st.markdown('<h3 class="subheader">📊 Ejemplos del Dataset</h3>', unsafe_allow_html=True)
            
            # Seleccionar algunos ejemplos aleatorios
            indices = np.random.choice(len(X_test), 10, replace=False)
            
            # Crear figura con ejemplos
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            for i, ax in enumerate(axes.flat):
                idx = indices[i]
                ax.imshow(digits.images[idx], cmap="gray")
                
                # Predecir y mostrar resultado
                pred = clf_example.predict([X_test[idx]])[0]
                true = y_test[idx]
                
                # Color verde si es correcto, rojo si es incorrecto
                color = "green" if pred == true else "red"
                ax.set_title(f"Pred: {pred}, Real: {true}", color=color)
                ax.axis("off")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Matriz de confusión
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            
            y_pred = clf_example.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
            disp.plot(cmap="Blues", values_format="d", ax=ax)
            plt.title("Matriz de Confusión")
            st.pyplot(fig)

# Pie de página mejorado
st.markdown("""
<div class="footer">
    <p>Desarrollado con ❤️ y Streamlit | Modelo SVM para reconocimiento de dígitos</p>
    <p>Versión 2.0 - Interfaz mejorada</p>
</div>
""", unsafe_allow_html=True)
