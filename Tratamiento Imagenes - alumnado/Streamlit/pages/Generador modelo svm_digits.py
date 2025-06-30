from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# Título de la aplicación con color
st.markdown("<h1 style='color: #1E90FF;'>Entrenamiento de un Modelo SVM para Dígitos</h1>", unsafe_allow_html=True)

# Descripción
st.write("Este programa entrena un modelo SVM para clasificar imágenes de dígitos (0-9). Presiona el botón para comenzar.")

# Botón para iniciar el entrenamiento
if st.button("Entrenar Modelo"):
    # 1. Cargar el dataset de dígitos (8x8 imágenes)
    with st.spinner("Cargando el conjunto de datos..."):
        digits = load_digits()
        X = digits.data  # Datos: (n_samples, 64)
        y = digits.target  # Etiquetas: (n_samples,)
        st.success("¡Conjunto de datos cargado exitosamente!")

    # 2. Escalar los datos
    with st.spinner("Escalando los datos..."):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.success("¡Datos escalados correctamente!")

    # 3. Dividir en entrenamiento y prueba
    with st.spinner("Dividiendo los datos en entrenamiento y prueba..."):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        st.success("¡Datos divididos!")

    # 4. Crear y entrenar el modelo SVM
    with st.spinner("Entrenando el modelo SVM..."):
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)
        st.success("¡Modelo entrenado!")

    # 5. Evaluar el modelo
    with st.spinner("Evaluando el modelo..."):
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Modelo evaluado. Precisión: {accuracy:.2f}")

    # Guardar el modelo y el scaler juntos en un diccionario
    modelo = {
        "scaler": scaler,
        "clf": clf
    }

    # Serializar con pickle
    with st.spinner("Guardando el modelo..."):
        with open("svm_digits_model.pkl", "wb") as f:
            pickle.dump(modelo, f)
        st.success("¡Modelo guardado como 'svm_digits_model.pkl'!")

    # Mensaje final con animación de globos
    st.write("¡El entrenamiento ha finalizado con éxito!")
    st.balloons()

# Estilos CSS opcionales
st.markdown("""
    <style>
    .success {
        color: green;
        font-weight: bold;
    }
    .spinner {
        color: blue;
    }
    </style>
""", unsafe_allow_html=True)