import streamlit as st

st.markdown("# Página principal 🎈")
st.sidebar.markdown("## Main principal 🎈🎈")

nombre = st.text_input("Escribe tu nombre")
if st.button("Saludar"):
    st.write(f"Hola {nombre} 👋🏼")

st.header("Subir una imagen para predicción")
archivo_subido = st.file_uploader("Sube una imagen con un número manuscrito (JPG, PNG o JPEG)", type=["jpg", "png", "jpeg"])