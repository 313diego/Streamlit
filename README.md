# Clasificador de Dígitos Manuscritos con Streamlit y SVM

## Descripción
Este proyecto consiste en una aplicación web interactiva desarrollada con **Streamlit** que clasifica dígitos manuscritos utilizando un modelo de **Support Vector Machine (SVM)** entrenado con el dataset `digits` de Scikit-learn. Los usuarios pueden dibujar números en un lienzo o subir imágenes JPG/PNG para obtener predicciones en tiempo real.

## Tecnologías utilizadas
- **Lenguajes**: Python
- **Bibliotecas**: Streamlit, OpenCV, Scikit-learn, Pillow, NumPy, Matplotlib, streamlit-drawable-canvas
- **Herramientas**: GitHub, Streamlit Community Cloud

## Funcionalidades
- Dibujar dígitos en un lienzo interactivo.
- Subir imágenes de dígitos manuscritos.
- Preprocesamiento de imágenes (escala de grises, redimensionado a 8x8, normalización).
- Predicción con un modelo SVM preentrenado.
- Visualización de resultados y comparación con el dataset original.

## Resultados
La aplicación está desplegada en [Streamlit Community Cloud](https://tratamientoimagenesdiego.streamlit.app) y el código fuente está disponible en mi repositorio de GitHub. El modelo alcanza una precisión notable gracias al preprocesamiento y la optimización del SVM.

## Lecciones aprendidas
- Manejo de imágenes con OpenCV y Pillow.
- Despliegue de aplicaciones web con Streamlit.
- Solución de errores en entornos remotos (como normalización de imágenes y manejo de dependencias).

## Enlaces
- [Aplicación en vivo](https://tratamientoimagenesdiego.streamlit.app)
- [Código en GitHub](https://github.com/313diego/TratamientoImagenes)