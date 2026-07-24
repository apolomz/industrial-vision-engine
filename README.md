# 🚀 Industrial Gesture & Biometric Vision Engine

Suite industrial orientada a objetos (POO) en Python para auditoría de calidad de imagen, extracción de mapas de características y reconocimiento biométrico gestual en tiempo real. Diseñado para entornos sin contacto de alta precisión (laboratorios biomédicos y plantas industriales).

---

## 🏗️ Arquitectura del Sistema

El proyecto está diseñado bajo principios de modularidad y bajo acoplamiento:

* **`VisualFeatureExtractor`**: Encargado del procesamiento matricial con **OpenCV** y **NumPy**. Realiza análisis de iluminación en espacio de color HSV y extracción de bordes estructurales mediante filtrado Gaussiano y detector Canny.
* **`MediaPipeGestureEngine`**: Motor de inferencia en Deep Learning que procesa *keypoints* biométricos en 3D. Evalúa distancias euclidianas relativas entre puntos anatómicos (pulgar e índice) para la clasificación de gestos (`PINCH_CLICK`).
* **`VisionInspectorPipeline`**: Orchestrator central que sincroniza la captura de métricas, mide la latencia de procesamiento y genera reportes consolidados en formato JSON.

---

## 🛠️ Requisitos Previos e Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/industrial-vision-engine.git](https://github.com/TU_USUARIO/industrial-vision-engine.git)
   cd industrial-vision-engine