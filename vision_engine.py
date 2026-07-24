"""
Industrial & Biometric Vision Engine (InspecVision AI)
------------------------------------------------------
Módulo unificado de Visión Artificial que combina análisis matricial en OpenCV/NumPy
e Inferencia Inteligente con MediaPipe para auditoría industrial e interfaz gestual.

Desarrollado como proyecto de integración de habilidades técnicas en Visión por Computador.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import time
from typing import Dict, Any, Tuple


class VisualFeatureExtractor:
    """Módulo para análisis matricial clásico y extracción de mapa de características."""

    @staticmethod
    def auditar_iluminacion(imagen_bgr: np.ndarray) -> Tuple[bool, float]:
        """Calcula el brillo promedio en el canal V de HSV."""
        hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        brillo_promedio = float(np.mean(hsv[:, :, 2]))
        es_valida = 40.0 <= brillo_promedio <= 220.0
        return es_valida, brillo_promedio

    @staticmethod
    def generar_mapa_bordes_canny(imagen_bgr: np.ndarray) -> np.ndarray:
        """Pipeline de suavizado Gaussiano y extracción Canny."""
        gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(suave, 50, 150)
        return bordes


class MediaPipeGestureEngine:
    """Motor de inferencia gestual para análisis biométrico de manos."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence
        )

    def evaluar_gesto_pellizco(self, imagen_bgr: np.ndarray) -> Dict[str, Any]:
        """Detecta manos y calcula la distancia métrica entre pulgar e índice."""
        img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(img_rgb)

        if not resultados.multi_hand_landmarks:
            return {
                "mano_detectada": False,
                "gesto": "NINGUNO",
                "distancia_normalizada": None
            }

        # Tomamos la primera mano detectada
        landmarks = resultados.multi_hand_landmarks[0].landmark
        pulgar = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        indice = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])

        # Distancia euclidiana tridimensional
        distancia = float(np.linalg.norm(pulgar - indice))
        gesto_activo = "PINCH_CLICK" if distancia < 0.06 else "MANO_ABIERTA"

        return {
            "mano_detectada": True,
            "gesto": gesto_activo,
            "distancia_normalizada": round(distancia, 4)
        }

    def cerrar(self):
        """Libera los recursos del modelo MediaPipe."""
        self.hands.close()


class VisionInspectorPipeline:
    """Pipeline central orchestrator de inspección y exportación de datos."""

    def __init__(self, ID_operador: str):
        self.id_operador = ID_operador
        self.extractor = VisualFeatureExtractor()
        self.gestor = MediaPipeGestureEngine()

    def procesar_fotograma(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Ejecuta la auditoría integral y compila métricas ejecutivas."""
        inicio_tiempo = time.time()

        # 1. Auditoría de Calidad
        calidad_ok, brillo = self.extractor.auditar_iluminacion(frame_bgr)

        # 2. Análisis Gestual
        resultado_gesto = self.gestor.evaluar_gesto_pellizco(frame_bgr)

        # 3. Métricas del Mapa de Características
        bordes = self.extractor.generar_mapa_bordes_canny(frame_bgr)
        densidad_bordes = float(np.count_nonzero(bordes) / bordes.size) * 100.0

        latencia_ms = (time.time() - inicio_tiempo) * 1000.0

        reporte = {
            "metadata": {
                "operador": self.id_operador,
                "timestamp": time.ctime(),
                "latencia_procesamiento_ms": round(latencia_ms, 2)
            },
            "auditoria_imagen": {
                "brillo_promedio": round(brillo, 2),
                "estado_iluminacion": "APROBADO" if calidad_ok else "RECHAZADO"
            },
            "telemetria_biometrica": resultado_gesto,
            "analisis_estructural": {
                "densidad_bordes_porcentaje": round(densidad_bordes, 2)
            }
        }

        return reporte


# --- Ejecución del Pipeline ---
if __name__ == "__main__":
    print("⚡ Inicializando Industrial & Biometric Vision Engine...")

    # 1. Crear un marco sintético de pruebas (simulación de cámara)
    marco_sintetico = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(marco_sintetico, (100, 100), (300, 300), (200, 200, 200), -1)

    # 2. Instanciar Pipeline
    inspector = VisionInspectorPipeline(ID_operador="Ing_Sanchez_Univalle")

    # 3. Procesar datos
    reporte_final = inspector.procesar_fotograma(marco_sintetico)

    # 4. Guardar JSON estructurado para el portafolio
    nombre_archivo_json = "reporte_vision_industrial.json"
    with open(nombre_archivo_json, "w", encoding="utf-8") as f:
        json.dump(reporte_final, f, indent=4)

    print(f"✅ Inspección finalizada con éxito. Reporte generado en '{nombre_archivo_json}'.")
    print(json.dumps(reporte_final, indent=4))

    inspector.gestor.cerrar()