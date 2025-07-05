# Trabajo Práctico N°3 – Deteccion de Dados

Este repositorio contiene la solución al **Trabajo Práctico N°3** de la materia Procesamiento de Imágenes I, de la Tecnicatura Universitaria en IA - UNR. El objetivo es procesar secuencias de video de 5 dados en movimiento, detectar cuando se detienen y leer el número de puntos en cada dado, generando videos anotados.

---

## Contenido del repositorio

* **TP3\_REC.py**: script principal en Python que implementa la detección y anotación de los dados.
* **inputs**: carpeta con los videos de las tiradas de 5 dados.
* **outputs**: carpeta con videos de salida generados, con bounding box azul y número reconocido.
* **procesos**: carpeta con ejemplos de imagenes de depuracion  
* **README.md**: este archivo.

---

## Requisitos

* Python 3.7 o superior
* OpenCV (`opencv-python`)
* NumPy
* Matplotlib

Instalalos con:

```bash
pip install opencv-python numpy matplotlib
```

---

## Uso

1. Clonar o descargar el repositorio.
2. Modificar la ruta base en la sección de variables de `TP3_REC.py`:

   ```python
   ruta = r"/ruta/a/TUIA_PDI_TP3_REC"
   ```
3. Ejecutar el script:

   ```bash
   python TP3_REC.py
   ```
4. El script procesará cada video (`tirada_1.mp4` a `tirada_4.mp4`) y generará los archivos de salida `Output_tirada_*.mp4` en la misma carpeta.
5. Durante la ejecución se mostrará una ventana en tiempo real con la detección. Presionar `q` para detener antes de finalizar.

---

## Estructura de funciones

* `show(img, ...)` – Muestra imágenes con Matplotlib (útil para depuración).
* `deteccion(frame)` – Procesa un frame BGR, detecta dados, dibuja bounding box y número, devuelve frame anotado.
* `procesar_video(video_path, output_path)` – Recorre todos los frames de un video, aplica `deteccion` y guarda el resultado.

---


