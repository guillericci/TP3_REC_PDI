import cv2
import numpy as np
import matplotlib.pyplot as plt
def show(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def deteccion(frame):
    """
    Toma un frame BGR, detecta los dados, dibuja rectángulos y círculos, y devuelve el frame con el numero reconocido
    """
    mostrar_frame = frame.copy()

    blur = cv2.GaussianBlur(frame, (5,5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #show(h,"canal h")
    #show(s,"canal s")

    # umbral en S: quiero todo lo con baja saturación (blancos/grises)
    _, mask_s = cv2.threshold(s, 130, 255, cv2.THRESH_BINARY_INV)
    # umbral en H: descarto tonos oscuros
    lower_h, upper_h = 0, 180
    mask_h = cv2.inRange(h, lower_h, upper_h) 
    #máscara combinada
    mask = cv2.bitwise_and(mask_s, mask_h)
    #show(mask)

    # contornos cuadrados 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        aspecto = w/float(h)
        # filtra por tamaño y proporción aproximada de cuadrado
        if 2000 < area < 10000 and 0.7 < aspecto < 1.3:
            # Dibuja rectángulo del dado
            cv2.rectangle(mostrar_frame, (x,y), (x+w,y+h), (255,0,0), 2)

            # creo roi de mask_s y binariza 
            roi_dado = mask_s[y:y+h, x:x+w]
            _, roi_bin = cv2.threshold(roi_dado, 100, 255, cv2.THRESH_BINARY_INV)

            # deteccion de circulos
            blur2 = cv2.GaussianBlur(roi_bin, (9,9), 2)
            circles = cv2.HoughCircles(blur2, cv2.HOUGH_GRADIENT,dp=1, minDist=5, param1=150, param2=13, minRadius=2, maxRadius=7)
            #cuenta y dibuja circulos
            count = 0
            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                count = len(circles)
                for (cx,cy,r) in circles:
                    cv2.circle(mostrar_frame, (x+cx, y+cy), r, (255,0,0), 2)

            text = str(count)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = w / 100.0
            th = 2
            (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
            tx = x + (w-tw)//2
            ty = y - 10 if y>20 else y + tht + 10
            cv2.putText(mostrar_frame, text, (tx,ty), font, fs, (255,0,0), th, cv2.LINE_AA)

    return mostrar_frame

def procesar_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #Bucle principal 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #frame = cv2.resize(annotated, (width//3, height//3))

        # Procesa el frame y anota
        anota = deteccion(frame)
        
        # Escribe al vídeo de salida
        out.write(anota)

        ventana = cv2.resize(anota, (width//3, height//3))
        cv2.imshow('Deteccion en vivo', ventana)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


ruta = r"D:\TUIA\PROCESAMIENTO IMAGEN\TUIA_PDI_TP3_2025_C1_REC"

ruta_video_1 = f"{ruta}/tirada_1.mp4"
ruta_video_2 = f"{ruta}/tirada_2.mp4"
ruta_video_3 = f"{ruta}/tirada_3.mp4"
ruta_video_4 = f"{ruta}/tirada_4.mp4"

procesar_video(f"{ruta}/tirada_1.mp4", f"{ruta}/Output_tirada_1.mp4")
procesar_video(f"{ruta}/tirada_2.mp4", f"{ruta}/Output_tirada_2.mp4")
procesar_video(f"{ruta}/tirada_3.mp4", f"{ruta}/Output_tirada_3.mp4")
procesar_video(f"{ruta}/tirada_4.mp4", f"{ruta}/Output_tirada_4.mp4")