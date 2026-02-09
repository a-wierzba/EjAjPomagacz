import cv2
import numpy as np
from openvino.runtime import Core
import os
import math

# --- KONFIGURACJA ---
MODEL_DIR = "intel"

# Ścieżki do modeli
FACE_XML = f"{MODEL_DIR}/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
FACE_BIN = f"{MODEL_DIR}/face-detection-retail-0004/FP32/face-detection-retail-0004.bin"

EMOTION_XML = f"{MODEL_DIR}/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"
EMOTION_BIN = f"{MODEL_DIR}/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin"

HEAD_POSE_XML = f"{MODEL_DIR}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
HEAD_POSE_BIN = f"{MODEL_DIR}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin"

EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']

# --- FUNKCJA RYSUJĄCA OSIE (MATEMATYKA 3D) ---
def draw_axes(frame, center_x, center_y, yaw, pitch, roll, scale=50):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0

    cx = int(center_x)
    cy = int(center_y)

    # Obliczenia trygonometryczne dla końcówek linii
    Rx = np.array([[1, 0, 0], 
                   [0, math.cos(pitch), -math.sin(pitch)], 
                   [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)], 
                   [0, 1, 0], 
                   [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0], 
                   [math.sin(roll), math.cos(roll), 0], 
                   [0, 0, 1]])

    # Macierz rotacji
    R = Rz @ Ry @ Rx

    # Oś X (Czerwona) - "Góra/Dół" nosa
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    # Oś Y (Zielona) - "Lewo/Prawo" uszu
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    # Oś Z (Niebieska) - "Wprzód" z twarzy
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)

    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = 1 # Przesunięcie kamery

    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o

    xp2 = (xaxis[0] / xaxis[2] * center_x) + cx
    yp2 = (xaxis[1] / xaxis[2] * center_x) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2) # CZERWONY

    xp2 = (yaxis[0] / yaxis[2] * center_x) + cx
    yp2 = (yaxis[1] / yaxis[2] * center_x) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2) # ZIELONY

    xp1 = (zaxis1[0] / zaxis1[2] * center_x) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * center_x) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * center_x) + cx
    yp2 = (zaxis[1] / zaxis[2] * center_x) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2) # NIEBIESKI
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)

    return frame

def main():
    core = Core()
    
    # 1. Wczytanie modeli
    face_net = core.compile_model(model=FACE_XML, device_name="CPU")
    emotion_net = core.compile_model(model=EMOTION_XML, device_name="CPU")
    head_pose_net = core.compile_model(model=HEAD_POSE_XML, device_name="CPU")

    # Pobranie warstw wejściowych/wyjściowych
    face_output = face_net.output(0)
    emotion_output = emotion_net.output(0)
    
    # Dla Head Pose musimy znać nazwy wyjść (Yaw, Pitch, Roll)
    hp_out_y = head_pose_net.output("angle_y_fc")
    hp_out_p = head_pose_net.output("angle_p_fc")
    hp_out_r = head_pose_net.output("angle_r_fc")

    cap = cv2.VideoCapture(0)

    print("System gotowy! Naciśnij 'q' aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]

        # 1. Wykrywanie twarzy
        input_image = cv2.resize(frame, (300, 300))
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, 0)

        results = face_net([input_image])[face_output]
        detections = results[0][0]

        for detection in detections:
            confidence = detection[2]
            if confidence > 0.6:
                xmin = int(detection[3] * w)
                ymin = int(detection[4] * h)
                xmax = int(detection[5] * w)
                ymax = int(detection[6] * h)

                xmin = max(0, xmin); ymin = max(0, ymin)
                xmax = min(w, xmax); ymax = min(h, ymax)

                face_crop = frame[ymin:ymax, xmin:xmax]
                if face_crop.size == 0: continue

                # 2. Emocje (64x64)
                face_em = cv2.resize(face_crop, (64, 64))
                face_em = face_em.transpose(2, 0, 1)
                face_em = np.expand_dims(face_em, 0)
                
                em_res = emotion_net([face_em])[emotion_output]
                emotion_text = EMOTION_LABELS[np.argmax(em_res)]

                # 3. Head Pose (60x60) - UWAGA: Inny rozmiar niż emocje!
                face_hp = cv2.resize(face_crop, (60, 60))
                face_hp = face_hp.transpose(2, 0, 1)
                face_hp = np.expand_dims(face_hp, 0)

                hp_res = head_pose_net([face_hp])
                yaw = hp_res[hp_out_y][0][0]
                pitch = hp_res[hp_out_p][0][0]
                roll = hp_res[hp_out_r][0][0]

                # Rysowanie
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Rysujemy osie na środku twarzy
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                draw_axes(frame, center_x, center_y, yaw, pitch, roll)

                # Podpis
                info = f"{emotion_text} | Y:{yaw:.0f} P:{pitch:.0f}"
                cv2.putText(frame, info, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Asystent AI - Full Version", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()