import cv2
import numpy as np
from openvino.runtime import Core
import matplotlib.pyplot as plt  # Do rysowania wykresu na koniec
import collections # Do tworzenia bufora historii (średnia krocząca)
import time

# --- KONFIGURACJA ŚCIEŻEK (KLOCKI LEGO) ---
MODEL_DIR = "intel"
FACE_XML = f"{MODEL_DIR}/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
FACE_BIN = f"{MODEL_DIR}/face-detection-retail-0004/FP32/face-detection-retail-0004.bin"
EMOTION_XML = f"{MODEL_DIR}/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"
EMOTION_BIN = f"{MODEL_DIR}/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin"
HEAD_POSE_XML = f"{MODEL_DIR}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
HEAD_POSE_BIN = f"{MODEL_DIR}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin"

EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']

# --- ZMIENNE DO ANALIZY (PAMIĘĆ ASYSTENTA) ---
# Bufor na 5 ostatnich pomiarów (do wygładzania drgań)
historia_yaw = collections.deque(maxlen=5)
historia_pitch = collections.deque(maxlen=5)

# Lista do zapisu przebiegu lekcji (do wykresu)
raport_skupienia = [] 
czasy_pomiarow = []
start_czas = time.time()

# --- FUNKCJA 1: MATEMATYKA STABILIZACJI (Średnia Krocząca) ---
def wygladz_dane(nowa_wartosc, bufor):
    bufor.append(nowa_wartosc)
    # Obliczamy średnią z tego co jest w buforze
    srednia = sum(bufor) / len(bufor)
    return srednia

# --- FUNKCJA 2: DRZEWO DECYZYJNE (LOGIKA) ---
def ocen_skupienie(yaw, pitch, emocja):
    # Zasada 1: Rozproszenie (patrzenie w bok)
    # Używamy abs(), bo -30 stopni (lewo) i 30 stopni (prawo) to to samo rozproszenie
    if abs(yaw) > 25:
        return "ROZPROSZONY (BOK)", 0 # 0 pkt skupienia
    
    # Zasada 2: Telefon pod ławką (głowa w dół)
    elif pitch > 20:
        return "ROZPROSZONY (DOL)", 0
    
    # Zasada 3: Zmęczenie (Smutek)
    elif emocja == "Sad":
        return "ZMECZENIE", 0.5 # Pół punktu
    
    # Jeśli nic z powyższych -> Skupienie
    else:
        return "SKUPIONY", 1 # 1 pkt skupienia

def main():
    # 1. Inicjalizacja Silnika AI
    print("Ładowanie modeli...")
    core = Core()
    
    face_net = core.compile_model(model=FACE_XML, device_name="CPU")
    emotion_net = core.compile_model(model=EMOTION_XML, device_name="CPU")
    head_pose_net = core.compile_model(model=HEAD_POSE_XML, device_name="CPU")

    # Pobieranie "uchwytów" do wyników
    face_output = face_net.output(0)
    emotion_output = emotion_net.output(0)
    hp_out_y = head_pose_net.output("angle_y_fc")
    hp_out_p = head_pose_net.output("angle_p_fc")

    cap = cv2.VideoCapture(0)
    print("Start! Naciśnij 'q' aby zakończyć i zobaczyć wykres.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        
        # --- KLOCEK 1: DETEKTOR TWARZY (300x300) ---
        img_input = cv2.resize(frame, (300, 300))
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, 0)
        
        results = face_net([img_input])[face_output]
        detections = results[0][0]

        twarz_wykryta = False

        for detection in detections:
            confidence = detection[2]
            if confidence > 0.6:
                twarz_wykryta = True
                
                # Współrzędne ramki
                xmin = int(detection[3] * w)
                ymin = int(detection[4] * h)
                xmax = int(detection[5] * w)
                ymax = int(detection[6] * h)

                # Zabezpieczenie krawędzi
                xmin = max(0, xmin); ymin = max(0, ymin)
                xmax = min(w, xmax); ymax = min(h, ymax)

                face_crop = frame[ymin:ymax, xmin:xmax]
                if face_crop.size == 0: continue

                # --- KLOCEK 2: ANALITYK GŁOWY (60x60) ---
                face_hp = cv2.resize(face_crop, (60, 60)) # Ważne: 60x60!
                face_hp = face_hp.transpose(2, 0, 1)
                face_hp = np.expand_dims(face_hp, 0)

                hp_res = head_pose_net([face_hp])
                raw_yaw = hp_res[hp_out_y][0][0]   # Surowe dane (drżące)
                raw_pitch = hp_res[hp_out_p][0][0]

                # TU DZIEJE SIĘ MAGIA WYGŁADZANIA (FILTR)
                # Uczniowie mogą zakomentować tę linię, żeby zobaczyć różnicę!
                yaw = wygladz_dane(raw_yaw, historia_yaw)
                pitch = wygladz_dane(raw_pitch, historia_pitch)

                # --- KLOCEK 3: EMPATIA (64x64) ---
                face_em = cv2.resize(face_crop, (64, 64))
                face_em = face_em.transpose(2, 0, 1)
                face_em = np.expand_dims(face_em, 0)
                em_res = emotion_net([face_em])[emotion_output]
                emocja = EMOTION_LABELS[np.argmax(em_res)]

                # --- LOGIKA DECYZYJNA ---
                status_tekst, punktacja = ocen_skupienie(yaw, pitch, emocja)

                # Zbieranie danych do wykresu
                raport_skupienia.append(punktacja)
                czasy_pomiarow.append(time.time() - start_czas)

                # Rysowanie na ekranie
                color = (0, 255, 0) if punktacja == 1 else (0, 0, 255) # Zielony/Czerwony
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                info = f"{status_tekst}"
                dane = f"Yaw: {int(yaw)} Pitch: {int(pitch)} Emocja: {emocja}"
                
                cv2.putText(frame, info, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, dane, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Jeśli brak twarzy -> też zapisujemy jako brak skupienia (0)
        if not twarz_wykryta:
            raport_skupienia.append(0)
            czasy_pomiarow.append(time.time() - start_czas)
            cv2.putText(frame, "BRAK UCZNIA!", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Cyfrowy Trener Skupienia", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- FINAŁ: GENEROWANIE WYKRESU (MATPLOTLIB) ---
    print("Generowanie raportu...")
    plt.figure(figsize=(10, 4))
    plt.plot(czasy_pomiarow, raport_skupienia, label='Poziom Skupienia')
    plt.axhline(y=0.8, color='r', linestyle='--', label='Próg sukcesu')
    plt.title("Raport Uważności Ucznia")
    plt.xlabel("Czas lekcji (sekundy)")
    plt.ylabel("Status (1=Skupiony, 0=Rozproszony)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()