Markdown
# EjAj pomagacz skupienia podczas nauki 🧠

Projekt w ramach szkolenia dla nauczycielek i nauczycieli Lekcja: AI. Aplikacja wykorzystuje Computer Vision (OpenVINO) do analizy uwagi ucznia w czasie rzeczywistym.

## Funkcje
- Wykrywanie twarzy i emocji.
- Analiza kątów głowy (Yaw, Pitch) do oceny rozproszenia.
- Raport końcowy z wykresem skupienia.

## Jak uruchomić (Instrukcja)

1. **Zainstaluj biblioteki:**
   ```bash
   pip install -r requirements.txt
2. **Pobierz modele AI (Wymagane!):**
    ```bash
    omz_downloader --name face-detection-retail-0004,head-pose-estimation-adas-0001,emotions-recognition-retail-0003
3. **Uruchom aplikację:**
    ```bash
    python main.py

## AndrzejW