import os
import cv2
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "raw")
CLASSES = ["A", "E", "I", "O", "U"]

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, c), exist_ok=True)

def next_index(class_dir: str) -> int:
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(".jpg")]
    idxs = []
    for f in files:
        name = os.path.splitext(f)[0]
        if name.isdigit():
            idxs.append(int(name))
    return (max(idxs) + 1) if idxs else 0

def main():
    ensure_dirs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam. Verifique permissões/driver.")

    current = "A"
    capturing = False
    interval_s = 0.15
    last_shot = 0.0

    idx = next_index(os.path.join(OUT_DIR, current))

    print("Controles: a/e/i/o/u = classe | SPACE = capturar on/off | q = sair")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler frame.")
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Classe: {current}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Capturando: {capturing}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, "a/e/i/o/u: trocar | SPACE: on/off | q: sair", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        now = time.time()
        if capturing and (now - last_shot) >= interval_s:
            class_dir = os.path.join(OUT_DIR, current)
            path = os.path.join(class_dir, f"{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            idx += 1
            last_shot = now

        cv2.imshow("Coleta - Vogais Libras", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            capturing = not capturing
        elif key in [ord("a"), ord("e"), ord("i"), ord("o"), ord("u")]:
            current = chr(key).upper()
            idx = next_index(os.path.join(OUT_DIR, current))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()