import cv2

API_NAMES = {
    getattr(cv2, "CAP_DSHOW", None): "CAP_DSHOW (DirectShow)",
    getattr(cv2, "CAP_MSMF", None): "CAP_MSMF (Media Foundation)",
    getattr(cv2, "CAP_ANY", None): "CAP_ANY",
}

def api_name(api):
    return API_NAMES.get(api, str(api))

def try_open(index: int, api: int) -> bool:
    cap = None
    try:
        cap = cv2.VideoCapture(index, api)
        ok = cap.isOpened()

        if not ok:
            print(f"[FAIL] index={index} api={api_name(api)}")
            return False

        ret, frame = cap.read()
        shape = None if frame is None else frame.shape
        print(f"[OK]   index={index} api={api_name(api)} read={ret} shape={shape}")
        return True

    except Exception as e:
        print(f"[ERROR] index={index} api={api_name(api)} -> {type(e).__name__}: {e}")
        return False

    finally:
        if cap is not None:
            cap.release()

def main():
    apis = []
    # Só adiciona se existir no OpenCV instalado
    if hasattr(cv2, "CAP_DSHOW"):
        apis.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        apis.append(cv2.CAP_MSMF)
    apis.append(cv2.CAP_ANY)

    print("=== Teste de webcam (OpenCV) ===")
    print("Vou tentar indices 0..4 com backends:", ", ".join(api_name(a) for a in apis))
    print("--------------------------------")

    any_ok = False
    for idx in range(0, 5):
        for api in apis:
            ok = try_open(idx, api)
            any_ok = any_ok or ok

    print("--------------------------------")
    if not any_ok:
        print("❌ Nenhuma combinação abriu a câmera.")
        print("Sugestões rápidas:")
        print("- Feche Zoom/Meet/Discord/WhatsApp/Camera do Windows/OBS")
        print("- Ative permissões: Configurações > Privacidade e Segurança > Câmera")
        print("- Teste instalar opencv-contrib-python (removendo opencv-python)")
    else:
        print("✅ Pelo menos uma combinação abriu a câmera. Use a combinação [OK] no seu projeto.")

if __name__ == "__main__":
    main()