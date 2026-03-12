import os
import json
import time
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

VIEW_W, VIEW_H = 480, 270     
SIDE_W = 320                   
CANVAS_W = VIEW_W + VIEW_W + SIDE_W
CANVAS_H = VIEW_H + VIEW_H

IMG_SIZE = (224, 224)

OCC_GRID = 8
OCC_STRIDE = 2
OCC_UPDATE_EVERY = 12

ALPHA_CAM = 0.40
ALPHA_OCC = 0.45


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["labels"]


def preprocess_for_model(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return np.expand_dims(x, axis=0)


def predict(model: tf.keras.Model, frame_bgr: np.ndarray):
    x = preprocess_for_model(frame_bgr)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs, x


def overlay_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, hm_color, alpha, 0)


def occlusion_sensitivity_map(model: tf.keras.Model, frame_bgr: np.ndarray, target_class: int, base_prob: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    grid = OCC_GRID
    cell_h = max(1, h // grid)
    cell_w = max(1, w // grid)

    impact = np.zeros((grid, grid), dtype=np.float32)

    for gy in range(0, grid, OCC_STRIDE):
        for gx in range(0, grid, OCC_STRIDE):
            y0 = gy * cell_h
            x0 = gx * cell_w
            y1 = min(h, y0 + cell_h)
            x1 = min(w, x0 + cell_w)

            occluded = frame_bgr.copy()
            occluded[y0:y1, x0:x1] = 0

            _, _, probs, _ = predict(model, occluded)
            new_prob = float(probs[target_class])

            drop = max(0.0, base_prob - new_prob)
            impact[gy, gx] = drop

    if impact.max() > 0:
        impact = impact / (impact.max() + 1e-8)

    impact_img = cv2.resize(impact, (w, h), interpolation=cv2.INTER_NEAREST)
    return impact_img



def build_cam_model_from_trained(trained_model: tf.keras.Model):
    backbone_old = trained_model.get_layer("mobilenetv2_1.00_224")
    gap_old = trained_model.get_layer("global_average_pooling2d")
    drop_old = trained_model.get_layer("dropout")
    dense_old = trained_model.get_layer("dense")
    out_old = trained_model.get_layer("dense_1")

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights=None
    )
    backbone.set_weights(backbone_old.get_weights())

    conv_layer_name = "Conv_1"
    backbone_cam = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(conv_layer_name).output, backbone.output],
        name="backbone_cam"
    )

    inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="cam_input")
    conv_out, base_out = backbone_cam(inp, training=False)

    gap = tf.keras.layers.GlobalAveragePooling2D(name=gap_old.name)
    drop = tf.keras.layers.Dropout(rate=float(drop_old.rate), name=drop_old.name)
    dense = tf.keras.layers.Dense(
        units=int(dense_old.units),
        activation=dense_old.activation,
        name=dense_old.name
    )
    out = tf.keras.layers.Dense(
        units=int(out_old.units),
        activation=out_old.activation,
        name=out_old.name
    )

    x = gap(base_out)
    x = drop(x, training=False)
    x = dense(x)
    preds = out(x)

    cam_model = tf.keras.Model(inputs=inp, outputs=[conv_out, preds], name="cam_model")
    dense.set_weights(dense_old.get_weights())
    out.set_weights(out_old.get_weights())

    return cam_model, conv_layer_name


def gradcam_heatmap_from_cam_model(cam_model: tf.keras.Model, x: np.ndarray, class_index: int):
    x_tf = tf.convert_to_tensor(x)

    with tf.GradientTape() as tape:
        conv_out, preds = cam_model(x_tf, training=False)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ==========================
# Painel lateral:A/E/I/O/U 
# ==========================
def draw_prob_bars_fixed(panel, labels, probs, y0, pred_idx):
    x_label = 14
    x_bar = 55
    x_pct = SIDE_W - 70

    bar_w = SIDE_W - (x_bar + 85)
    bar_h = 10
    gap = 18

    desired = ["A", "E", "I", "O", "U"]
    idx_map = {lab: i for i, lab in enumerate(labels)}
    order = [idx_map[lab] for lab in desired if lab in idx_map]

    y = y0
    for k in order:
        p = float(probs[k])
        lab = labels[k]
        is_pred = (k == pred_idx)

        color_label = (0, 220, 255) if is_pred else (255, 255, 255)

        cv2.putText(panel, lab, (x_label, y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_label, 1)

        cv2.rectangle(panel, (x_bar, y), (x_bar + bar_w, y + bar_h), (60, 60, 60), -1)

        fill = int(bar_w * p)
        fill_color = (255, 255, 255) if is_pred else (210, 210, 210)
        cv2.rectangle(panel, (x_bar, y), (x_bar + fill, y + bar_h), fill_color, -1)

        cv2.putText(panel, f"{p*100:4.1f}%", (x_pct, y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        y += gap

    return y


def make_side_panel(labels, probs, pred_idx, conf, fps, gradcam_on, occ_on):
    panel = np.zeros((VIEW_H, SIDE_W, 3), dtype=np.uint8)

    def put(line, y, scale=0.55, thick=1, color=(255, 255, 255)):
        cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    put("AUTO ML + XAI", 24, 0.70, 2)

    put("Predicao:", 48, 0.55, 1)
    put(f"{labels[pred_idx]}  ({conf*100:.1f}%)", 72, 0.78, 2, (0, 220, 255))

    put(f"FPS: {fps:.1f}", 98, 0.55, 1)
    put(f"Grad-CAM: {'ON' if gradcam_on else 'OFF'}", 122, 0.52, 1)
    put(f"Occlusion: {'ON' if occ_on else 'OFF'}", 142, 0.52, 1)

    cv2.line(panel, (14, 154), (SIDE_W - 14, 154), (70, 70, 70), 1)

    put("Distribuicao (A,E,I,O,U):", 176, 0.58, 2)

    # Como removemos o rodapé, dá pra subir um pouco se quiser:
    # antes: 190
    _ = draw_prob_bars_fixed(panel, labels, probs, 188, pred_idx)

    return panel


def make_blank(title: str):
    img = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    if title:
        cv2.putText(img, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def main():
    print("\n[START] realtime_infer_xai_layout.py iniciou ✅")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise RuntimeError(f"labels.json não encontrado em: {LABELS_PATH}")

    labels = load_labels()
    trained_model = tf.keras.models.load_model(MODEL_PATH)
    cam_model, conv_layer = build_cam_model_from_trained(trained_model)

    print(f"[OK] Modelo: {MODEL_PATH}")
    print(f"[OK] Labels: {labels}")
    print(f"[OK] Grad-CAM camada: {conv_layer}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam (VideoCapture(0)).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win = "Vogais Libras - AutoML + XAI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CANVAS_W, CANVAS_H)

    prev = time.time()
    fps = 0.0
    frame_count = 0
    last_occ_map = None

    gradcam_on = True
    occ_on = True

    print("[RUN] Abrindo webcam... (q = sair, g = Grad-CAM, o = Occlusion)")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Falha ao ler frame da webcam.")
            break

        frame = cv2.flip(frame, 1)
        webcam_view = cv2.resize(frame, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)

        pred_idx, conf, probs, x = predict(trained_model, webcam_view)

        now = time.time()
        dt = now - prev
        prev = now
        fps = 0.90 * fps + 0.10 * (1.0 / max(dt, 1e-6))

        # Grad-CAM view
        if gradcam_on:
            heatmap = gradcam_heatmap_from_cam_model(cam_model, x, pred_idx)
            cam_view = overlay_heatmap(webcam_view, heatmap, alpha=ALPHA_CAM)
        else:
            cam_view = webcam_view.copy()

        # Occlusion view
        if occ_on:
            frame_count += 1
            if last_occ_map is None or (frame_count % OCC_UPDATE_EVERY == 0):
                base_prob = float(probs[pred_idx])
                last_occ_map = occlusion_sensitivity_map(trained_model, webcam_view, pred_idx, base_prob)
            occ_view = overlay_heatmap(webcam_view, last_occ_map, alpha=ALPHA_OCC)
        else:
            occ_view = make_blank("OCCLUSION (OFF)")

        # títulos
        cv2.putText(webcam_view, "WEBCAM", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(cam_view, "GRAD-CAM (XAI)", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if occ_on:
            cv2.putText(occ_view, "OCCLUSION (EVIDENCIA)", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        side = make_side_panel(labels, probs, pred_idx, conf, fps, gradcam_on, occ_on)

        blank = make_blank("")

        top_row = np.hstack([webcam_view, cam_view, side])
        bottom_row = np.hstack([occ_view, blank, np.zeros((VIEW_H, SIDE_W, 3), dtype=np.uint8)])
        canvas = np.vstack([top_row, bottom_row])

        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            gradcam_on = not gradcam_on
        elif key == ord("o"):
            occ_on = not occ_on

    cap.release()
    cv2.destroyAllWindows()
    print("[END] Finalizado ✅")


if __name__ == "__main__":
    main()