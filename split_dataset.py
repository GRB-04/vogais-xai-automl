import os
import random
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Em vez de data/train e data/val fixos, cria pastas versionadas
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAIN_DIR = os.path.join(BASE_DIR, "data", f"train_{STAMP}")
VAL_DIR   = os.path.join(BASE_DIR, "data", f"val_{STAMP}")

CLASSES = ["A", "E", "I", "O", "U"]

VAL_RATIO = 0.2
SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png")

def list_images(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]

def main():
    random.seed(SEED)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    for c in CLASSES:
        src_dir = os.path.join(RAW_DIR, c)
        if not os.path.exists(src_dir):
            raise RuntimeError(f"Pasta não encontrada: {src_dir}")

        imgs = list_images(src_dir)
        if len(imgs) == 0:
            raise RuntimeError(f"Sem imagens na classe {c}")

        random.shuffle(imgs)
        n_val = int(len(imgs) * VAL_RATIO)
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        dst_train = os.path.join(TRAIN_DIR, c)
        dst_val = os.path.join(VAL_DIR, c)
        os.makedirs(dst_train, exist_ok=True)
        os.makedirs(dst_val, exist_ok=True)

        for fname in train_imgs:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_train, fname))

        for fname in val_imgs:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_val, fname))

        print(f"{c}: raw={len(imgs)} | train={len(train_imgs)} | val={len(val_imgs)}")

    # Salva um arquivo de "ponteiro" pra treino/inferência saberem onde está
    pointer_path = os.path.join(BASE_DIR, "data", "SPLIT_POINTER.txt")
    with open(pointer_path, "w", encoding="utf-8") as f:
        f.write(f"TRAIN={TRAIN_DIR}\n")
        f.write(f"VAL={VAL_DIR}\n")

    print("\n✅ Split concluído com sucesso.")
    print("TRAIN:", TRAIN_DIR)
    print("VAL:  ", VAL_DIR)
    print("Pointer salvo em:", pointer_path)

if __name__ == "__main__":
    main()