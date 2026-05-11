import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")

def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

def show_counts(base_path, title):
    print(f"\n===== {title} =====")
    if not os.path.exists(base_path):
        print("Pasta não encontrada.")
        return

    total = 0
    for class_name in sorted(os.listdir(base_path)):
        class_path = os.path.join(base_path, class_name)
        if os.path.isdir(class_path):
            count = count_images(class_path)
            total += count
            print(f"{class_name}: {count} imagens")
    print(f"TOTAL: {total} imagens")

def main():
    show_counts(RAW_DIR, "DATA RAW")
    show_counts(TRAIN_DIR, "DATA TRAIN")
    show_counts(VAL_DIR, "DATA VAL")

if __name__ == "__main__":
    main()