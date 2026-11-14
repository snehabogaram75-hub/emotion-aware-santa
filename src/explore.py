import os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root = "data/train"
classes = sorted(os.listdir(root))
classes = [c for c in classes if os.path.isdir(os.path.join(root, c))]
counts = {c: len(os.listdir(os.path.join(root, c))) for c in classes}

# 1) Class distribution
plt.figure(figsize=(8, 4))
plt.bar(counts.keys(), counts.values(), color="#c0392b")
plt.title("FER2013 train: class distribution")
plt.ylabel("images")
plt.tight_layout()
plt.savefig("models/class_distribution.png", dpi=120)

# 2) Sample grid: 6 random images per class
random.seed(42)
fig, axes = plt.subplots(len(classes), 6, figsize=(9, 1.6 * len(classes)))
for i, c in enumerate(classes):
    files = random.sample(os.listdir(os.path.join(root, c)), 6)
    for j, f in enumerate(files):
        img = Image.open(os.path.join(root, c, f))
        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(c, fontsize=9, loc="left")
plt.tight_layout()
plt.savefig("models/samples.png", dpi=120)

print("Image size:", img.size, "| mode:", img.mode)
print("Saved: models/class_distribution.png, models/samples.png")
