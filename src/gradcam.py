import json, random
import numpy as np, cv2, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_cnn import EmotionCNN

ck = torch.load("models/cnn_best.pt", map_location="cpu")
classes = ck["classes"]
model = EmotionCNN(len(classes)); model.load_state_dict(ck["state"]); model.eval()

store = {}
def fwd_hook(m, i, o):
    store["a"] = o
    o.register_hook(lambda g: store.__setitem__("g", g))
model.features[3][5].register_forward_hook(fwd_hook)   # last conv block, before pooling

def gradcam(x, cls=None):
    model.zero_grad()
    out = model(x)
    k = int(out.argmax(1)) if cls is None else cls
    out[0, k].backward()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1))[0].detach().numpy()
    cam = cv2.resize(cam, (48, 48))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), k, torch.softmax(out, 1)[0, k].item()

random.seed(7)
test = json.load(open("models/split.json"))["test"]
fig, axes = plt.subplots(len(classes), 6, figsize=(11, 1.9 * len(classes)))
for ci, c in enumerate(classes):
    items = random.sample([t for t in test if t[1] == ci], 3)
    for j, (path, _) in enumerate(items):
        g = np.array(Image.open(path).convert("L").resize((48, 48)))
        x = torch.tensor((g / 255.0 - 0.5) / 0.5, dtype=torch.float32)[None, None]
        cam, k, p = gradcam(x)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        base = np.stack([g] * 3, -1)
        over = (0.55 * base + 0.45 * heat).astype(np.uint8)
        a1, a2 = axes[ci, 2 * j], axes[ci, 2 * j + 1]
        a1.imshow(g, cmap="gray"); a2.imshow(over)
        a1.set_title(f"true: {c}", fontsize=7); a2.set_title(f"pred: {classes[k]} {p:.0%}", fontsize=7)
        a1.axis("off"); a2.axis("off")
plt.suptitle("Grad-CAM: where the CNN looks (red = most influential)", fontsize=10)
plt.tight_layout(); plt.savefig("models/gradcam.png", dpi=130)
print("Saved models/gradcam.png")
