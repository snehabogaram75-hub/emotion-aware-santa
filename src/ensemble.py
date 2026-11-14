import json, numpy as np, cv2, torch, torch.nn as nn
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.metrics import f1_score
from tqdm import tqdm
from train_cnn import EmotionCNN

split = json.load(open("models/split.json")); classes = split["classes"]

ck = torch.load("models/cnn_best.pt", map_location="cpu")
cnn = EmotionCNN(len(classes)); cnn.load_state_dict(ck["state"]); cnn.eval()

class LandmarkMLP(nn.Module):
    def __init__(self, d_in=1404, n=7):
        super().__init__()
        def blk(i, o, p):
            return nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(p))
        self.net = nn.Sequential(blk(d_in, 512, 0.4), blk(512, 256, 0.4), blk(256, 128, 0.3), nn.Linear(128, n))
    def forward(self, x):
        return self.net(x)

mk = torch.load("models/mlp_best.pt", map_location="cpu", weights_only=False)
mlp = LandmarkMLP(1404, len(classes)); mlp.load_state_dict(mk["state"]); mlp.eval()
mu, sd = mk["mu"].numpy(), mk["sd"].numpy()

det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE, num_faces=1))

def normalize(lm):
    p = np.array([[l.x, l.y, l.z] for l in lm[:468]], dtype=np.float32)
    p -= p[1]
    v = p[263, :2] - p[33, :2]
    a = np.arctan2(v[1], v[0]); c, s = np.cos(a), np.sin(a)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0], p[:, 1] = c * x + s * y, -s * x + c * y
    p /= (np.linalg.norm(v) + 1e-6)
    return p.flatten()

def run(items, name):
    y = np.array([l for _, l in items])
    G = np.stack([np.array(Image.open(p).convert("L").resize((48, 48))) for p, _ in items])
    X = torch.tensor((G / 255.0 - 0.5) / 0.5, dtype=torch.float32)[:, None]
    with torch.no_grad():
        pc = np.concatenate([torch.softmax(cnn(X[i:i + 256]), 1).numpy() for i in range(0, len(X), 256)])
    d, feats = [], []
    for i, (path, _) in enumerate(tqdm(items, desc=f"landmarks {name}")):
        img = cv2.resize(cv2.imread(path), (192, 192), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, 48, 48, 48, 48, cv2.BORDER_REPLICATE)
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if res.face_landmarks:
            d.append(i); feats.append(normalize(res.face_landmarks[0]))
    with torch.no_grad():
        pm = torch.softmax(mlp(torch.tensor((np.array(feats) - mu) / sd, dtype=torch.float32)), 1).numpy()
    return y, pc, np.array(d), pm

f1 = lambda y, p: f1_score(y, p, average="macro")
acc = lambda y, p: float((y == p).mean())

yv, pcv, dv, pmv = run(split["val"], "val")
best_w, best_f1 = 0.5, -1
for w in np.linspace(0, 1, 11):
    s = f1(yv[dv], (w * pcv[dv] + (1 - w) * pmv).argmax(1))
    if s > best_f1: best_w, best_f1 = float(w), s
print(f"\nBest CNN weight on VAL: {best_w:.1f} (val macro-F1 {best_f1:.3f})")

yt, pct, dt, pmt = run(split["test"], "test")
pe = best_w * pct[dt] + (1 - best_w) * pmt
print(f"\n=== TEST, detected faces only ({len(dt)}/{len(yt)}) ===")
print(f"CNN      : acc {acc(yt[dt], pct[dt].argmax(1)):.3f} | macro-F1 {f1(yt[dt], pct[dt].argmax(1)):.3f}")
print(f"MLP      : acc {acc(yt[dt], pmt.argmax(1)):.3f} | macro-F1 {f1(yt[dt], pmt.argmax(1)):.3f}")
print(f"Ensemble : acc {acc(yt[dt], pe.argmax(1)):.3f} | macro-F1 {f1(yt[dt], pe.argmax(1)):.3f}")

full = pct.argmax(1).copy(); full[dt] = pe.argmax(1)
print(f"\n=== TEST, all {len(yt)} images (ensemble where face detected, CNN otherwise) ===")
print(f"CNN only : acc {acc(yt, pct.argmax(1)):.3f} | macro-F1 {f1(yt, pct.argmax(1)):.3f}")
print(f"Ensemble : acc {acc(yt, full):.3f} | macro-F1 {f1(yt, full):.3f}")
json.dump({"cnn_weight": best_w,
           "detected": {"n": int(len(dt)), "cnn_acc": acc(yt[dt], pct[dt].argmax(1)), "cnn_f1": f1(yt[dt], pct[dt].argmax(1)),
                        "mlp_acc": acc(yt[dt], pmt.argmax(1)), "mlp_f1": f1(yt[dt], pmt.argmax(1)),
                        "ens_acc": acc(yt[dt], pe.argmax(1)), "ens_f1": f1(yt[dt], pe.argmax(1))},
           "all": {"n": int(len(yt)), "cnn_acc": acc(yt, pct.argmax(1)), "cnn_f1": f1(yt, pct.argmax(1)),
                   "ens_acc": acc(yt, full), "ens_f1": f1(yt, full)}},
          open("models/ensemble_results.json", "w"), indent=1)
