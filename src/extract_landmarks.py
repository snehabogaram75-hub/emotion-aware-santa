import json, numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

opts = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE, num_faces=1)
det = vision.FaceLandmarker.create_from_options(opts)

def normalize(lm):
    p = np.array([[l.x, l.y, l.z] for l in lm[:468]], dtype=np.float32)
    p -= p[1]                                   # nose tip = origin
    v = p[263, :2] - p[33, :2]                  # eye-corner line
    a = np.arctan2(v[1], v[0]); c, s = np.cos(a), np.sin(a)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0], p[:, 1] = c * x + s * y, -s * x + c * y   # undo head roll
    p /= (np.linalg.norm(v) + 1e-6)             # scale by eye distance
    return p.flatten()

split = json.load(open("models/split.json"))
classes = split["classes"]
X, y, sp = [], [], []
stats = {}
for si, name in enumerate(["train", "val", "test"]):
    hit = 0
    for path, label in tqdm(split[name], desc=name):
        img = cv2.imread(path)
        img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, 48, 48, 48, 48, cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.face_landmarks:
            X.append(normalize(res.face_landmarks[0])); y.append(label); sp.append(si); hit += 1
    stats[name] = (hit, len(split[name]))

np.savez_compressed("data/landmarks.npz", X=np.array(X), y=np.array(y), split=np.array(sp), classes=np.array(classes))
print("\nCoverage (detected / total):")
for k, (h, t) in stats.items():
    print(f"  {k:5s}: {h}/{t} = {h/t:.1%}")
print("Features shape:", np.array(X).shape, "-> saved data/landmarks.npz")
