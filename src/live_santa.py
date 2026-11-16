import os, sys, time, random
import numpy as np, cv2, torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from train_cnn import EmotionCNN

ck = torch.load("models/cnn_best.pt", map_location="cpu")
classes = ck["classes"]
model = EmotionCNN(len(classes)); model.load_state_dict(ck["state"]); model.eval()
det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE, num_faces=1))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Cannot open camera. Allow Terminal in System Settings > Privacy & Security > Camera.")
os.makedirs("captures", exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
JAW = [234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454]
CONF = [(0,0,255),(0,200,255),(0,255,0),(255,100,0),(255,0,255),(0,255,255)]
MSG = {"happy": ("Ho Ho Ho!", (0,255,255)), "surprise": ("Wow! A surprise gift!", (0,200,255)),
       "sad": ("Santa is here for you", (255,200,100)), "neutral": ("Merry Christmas!", (255,255,255)),
       "angry": ("NAUGHTY LIST!", (60,60,255)), "fear": ("Don't worry, Santa's here", (255,255,0)),
       "disgust": ("Coal for you!", (200,200,200))}

ALPHA, THRESH = 0.3, 0.40
smooth = np.ones(len(classes)) / len(classes)
ps_prev, mode, parts = None, "neutral", []
fps, t_prev = 0.0, time.time()

def put(img, text, org, scale, color, th=2):
    cv2.putText(img, text, org, FONT, scale, (0,0,0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, th, cv2.LINE_AA)

def axes(ps):
    up = ps[10] - ps[152]; fh = np.linalg.norm(up) + 1e-6; up = up / fh
    right = np.array([-up[1], up[0]], dtype=np.float32)
    fw = np.linalg.norm(ps[454] - ps[234]) + 1e-6
    return up, right, fw, fh

def draw_beard(frame, ps, up, right, fw, fh):
    n = len(JAW); J = ps[JAW].copy()
    J = J - up * fh * 0.14 * np.sin(np.pi * np.arange(n) / (n - 1))[:, None]
    poly = np.vstack([J, ps[[425, 17, 205]]]).astype(np.int32)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    ang = np.degrees(np.arctan2(right[1], right[0]))
    for s in (-1, 1):
        c = ps[164] + right * s * 0.11 * fw
        cv2.ellipse(mask, (int(c[0]), int(c[1])), (int(0.14*fw), int(0.045*fh)), ang, 0, 360, 255, -1)
    k = int(max(fw * 0.08, 3)) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    a = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.95
    frame[:] = (frame * (1 - a) + np.array([245, 245, 245], np.float32) * a).astype(np.uint8)

def draw_hat(frame, ps, up, right, fw, fh):
    top = ps[10]
    P = lambda u, v: (top + right * u * fw + up * v * fh).astype(np.int32)
    cv2.fillPoly(frame, [np.array([P(-0.52,0.08), P(0.52,0.08), P(0.25,0.85)])], (30,30,200))
    cv2.fillPoly(frame, [np.array([P(-0.58,-0.04), P(0.58,-0.04), P(0.58,0.12), P(-0.58,0.12)])], (245,245,245))
    c = P(0.25, 0.85)
    cv2.circle(frame, (int(c[0]), int(c[1])), int(max(0.09 * fw, 4)), (255,255,255), -1)

def spawn(mode, w):
    def add(kind, col, size, vy, n=1):
        for _ in range(n):
            parts.append([random.uniform(0, w), -10, random.uniform(-1, 1), random.uniform(*vy), kind, col, size()])
    snow = lambda n: add("snow", (255,255,255), lambda: random.randint(2, 4), (1.5, 3.5), n)
    if mode == "happy": add("conf", None, lambda: random.randint(4, 8), (3, 7), 4)
    elif mode == "surprise":
        if random.random() < 0.3: add("gift", None, lambda: random.randint(14, 22), (3, 6))
    elif mode == "disgust":
        if random.random() < 0.3: add("coal", (60,60,60), lambda: random.randint(8, 14), (3, 6))
    elif mode == "sad": snow(3)
    elif mode == "fear": snow(2)
    elif mode == "neutral": snow(1)
    for p in parts:
        if p[4] == "conf" and p[5] is None: p[5] = random.choice(CONF)
        if p[4] == "gift" and p[5] is None: p[5] = random.choice(CONF)

def step_draw(frame, h):
    for p in parts:
        p[0] += p[2]; p[1] += p[3]
        if p[4] == "snow": p[2] += random.uniform(-0.2, 0.2)
        x, y, kind, col, s = int(p[0]), int(p[1]), p[4], p[5], p[6]
        if kind == "snow": cv2.circle(frame, (x, y), s, col, -1)
        elif kind == "coal": cv2.circle(frame, (x, y), s, col, -1)
        elif kind == "conf": cv2.rectangle(frame, (x, y), (x + s, y + s // 2 + 1), col, -1)
        elif kind == "gift":
            cv2.rectangle(frame, (x, y), (x + s, y + s), col, -1)
            cv2.line(frame, (x + s // 2, y), (x + s // 2, y + s), (255,255,255), 2)
            cv2.line(frame, (x, y + s // 2), (x + s, y + s // 2), (255,255,255), 2)
    parts[:] = [p for p in parts if p[1] < h + 20][-300:]

AUTO = float(os.environ.get("AUTO_CAPTURE", "0")); t_start = time.time()
print("Running. q = quit | s = save photo to captures/")
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    face, label, conf = False, "no face", 0.0
    if res.face_landmarks:
        lm = res.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], np.float32)
        ps = pts if ps_prev is None else 0.6 * pts + 0.4 * ps_prev
        ps_prev = ps; face = True
        x0, y0 = ps[:468].min(0); x1, y1 = ps[:468].max(0)
        side = max(x1 - x0, y1 - y0) * 1.1; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = int(max(cx - side/2, 0)), int(max(cy - side/2, 0))
        c, d = int(min(cx + side/2, w)), int(min(cy + side/2, h))
        crop = frame[b:d, a:c]                       # crop BEFORE drawing hat/beard
        if crop.size > 0:
            g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 48))
            x = torch.tensor((g / 255.0 - 0.5) / 0.5, dtype=torch.float32)[None, None]
            with torch.no_grad():
                p = torch.softmax(model(x), 1)[0].numpy()
            smooth = ALPHA * p + (1 - ALPHA) * smooth
            k = int(smooth.argmax()); label, conf = classes[k], float(smooth[k])
            if conf > THRESH: mode = label
        up, right, fw, fh = axes(ps)
        draw_beard(frame, ps, up, right, fw, fh)
        draw_hat(frame, ps, up, right, fw, fh)
    else:
        ps_prev = None

    spawn(mode if face else "neutral", w); step_draw(frame, h)
    if face:
        text, col = MSG[mode]
        scale = 1.0 if mode != "angry" else 1.3
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 3)
        put(frame, text, ((w - tw) // 2, h - 40), scale, col, 3)
    put(frame, f"{label} {conf:.0%}", (10, 30), 0.8, (0,255,0), 2)
    now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6); t_prev = now
    put(frame, f"{fps:.0f} FPS", (w - 110, 30), 0.6, (255,255,0), 2)
    put(frame, "q quit | s save", (10, h - 10), 0.45, (255,255,255), 1)
    if AUTO and time.time() - t_start >= AUTO:
        fn = f"captures/santa_{int(time.time())}.png"; cv2.imwrite(fn, frame); print("Saved", fn); break
    if AUTO:
        put(frame, f"Smile! {int(AUTO - (time.time() - t_start)) + 1}", (w // 2 - 80, 70), 1.4, (0, 255, 255), 3)
    cv2.imshow("Emotion-Aware Santa", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord("s"):
        fn = f"captures/santa_{int(time.time())}.png"; cv2.imwrite(fn, frame); print("Saved", fn)

cap.release(); cv2.destroyAllWindows()
