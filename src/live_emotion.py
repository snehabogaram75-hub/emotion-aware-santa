import sys, time
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
    sys.exit("Cannot open camera. Check System Settings > Privacy & Security > Camera > allow Terminal.")

ALPHA = 0.3                      # smoothing: lower = smoother, higher = faster reaction
smooth = np.ones(len(classes)) / len(classes)
fps, t_prev = 0.0, time.time()
print("Running. Press q in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    label, conf = "no face", 0.0
    if res.face_landmarks:
        lm = res.face_landmarks[0]
        xs = np.array([p.x for p in lm]) * w; ys = np.array([p.y for p in lm]) * h
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        side = max(x1 - x0, y1 - y0) * 1.1
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        c, d = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        face = frame[b:d, a:c]
        if face.size > 0:
            g = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (48, 48))
            x = torch.tensor((g / 255.0 - 0.5) / 0.5, dtype=torch.float32)[None, None]
            with torch.no_grad():
                p = torch.softmax(model(x), 1)[0].numpy()
            smooth = ALPHA * p + (1 - ALPHA) * smooth
            k = int(smooth.argmax()); label, conf = classes[k], float(smooth[k])
            cv2.rectangle(frame, (a, b), (c, d), (0, 200, 0), 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (a, max(b - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    # probability bars
    for i, cl in enumerate(classes):
        cv2.putText(frame, cl, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (90, 12 + i * 22), (90 + int(smooth[i] * 150), 26 + i * 22), (0, 140, 255), -1)
    now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6); t_prev = now
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Emotion Santa - live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release(); cv2.destroyAllWindows()
