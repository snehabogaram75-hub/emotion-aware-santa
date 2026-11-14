import os, random
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

opts = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE, num_faces=1)
det = vision.FaceLandmarker.create_from_options(opts)

random.seed(0)
total_hit = total = 0
for c in sorted(os.listdir("data/train")):
    d = f"data/train/{c}"
    if not os.path.isdir(d):
        continue
    hit = 0
    files = random.sample(os.listdir(d), 30)
    for f in files:
        img = cv2.imread(os.path.join(d, f))
        img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, 48, 48, 48, 48, cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.face_landmarks:
            hit += 1
    total_hit += hit; total += len(files)
    print(f"{c:9s}: {hit}/{len(files)} detected")
print(f"\nOVERALL detection rate: {total_hit}/{total} = {total_hit/total:.0%}")
