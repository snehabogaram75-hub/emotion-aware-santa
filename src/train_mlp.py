import json, random, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, EPOCHS, PATIENCE, BS, LR = 42, 150, 15, 128, 1e-3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

d = np.load("data/landmarks.npz")
X, y, sp, classes = d["X"], d["y"], d["split"], [str(c) for c in d["classes"]]
Xtr, ytr, Xva, yva, Xte, yte = X[sp == 0], y[sp == 0], X[sp == 1], y[sp == 1], X[sp == 2], y[sp == 2]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
prep = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
print(f"Samples -> train {len(ytr)} | val {len(yva)} | test {len(yte)} | device: {device}")

class LandmarkMLP(nn.Module):
    def __init__(self, d_in=1404, n=7):
        super().__init__()
        def blk(i, o, p):
            return nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(p))
        self.net = nn.Sequential(blk(d_in, 512, 0.4), blk(512, 256, 0.4), blk(256, 128, 0.3), nn.Linear(128, n))
    def forward(self, x):
        return self.net(x)

def evaluate(model, X_, y_, criterion):
    model.eval()
    with torch.no_grad():
        out = model(X_.to(device)); yy = torch.tensor(y_).to(device)
        loss = criterion(out, yy).item(); preds = out.argmax(1).cpu().numpy()
    return loss, float((preds == y_).mean()), f1_score(y_, preds, average="macro"), preds

counts = np.bincount(ytr, minlength=len(classes))
weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
model = LandmarkMLP(X.shape[1], len(classes)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
loader = DataLoader(TensorDataset(Xtr_t, torch.tensor(ytr)), batch_size=BS, shuffle=True)

hist = {"tr_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
best_f1, bad = 0.0, 0
for ep in range(1, EPOCHS + 1):
    model.train(); tl, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = xb + 0.05 * torch.randn_like(xb)           # noise augmentation
        opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
        tl += loss.item() * len(yb); n += len(yb)
    vl, va, vf, _ = evaluate(model, Xva_t, yva, criterion); sched.step(vl)
    for k, v in zip(hist, [tl / n, vl, va, vf]): hist[k].append(v)
    if vf > best_f1:
        best_f1, bad = vf, 0
        torch.save({"state": model.state_dict(), "classes": classes,
                    "mu": torch.tensor(mu), "sd": torch.tensor(sd)}, "models/mlp_best.pt")
        flag = " ✅ saved"
    else:
        bad += 1; flag = ""
    if ep % 5 == 0 or flag:
        print(f"Epoch {ep:03d} | train_loss {tl/n:.3f} | val_loss {vl:.3f} | val_acc {va:.3f} | val_F1 {vf:.3f}{flag}")
    if bad >= PATIENCE:
        print("Early stopping."); break

ck = torch.load("models/mlp_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ck["state"])
_, ta, tf1, preds = evaluate(model, Xte_t, yte, criterion)
print(f"\nMLP TEST (detected faces only) accuracy: {ta:.3f} | macro-F1: {tf1:.3f}\n")
print(classification_report(yte, preds, target_names=classes, digits=3))

n_test_all = len(json.load(open("models/split.json"))["test"])
strict = (preds == yte).sum() / n_test_all
print(f"Strict accuracy (undetected faces counted as wrong): {strict:.3f}  [{len(yte)}/{n_test_all} detected]")

cpu_model = model.to("cpu").eval(); x1 = Xte_t[:1]
with torch.no_grad():
    t0 = time.time()
    for _ in range(300): cpu_model(x1)
ms = (time.time() - t0) / 300 * 1000
print(f"MLP inference: {ms:.2f} ms/face (CPU, excludes MediaPipe)")

ConfusionMatrixDisplay(confusion_matrix(yte, preds), display_labels=classes).plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout(); plt.savefig("models/mlp_confusion.png", dpi=120); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["val_acc"], label="val acc"); ax[1].plot(hist["val_f1"], label="val macro-F1"); ax[1].set_title("Validation"); ax[1].legend()
plt.tight_layout(); plt.savefig("models/mlp_curves.png", dpi=120)
json.dump({"model": "MLP", "test_acc": ta, "test_macro_f1": tf1, "strict_acc": float(strict), "ms_per_face": ms},
          open("models/mlp_results.json", "w"))

cnn = json.load(open("models/cnn_results.json"))
print("\n=== COMPARISON ===")
print(f"CNN (pixels, all test images) : acc {cnn['test_acc']:.3f} | macro-F1 {cnn['test_macro_f1']:.3f}")
print(f"MLP (landmarks, detected only): acc {ta:.3f} | macro-F1 {tf1:.3f}")
