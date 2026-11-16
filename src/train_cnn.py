import os, json, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, EPOCHS, PATIENCE, BS, LR = 42, 50, 8, 64, 1e-3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

norm = transforms.Normalize((0.5,), (0.5,))
train_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), norm])

class PathDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p)), y

class EmotionCNN(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        def block(i, o, p):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(p))
        self.features = nn.Sequential(block(1, 32, 0.1), block(32, 64, 0.2),
                                      block(64, 128, 0.3), block(128, 256, 0.3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(256, n_classes))
    def forward(self, x):
        return self.head(self.features(x))

def evaluate(model, loader, criterion):
    model.eval(); loss_sum, preds, labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            preds += out.argmax(1).cpu().tolist(); labels += y.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    f1 = f1_score(labels, preds, average="macro")
    return loss_sum / len(labels), acc, f1, preds, labels

def build_splits():
    classes = sorted(c for c in os.listdir("data/train") if os.path.isdir(f"data/train/{c}"))
    items = []
    for folder in ["data/train", "data/test"]:
        for ci, c in enumerate(classes):
            d = os.path.join(folder, c)
            for f in sorted(os.listdir(d)):
                items.append((os.path.join(d, f), ci))
    labels = np.array([y for _, y in items]); idx = np.arange(len(items))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=labels[tmp], random_state=SEED)
    split = {"classes": classes,
             "train": [items[i] for i in tr], "val": [items[i] for i in va], "test": [items[i] for i in te]}
    json.dump(split, open("models/split.json", "w"))
    return split

def main():
    split = build_splits(); classes = split["classes"]
    print("Classes:", classes, "| device:", device)
    print(f"Split -> train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")

    train_loader = DataLoader(PathDS(split["train"], train_tf), batch_size=BS, shuffle=True)
    val_loader = DataLoader(PathDS(split["val"], eval_tf), batch_size=BS)
    test_loader = DataLoader(PathDS(split["test"], eval_tf), batch_size=BS)

    counts = np.bincount([y for _, y in split["train"]], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
    print("Class weights:", dict(zip(classes, weights.cpu().numpy().round(2))))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    model = EmotionCNN(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    hist = {"tr_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, bad = 0.0, 0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); tl, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * len(y); n += len(y)
        vl, va, vf, _, _ = evaluate(model, val_loader, criterion)
        sched.step(vl)
        for k, v in zip(hist, [tl / n, vl, va, vf]): hist[k].append(v)
        flag = ""
        if vf > best_f1:
            best_f1, bad = vf, 0
            torch.save({"state": model.state_dict(), "classes": classes}, "models/cnn_best.pt")
            flag = " ✅ saved"
        else:
            bad += 1
        print(f"Epoch {ep:02d} | train_loss {tl/n:.3f} | val_loss {vl:.3f} | val_acc {va:.3f} | val_F1 {vf:.3f} | {time.time()-t0:.0f}s{flag}")
        if bad >= PATIENCE:
            print("Early stopping."); break

    ck = torch.load("models/cnn_best.pt", map_location=device)
    model.load_state_dict(ck["state"])
    _, ta, tf1, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\nTEST accuracy: {ta:.3f} | macro-F1: {tf1:.3f}\n")
    print(classification_report(labels, preds, target_names=classes, digits=3))

    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=classes).plot(cmap="Reds", xticks_rotation=45)
    plt.tight_layout(); plt.savefig("models/cnn_confusion.png", dpi=120); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["val_acc"], label="val acc"); ax[1].plot(hist["val_f1"], label="val macro-F1"); ax[1].set_title("Validation"); ax[1].legend()
    plt.tight_layout(); plt.savefig("models/cnn_curves.png", dpi=120)
    json.dump({"model": "CNN", "test_acc": ta, "test_macro_f1": tf1}, open("models/cnn_results.json", "w"))

if __name__ == "__main__":
    main()
