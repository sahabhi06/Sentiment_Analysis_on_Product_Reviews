import os, time, json, random
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ---------- SETTINGS ----------
USE_NEUTRAL = False       # True = 3 classes (neg/neu/pos), False = 2 classes (neg/pos)
BERT_NAME   = "distilbert-base-uncased"
MAX_LEN     = 128         # reviews are cut/padded to 128 tokens
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN     = 50000 if DEVICE == "cuda" else 5000   # fewer reviews on CPU = faster
N_TEST      = 10000
BERT_EPOCHS, LSTM_EPOCHS = 2, 5
OUT_DIR     = "results"
# ------------------------------

random.seed(42); np.random.seed(42); torch.manual_seed(42)
os.makedirs(OUT_DIR, exist_ok=True)
NUM_CLASSES = 3 if USE_NEUTRAL else 2
CLASS_NAMES = ["negative", "neutral", "positive"] if USE_NEUTRAL else ["negative", "positive"]
print("Running on:", DEVICE, "| classes:", CLASS_NAMES)

raw = load_dataset("SetFit/amazon_reviews_multi_en")   # Amazon product reviews (English)

def stars_to_sentiment(example):
    s = example["label"]                      # 0..4 means 1..5 stars
    if USE_NEUTRAL:
        example["label"] = 0 if s < 2 else (1 if s == 2 else 2)
    else:
        example["label"] = 0 if s < 2 else 1  # neutral reviews are removed below
    return example

train_ds = raw["train"].shuffle(seed=42)
test_ds  = raw["test"].shuffle(seed=42)
if not USE_NEUTRAL:
    train_ds = train_ds.filter(lambda x: x["label"] != 2)
    test_ds  = test_ds.filter(lambda x: x["label"] != 2)
train_ds = train_ds.map(stars_to_sentiment).select(range(N_TRAIN))
test_ds  = test_ds.map(stars_to_sentiment).select(range(min(N_TEST, len(test_ds))))
print(len(train_ds), "train reviews |", len(test_ds), "test reviews")
print(train_ds[0]["text"][:150], "->", CLASS_NAMES[train_ds[0]["label"]])

tok = AutoTokenizer.from_pretrained(BERT_NAME)

def make_dataset(ds):
    texts  = [str(t) for t in ds["text"]]     # force a plain list of strings
    labels = [int(l) for l in ds["label"]]    # force a plain list of ints
    enc = tok(texts, padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))

train_data = make_dataset(train_ds)
test_data  = make_dataset(test_ds)
test_loader = DataLoader(test_data, batch_size=64)

# Model 1: DistilBERT (pre-trained, transfer learning)
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = AutoModelForSequenceClassification.from_pretrained(BERT_NAME, num_labels=NUM_CLASSES)
    def forward(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask).logits

# Model 2: Bidirectional LSTM (trained from scratch)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab=tok.vocab_size, emb=128, hid=128):
        super().__init__()
        self.emb  = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hid * 2, NUM_CLASSES)
    def forward(self, ids, mask):
        out, _ = self.lstm(self.emb(ids))
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1)      # average over real (non-padding) words
        return self.fc(self.drop(pooled))

def evaluate(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            logits = model(ids.to(DEVICE), mask.to(DEVICE))
            preds += logits.argmax(1).cpu().tolist()
            labels += y.tolist()
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro"), preds, labels

def train(model, data, epochs, lr, batch_size):
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(DEVICE); start = time.time()
    for ep in range(epochs):
        model.train(); total = 0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward(); opt.step()
            total += loss.item()
        print(f"  epoch {ep+1}/{epochs}  loss={total/len(loader):.4f}")
    return time.time() - start      # training time in seconds

results = {}

print("Fine-tuning DistilBERT...")
bert = BertClassifier()
t = train(bert, train_data, BERT_EPOCHS, lr=3e-5, batch_size=32)
acc, f1, bert_preds, labels = evaluate(bert, test_loader)
results["BERT"] = {"accuracy": acc, "macro_f1": f1, "train_seconds": t}
print(f"BERT  accuracy={acc:.3f}  f1={f1:.3f}  time={t/60:.1f} min")

print("Training LSTM from scratch...")
lstm = LSTMClassifier()
t = train(lstm, train_data, LSTM_EPOCHS, lr=2e-3, batch_size=64)
acc, f1, lstm_preds, _ = evaluate(lstm, test_loader)
results["LSTM"] = {"accuracy": acc, "macro_f1": f1, "train_seconds": t}
print(f"LSTM  accuracy={acc:.3f}  f1={f1:.3f}  time={t/60:.1f} min")

small = Subset(train_data, range(1000))
for name, model_fn, ep, lr, bs in [("BERT", BertClassifier, 3, 3e-5, 16),
                                   ("LSTM", LSTMClassifier, 5, 2e-3, 32)]:
    m = model_fn()
    train(m, small, ep, lr, bs)
    acc, _, _, _ = evaluate(m, test_loader)
    results[name]["accuracy_with_1000_reviews"] = acc
    print(f"{name} trained on only 1000 reviews: accuracy={acc:.3f}")

def deployment_stats(model, name):
    model.to("cpu").eval()
    params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), f"{OUT_DIR}/tmp.pt")
    size_mb = os.path.getsize(f"{OUT_DIR}/tmp.pt") / 1e6; os.remove(f"{OUT_DIR}/tmp.pt")
    ids, mask, _ = test_data[0]; ids, mask = ids.unsqueeze(0), mask.unsqueeze(0)
    with torch.no_grad():
        model(ids, mask)                                  # warm-up
        t0 = time.time()
        for _ in range(30): model(ids, mask)
    ms = (time.time() - t0) / 30 * 1000
    results[name].update({"parameters_millions": params/1e6, "size_mb": size_mb, "cpu_latency_ms": ms})
    print(f"{name}: {params/1e6:.1f}M params | {size_mb:.0f} MB | {ms:.1f} ms per review on CPU")

deployment_stats(bert, "BERT")
deployment_stats(lstm, "LSTM")

# 1) numbers
json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=2)

# 2) detailed report per class
with open(f"{OUT_DIR}/classification_reports.txt", "w") as f:
    f.write("BERT\n" + classification_report(labels, bert_preds, target_names=CLASS_NAMES))
    f.write("\nLSTM\n" + classification_report(labels, lstm_preds, target_names=CLASS_NAMES))

# 3) comparison chart
names = ["BERT", "LSTM"]
plt.figure(figsize=(6, 4))
plt.bar(names, [results[n]["accuracy"] for n in names], width=0.35, label="full data")
plt.bar([x + ".." for x in names], [0, 0], alpha=0)  # (spacer, ignore)
plt.axhline(0.80, color="red", linestyle="--", label="80% target")
plt.ylabel("Test accuracy"); plt.title("BERT vs LSTM"); plt.ylim(0, 1); plt.legend()
plt.savefig(f"{OUT_DIR}/accuracy_comparison.png", dpi=150, bbox_inches="tight")

# 4) confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, n, p in zip(ax, names, [bert_preds, lstm_preds]):
    a.imshow(confusion_matrix(labels, p), cmap="Blues"); a.set_title(n)
    a.set_xticks(range(NUM_CLASSES)); a.set_xticklabels(CLASS_NAMES)
    a.set_yticks(range(NUM_CLASSES)); a.set_yticklabels(CLASS_NAMES)
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")

# 5) model weights (do NOT push these to GitHub, see note below)
os.makedirs("models", exist_ok=True)
torch.save(bert.state_dict(), "models/bert.pt"); torch.save(lstm.state_dict(), "models/lstm.pt")
print(json.dumps(results, indent=2))