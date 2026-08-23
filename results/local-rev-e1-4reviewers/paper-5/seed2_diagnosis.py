"""
Diagnose seed 2's 7.7%-accuracy outcome (round-3 reviewer question, raised
independently by 3/4 reviewers): is it a bad-init training-instability basin,
or just a slower-converging seed that would catch up with more steps?
Logs the training loss curve every 20 steps for seeds 0-4 (identical setup to
multiseed.py) and additionally trains seed 2 for 260 extra steps (520 total)
to see if it recovers.
"""
import random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = list("0123456789+=")
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
IN_LEN, OUT_LEN = 6, 3
SEQ_LEN = IN_LEN + OUT_LEN
BATCH = 128


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, seq_len=SEQ_LEN, d=64, nhead=4, layers=2, ff=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                                batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.ln(h)
        return self.head(h)


def make_example(rng):
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    s = a + b
    return f"{a:02d}+{b:02d}={s:03d}"


def encode(text):
    return [STOI[c] for c in text]


def train_curve(seed, steps):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def batch(n):
        return torch.tensor([encode(make_example(random)) for _ in range(n)], dtype=torch.long)

    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    curve = []
    for step in range(steps):
        x = batch(BATCH)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits[:, IN_LEN - 1:].reshape(-1, VOCAB), tgt[:, IN_LEN - 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0 or step == steps - 1:
            curve.append((step, round(loss.item(), 4)))
    return model, curve


def eval_acc(model, seed, n=400):
    random.seed(seed + 9999); np.random.seed(seed + 9999)
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            text = make_example(random)
            true_ans = text.split("=")[1]
            prefix = encode(text)[:IN_LEN]
            seq = list(prefix)
            for _ in range(OUT_LEN):
                x = torch.tensor([seq], dtype=torch.long)
                logits = model(x)
                nxt = int(logits[0, -1].argmax().item())
                seq.append(nxt)
            ans = "".join(str(CHARS[i]) if i < 10 else CHARS[i] for i in seq[IN_LEN:])
            ans = "".join(CHARS[i] for i in seq[IN_LEN:])
            correct += int(ans == true_ans)
    return correct / n


out = {}
for seed in [0, 1, 2, 3, 4]:
    model, curve = train_curve(seed, 260)
    acc260 = eval_acc(model, seed)
    out[f"seed{seed}"] = {"loss_curve": curve, "acc_at_260steps": acc260}
    print(f"seed {seed}: acc@260={acc260:.3f}  loss_curve={curve}")

# extra steps for seed 2 to see if it's just slow, not stuck
model2, curve2_more = train_curve(2, 520)
acc2_520 = eval_acc(model2, 2)
out["seed2_extended_520steps"] = {"loss_curve": curve2_more, "acc_at_520steps": acc2_520}
print(f"seed 2 extended to 520 steps: acc={acc2_520:.3f}")

with open("seed2_diagnosis.json", "w") as f:
    json.dump(out, f, indent=2)
