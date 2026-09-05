import numpy as np, torch, torch.nn.functional as F
from followup_v4 import load_data, train_model, ece, SEEDS

for hidden in [4, 8, 16]:
    pres = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed)
        model = train_model(hidden, X_train, y_train, label_smoothing=0.1)
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(X_test), dim=1)
        pres.append(ece(probs, y_test))
    print(hidden, "mean pre-ECE (isolated retrain):", np.mean(pres))
