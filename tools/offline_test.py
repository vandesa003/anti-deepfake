import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, recall_score, precision_score
import matplotlib.pyplot as plt

report = pd.read_csv("../logs/report.csv")
report["label"] = report["label"].replace(["FAKE", "REAL"], [1, 0])
report["label"].hist()
report["score"].hist()
plt.show()
label = report["label"].values
score = report["score"].values
print(label)
kaggle = log_loss(label, np.clip(score, 0.1, 0.9), eps=1e-7)
print(kaggle)
