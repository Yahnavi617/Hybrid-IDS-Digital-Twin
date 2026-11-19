import pandas as pd
import numpy as np
import joblib
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# Load dataset
df = pd.read_parquet(
    r"dataset/UNSW_NB15_testing-set.parquet"
)

# Features and labels
X = df.drop(columns=["label", "attack_cat"])
y = df["label"]

# Encode categorical columns
for col in ["proto", "service", "state"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Isolation Forest
iso = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)

iso.fit(X_train_scaled)

# Anomaly score
train_score = iso.decision_function(X_train_scaled)
test_score = iso.decision_function(X_test_scaled)


Q1 = np.percentile(train_score, 25)
Q3 = np.percentile(train_score, 75)

QD = (Q3 - Q1) / 2

k = 1.5

dynamic_threshold = Q1 - (k * QD)

print("\nQ1 :", Q1)
print("Q3 :", Q3)
print("Quartile Deviation :", QD)
print("Dynamic Threshold :", dynamic_threshold)

anomaly_flag_train = (
    train_score < dynamic_threshold
).astype(int)

anomaly_flag_test = (
    test_score < dynamic_threshold
).astype(int)


# Hybrid features
X_train_hybrid = np.column_stack(
    (
        X_train_scaled,
        train_score,
        anomaly_flag_train
    )
)

X_test_hybrid = np.column_stack(
    (
        X_test_scaled,
        test_score,
        anomaly_flag_test
    )
)

# XGBoost
model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_hybrid, y_train)

cv_scores = cross_val_score(
    model,
    X_train_hybrid,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nMean CV Accuracy:",
      round(cv_scores.mean()*100,2), "%")

print("CV Standard Deviation:",
      round(cv_scores.std()*100,2), "%")

plt.figure(figsize=(7,5))

plt.plot(
    range(1,6),
    cv_scores * 100,
    marker='o'
)

plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("5-Fold Cross Validation")

plt.grid(True)

plt.savefig("cross_validation.png")
plt.show()

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_train_hybrid,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8,6))

plt.plot(
    train_sizes,
    train_mean * 100,
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    train_sizes,
    val_mean * 100,
    marker='s',
    label='Validation Accuracy'
)

plt.xlabel("Training Samples")
plt.ylabel("Accuracy (%)")
plt.title("Learning Curve")

plt.legend()
plt.grid(True)

plt.savefig("learning_curve.png")
plt.show()

# Training Accuracy
train_pred = model.predict(X_train_hybrid)
train_acc = accuracy_score(y_train, train_pred)


# Prediction
pred = model.predict(X_test_hybrid)

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)

disp.plot()

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()


y_prob = model.predict_proba(X_test_hybrid)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()


# Accuracy
acc = accuracy_score(y_test, pred)

print("\nTraining Accuracy:",
      round(train_acc * 100, 2), "%")

print("Testing Accuracy:",
      round(acc * 100, 2), "%")

gap = train_acc - acc

print(
    "Generalization Gap:",
    round(gap * 100, 2),
    "%"
)

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

plt.figure(figsize=(6,4))

plt.bar(
    ["Training", "Testing"],
    [train_acc * 100, acc * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("Overfitting Check")

plt.savefig("overfitting_check.png")
plt.show()

# Feature Importance Graph

importance = model.feature_importances_

feature_names = list(X.columns)
feature_names.append("IsolationForest_Score")
feature_names.append("QD_Anomaly_Flag")

# Safety check
while len(feature_names) < len(importance):
    feature_names.append(f"Extra_Feature_{len(feature_names)}")

plt.figure(figsize=(15,6))

plt.bar(feature_names, importance)

plt.xticks(rotation=90)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Hybrid IDS Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance_named.png")
plt.show()


# Save model
joblib.dump(model, "unsw_model_hybrid.pkl")
joblib.dump(scaler, "unsw_scaler_hybrid.pkl")
joblib.dump(iso, "isolation_forest.pkl")

print("\nHybrid IDS Model Saved Successfully")

import networkx as nx

# -----------------------------
# DIGITAL TWIN NODE GRAPH
# -----------------------------

G = nx.Graph()

# Create nodes
nodes = [
    "IoT_Device_1",
    "IoT_Device_2",
    "Server",
    "Firewall",
    "Router",
    "Database",
    "Client_PC"
]

G.add_nodes_from(nodes)

# Create connections
edges = [
    ("IoT_Device_1", "Router"),
    ("IoT_Device_2", "Router"),
    ("Router", "Firewall"),
    ("Firewall", "Server"),
    ("Server", "Database"),
    ("Client_PC", "Firewall")
]

G.add_edges_from(edges)

# Example attacked nodes
attacked_nodes = [
    "IoT_Device_2",
    "Server"
]

# Node colors
node_colors = []

for node in G.nodes():

    if node in attacked_nodes:
        node_colors.append("red")

    else:
        node_colors.append("green")

# Draw graph
plt.figure(figsize=(10,7))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    node_size=3000,
    font_size=10,
    font_weight="bold"
)

plt.title("Digital Twin Network Graph")

plt.savefig("digital_twin_node_graph.png")

plt.show() 