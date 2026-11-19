import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import networkx as nx

from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from xgboost import XGBClassifier


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "dataset/CICIDS.csv"
)

print(df.head())


# =====================================================
# PREPROCESSING
# =====================================================

df.columns = df.columns.str.strip()

label_encoder = LabelEncoder()

df["Label"] = label_encoder.fit_transform(
    df["Label"]
)

X = df.drop("Label", axis=1)

y = df["Label"]


# =====================================================
# HANDLE INFINITE VALUES
# =====================================================

X.replace([np.inf, -np.inf], np.nan, inplace=True)

X.fillna(0, inplace=True)


# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# ISOLATION FOREST
# =====================================================

iforest = IsolationForest(
    contamination=0.1,
    random_state=42
)

iforest.fit(X_train)


# =====================================================
# DYNAMIC THRESHOLD
# =====================================================

scores = iforest.decision_function(X_train)

Q1 = np.percentile(scores, 25)
Q3 = np.percentile(scores, 75)

QD = (Q3 - Q1) / 2

threshold = Q1 - (1.5 * QD)

print("\nQ1 :", Q1)
print("Q3 :", Q3)
print("Quartile Deviation :", QD)
print("Dynamic Threshold :", threshold)


# =====================================================
# HYBRID FEATURES
# =====================================================

train_scores = iforest.decision_function(X_train)
test_scores = iforest.decision_function(X_test)

X_train_hybrid = np.column_stack(
    (X_train, train_scores)
)

X_test_hybrid = np.column_stack(
    (X_test, test_scores)
)


# =====================================================
# XGBOOST
# =====================================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train_hybrid, y_train)


# =====================================================
# PREDICTIONS
# =====================================================

pred = model.predict(X_test_hybrid)


# =====================================================
# ACCURACY
# =====================================================

acc = accuracy_score(y_test, pred)

print("\nAccuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:\n")

print(classification_report(y_test, pred))


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("CICIDS Confusion Matrix")

plt.savefig("cicids_confusion_matrix.png")

plt.show()


# =====================================================
# ROC CURVE
# =====================================================

y_prob = model.predict_proba(
    X_test_hybrid
)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("CICIDS ROC Curve")

plt.legend()

plt.savefig("cicids_roc_curve.png")

plt.show()


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_

feature_names = list(X.columns)

feature_names.append("IForest_Score")

plt.figure(figsize=(12,6))

plt.bar(feature_names, importance)

plt.xticks(rotation=90)

plt.title("CICIDS Feature Importance")

plt.tight_layout()

plt.savefig("cicids_feature_importance.png")

plt.show()


# =====================================================
# CROSS VALIDATION
# =====================================================

scores = cross_val_score(
    model,
    X_train_hybrid,
    y_train,
    cv=5
)

plt.figure(figsize=(8,5))

plt.plot(scores, marker='o')

plt.title("CICIDS Cross Validation")

plt.xlabel("Fold")

plt.ylabel("Accuracy")

plt.savefig("cicids_cross_validation.png")

plt.show()


# =====================================================
# LEARNING CURVE
# =====================================================

train_sizes, train_scores, test_scores = learning_curve(
    model,
    X_train_hybrid,
    y_train,
    cv=5
)

train_mean = np.mean(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,5))

plt.plot(train_sizes, train_mean, label="Training")

plt.plot(train_sizes, test_mean, label="Validation")

plt.legend()

plt.title("CICIDS Learning Curve")

plt.xlabel("Training Size")

plt.ylabel("Score")

plt.savefig("cicids_learning_curve.png")

plt.show()


# =====================================================
# SAVE MODELS
# =====================================================

joblib.dump(
    model,
    "cicids_model.pkl"
)

joblib.dump(
    scaler,
    "cicids_scaler.pkl"
)

joblib.dump(
    iforest,
    "cicids_iforest.pkl"
)

print("\nCICIDS Hybrid Model Saved Successfully")

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