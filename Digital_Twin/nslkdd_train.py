import pandas as pd
import numpy as np
import joblib
import networkx as nx

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
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# =====================================================
# LOAD DATASET
# =====================================================

column_names = [
'duration','protocol_type','service','flag','src_bytes',
'dst_bytes','land','wrong_fragment','urgent','hot',
'num_failed_logins','logged_in','num_compromised',
'root_shell','su_attempted','num_root',
'num_file_creations','num_shells','num_access_files',
'num_outbound_cmds','is_host_login',
'is_guest_login','count','srv_count',
'serror_rate','srv_serror_rate','rerror_rate',
'srv_rerror_rate','same_srv_rate',
'diff_srv_rate','srv_diff_host_rate',
'dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate',
'label','difficulty'
]

train_df = pd.read_csv(
    "Digital_Twin/dataset/KDDTrain+.txt",
    names=column_names
)

test_df = pd.read_csv(
    "Digital_Twin/dataset/KDDTest+.txt",
    names=column_names
)

df = pd.concat([train_df, test_df], ignore_index=True)

# =====================================================
# BINARY LABEL
# =====================================================

df['label'] = df['label'].apply(
    lambda x: 0 if x == 'normal' else 1
)

# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================

categorical_cols = [
    'protocol_type',
    'service',
    'flag'
]

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le

# =====================================================
# FEATURES / LABEL
# =====================================================

X = df.drop(
    columns=['label','difficulty']
)

y = df['label']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =====================================================
# ISOLATION FOREST
# =====================================================

iforest = IsolationForest(
    contamination=0.05,
    random_state=42
)

iforest.fit(X_train_scaled)

train_scores = iforest.decision_function(
    X_train_scaled
)

test_scores = iforest.decision_function(
    X_test_scaled
)

# =====================================================
# DYNAMIC THRESHOLD
# =====================================================

Q1 = np.percentile(test_scores, 25)

Q3 = np.percentile(test_scores, 75)

QD = (Q3 - Q1) / 2

threshold = Q1 - (1.5 * QD)

print("\nQ1 :", Q1)
print("Q3 :", Q3)
print("Quartile Deviation :", QD)
print("Dynamic Threshold :", threshold)

# =====================================================
# HYBRID FEATURES
# =====================================================

X_train_hybrid = np.hstack([
    X_train_scaled,
    train_scores.reshape(-1,1)
])

X_test_hybrid = np.hstack([
    X_test_scaled,
    test_scores.reshape(-1,1)
])

# =====================================================
# XGBOOST
# =====================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train_hybrid,
    y_train
)

# =====================================================
# PREDICTIONS
# =====================================================

pred = model.predict(
    X_test_hybrid
)

# =====================================================
# OVERFITTING CHECK
# =====================================================

train_pred = model.predict(
    X_train_hybrid
)

train_acc = accuracy_score(
    y_train,
    train_pred
)

test_acc = accuracy_score(
    y_test,
    pred
)

print("\nTraining Accuracy:",
      round(train_acc*100,2), "%")

print("Testing Accuracy:",
      round(test_acc*100,2), "%")

if train_acc - test_acc > 0.05:
    print("⚠ Overfitting Detected")
else:
    print("✓ No Significant Overfitting")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal","Attack"]
)

disp.plot()

plt.title("NSL-KDD Confusion Matrix")

plt.savefig(
    "nslkdd_confusion_matrix.png"
)

plt.show()

# =====================================================
# ROC CURVE
# =====================================================

y_prob = model.predict_proba(
    X_test_hybrid
)[:,1]

fpr,tpr,_ = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC={roc_auc:.4f}"
)

plt.plot([0,1],[0,1],'--')

plt.xlabel("FPR")
plt.ylabel("TPR")

plt.title("NSL-KDD ROC Curve")

plt.legend()

plt.savefig(
    "nslkdd_roc_curve.png"
)

plt.show()

# =====================================================
# ACCURACY
# =====================================================

acc = accuracy_score(
    y_test,
    pred
)

print("\nAccuracy:",
      round(acc*100,2), "%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred
    )
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_

feature_names = list(X.columns)

feature_names.append(
    "IF_Anomaly_Score"
)

plt.figure(figsize=(12,6))

plt.bar(
    feature_names,
    importance
)

plt.xticks(rotation=90)

plt.title(
    "NSL-KDD Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "nslkdd_feature_importance.png"
)

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

plt.figure(figsize=(6,4))

plt.plot(
    range(1,6),
    scores,
    marker='o'
)

plt.xlabel("Fold")

plt.ylabel("Accuracy")

plt.title(
    "NSL-KDD Cross Validation"
)

plt.savefig(
    "nslkdd_cross_validation.png"
)

plt.show()

print(
    "Cross Validation Mean:",
    scores.mean()
)

# =====================================================
# LEARNING CURVE
# =====================================================

train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    model,
    X_train_hybrid,
    y_train,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores_lc.mean(axis=1)

test_mean = test_scores_lc.mean(axis=1)

plt.figure(figsize=(8,6))

plt.plot(
    train_sizes,
    train_mean,
    label="Training Accuracy"
)

plt.plot(
    train_sizes,
    test_mean,
    label="Validation Accuracy"
)

plt.xlabel("Training Size")

plt.ylabel("Accuracy")

plt.title(
    "NSL-KDD Learning Curve"
)

plt.legend()

plt.savefig(
    "nslkdd_learning_curve.png"
)

plt.show()

# =====================================================
# SAVE MODELS
# =====================================================

joblib.dump(
    model,
    "nslkdd_model.pkl"
)

joblib.dump(
    scaler,
    "nslkdd_scaler.pkl"
)

joblib.dump(
    iforest,
    "nslkdd_iforest.pkl"
)

joblib.dump(
    encoders,
    "nslkdd_encoders.pkl"
)

print(
    "\nNSL-KDD Hybrid IDS Model Saved Successfully"
)

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