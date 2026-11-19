import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# LOAD MODELS
# =====================================================

# -------------------------
# UNSW
# -------------------------
unsw_model = joblib.load("unsw_model_hybrid.pkl")
unsw_scaler = joblib.load("unsw_scaler_hybrid.pkl")
unsw_iforest = joblib.load("isolation_forest.pkl")

# -------------------------
# CICIDS
# -------------------------
cicids_model = joblib.load("cicids_model.pkl")
cicids_scaler = joblib.load("cicids_scaler.pkl")
cicids_iforest = joblib.load("cicids_iforest.pkl")

# -------------------------
# NSL-KDD
# -------------------------
nsl_model = joblib.load("nslkdd_model.pkl")
nsl_scaler = joblib.load("nslkdd_scaler.pkl")
nsl_iforest = joblib.load("nslkdd_iforest.pkl")

print("\nAll Models Loaded Successfully")


# =====================================================
# DIGITAL TWIN GRAPH
# =====================================================
def draw_digital_twin(attacked_nodes):
    G = nx.Graph()

    nodes = [
        "IoT_1",
        "IoT_2",
        "Router",
        "Firewall",
        "Server",
        "Database",
        "Client"
    ]

    G.add_nodes_from(nodes)

    edges = [
        ("IoT_1", "Router"),
        ("IoT_2", "Router"),
        ("Router", "Firewall"),
        ("Firewall", "Server"),
        ("Server", "Database"),
        ("Client", "Firewall")
    ]

    G.add_edges_from(edges)

    colors = []
    for node in G.nodes():
        if node in attacked_nodes:
            colors.append("red")
        else:
            colors.append("green")

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=3000,
        font_weight="bold"
    )

    plt.title("Integrated Hybrid IDS Digital Twin")
    plt.savefig("integrated_digital_twin.png")
    plt.show()


# =====================================================
# UNSW TESTING EVALUATION
# =====================================================
print("\n==============================")
print("UNSW TESTING")
print("==============================")

unsw_df = pd.read_parquet("dataset/UNSW_NB15_testing-set.parquet")

X_unsw = unsw_df.drop(columns=["label", "attack_cat"])
y_unsw = unsw_df["label"]

# Encode categorical columns
for col in ["proto", "service", "state"]:
    le = LabelEncoder()
    X_unsw[col] = le.fit_transform(X_unsw[col].astype(str))

# Scale full official testing set
X_unsw_scaled = unsw_scaler.transform(X_unsw)

# Isolation Forest score
unsw_score = unsw_iforest.decision_function(X_unsw_scaled)

# IMPORTANT:
# Match hybrid feature structure used by the UNSW training pipeline
# If your UNSW training file used score + flag, keep both.
# If it used only score, then remove unsw_flag below.
Q1 = np.percentile(unsw_score, 25)
Q3 = np.percentile(unsw_score, 75)
QD = (Q3 - Q1) / 2
dynamic_threshold = Q1 - (1.5 * QD)

unsw_flag = (unsw_score < dynamic_threshold).astype(int)

X_unsw_hybrid = np.column_stack((
    X_unsw_scaled,
    unsw_score,
    unsw_flag
))

unsw_pred = unsw_model.predict(X_unsw_hybrid)

unsw_acc = accuracy_score(y_unsw, unsw_pred)
unsw_precision = precision_score(y_unsw, unsw_pred, average="weighted", zero_division=0)
unsw_recall = recall_score(y_unsw, unsw_pred, average="weighted", zero_division=0)
unsw_f1 = f1_score(y_unsw, unsw_pred, average="weighted", zero_division=0)

print("\nUNSW Testing Accuracy:", round(unsw_acc * 100, 2), "%")
print("UNSW Precision:", round(unsw_precision * 100, 2), "%")
print("UNSW Recall:", round(unsw_recall * 100, 2), "%")
print("UNSW F1-Score:", round(unsw_f1 * 100, 2), "%")


# =====================================================
# CICIDS TESTING EVALUATION
# =====================================================
print("\n==============================")
print("CICIDS TESTING")
print("==============================")

cicids_df = pd.read_csv("dataset/CICIDS.csv")
cicids_df.columns = cicids_df.columns.str.strip()

label_encoder = LabelEncoder()
cicids_df["Label"] = label_encoder.fit_transform(cicids_df["Label"])

X_cicids = cicids_df.drop("Label", axis=1)
y_cicids = cicids_df["Label"]

X_cicids.replace([np.inf, -np.inf], np.nan, inplace=True)
X_cicids.fillna(0, inplace=True)

# Recreate SAME split as training file and evaluate ONLY test set
X_train_cicids, X_test_cicids, y_train_cicids, y_test_cicids = train_test_split(
    X_cicids,
    y_cicids,
    test_size=0.2,
    random_state=42
)

X_test_cicids_scaled = cicids_scaler.transform(X_test_cicids)

test_score_cicids = cicids_iforest.decision_function(X_test_cicids_scaled)

X_test_cicids_hybrid = np.column_stack((
    X_test_cicids_scaled,
    test_score_cicids
))

cicids_pred = cicids_model.predict(X_test_cicids_hybrid)

cicids_acc = accuracy_score(y_test_cicids, cicids_pred)
cicids_precision = precision_score(y_test_cicids, cicids_pred, average="weighted", zero_division=0)
cicids_recall = recall_score(y_test_cicids, cicids_pred, average="weighted", zero_division=0)
cicids_f1 = f1_score(y_test_cicids, cicids_pred, average="weighted", zero_division=0)

print("\nCICIDS Testing Accuracy:", round(cicids_acc * 100, 2), "%")
print("CICIDS Precision:", round(cicids_precision * 100, 2), "%")
print("CICIDS Recall:", round(cicids_recall * 100, 2), "%")
print("CICIDS F1-Score:", round(cicids_f1 * 100, 2), "%")


# =====================================================
# NSL-KDD TESTING EVALUATION
# =====================================================
print("\n==============================")
print("NSL-KDD TESTING")
print("==============================")

column_names = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell',
    'su_attempted', 'num_root',
    'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

nsl_df = pd.read_csv(
    "dataset/KDDTest+.txt",
    names=column_names
)

nsl_df["label"] = nsl_df["label"].apply(lambda x: 0 if x == "normal" else 1)

for col in ["protocol_type", "service", "flag"]:
    le = LabelEncoder()
    nsl_df[col] = le.fit_transform(nsl_df[col].astype(str))

X_nsl = nsl_df.drop(columns=["label", "difficulty"])
y_nsl = nsl_df["label"]

X_nsl_scaled = nsl_scaler.transform(X_nsl)

nsl_score = nsl_iforest.decision_function(X_nsl_scaled)

X_nsl_hybrid = np.column_stack((
    X_nsl_scaled,
    nsl_score
))

nsl_pred = nsl_model.predict(X_nsl_hybrid)

nsl_acc = accuracy_score(y_nsl, nsl_pred)
nsl_precision = precision_score(y_nsl, nsl_pred, average="weighted", zero_division=0)
nsl_recall = recall_score(y_nsl, nsl_pred, average="weighted", zero_division=0)
nsl_f1 = f1_score(y_nsl, nsl_pred, average="weighted", zero_division=0)

print("\nNSL-KDD Testing Accuracy:", round(nsl_acc * 100, 2), "%")
print("NSL-KDD Precision:", round(nsl_precision * 100, 2), "%")
print("NSL-KDD Recall:", round(nsl_recall * 100, 2), "%")
print("NSL-KDD F1-Score:", round(nsl_f1 * 100, 2), "%")


# =====================================================
# INTEGRATED HYBRID IDS
# =====================================================
print("\n==============================")
print("INTEGRATED HYBRID IDS")
print("==============================")

# Combine true testing labels only
y_true = np.concatenate([
    y_unsw.values,
    y_test_cicids.values,
    y_nsl.values
])

# Combine predictions from testing data only
y_pred = np.concatenate([
    unsw_pred,
    cicids_pred,
    nsl_pred
])


# =====================================================
# INTEGRATED METRICS
# =====================================================
integrated_acc = accuracy_score(y_true, y_pred)
integrated_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
integrated_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
integrated_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\nIntegrated Accuracy:", round(integrated_acc * 100, 2), "%")
print("Integrated Precision:", round(integrated_precision * 100, 2), "%")
print("Integrated Recall:", round(integrated_recall * 100, 2), "%")
print("Integrated F1-Score:", round(integrated_f1 * 100, 2), "%")


# =====================================================
# CLASSIFICATION REPORT
# =====================================================
print("\n==============================")
print("INTEGRATED CLASSIFICATION REPORT")
print("==============================")

print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0
))


# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)

disp.plot()
plt.title("Integrated Hybrid IDS Confusion Matrix")
plt.savefig("integrated_confusion_matrix.png")
plt.show()


# =====================================================
# DIGITAL TWIN GRAPH
# =====================================================
draw_digital_twin(["IoT_2", "Server"])

print("\nIntegrated Hybrid IDS Execution Completed")