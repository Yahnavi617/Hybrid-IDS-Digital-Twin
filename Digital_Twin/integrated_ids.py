import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt

# =========================
# UNSW
# =========================

unsw_model = joblib.load(
    "unsw_model_hybrid.pkl"
)

unsw_scaler = joblib.load(
    "unsw_scaler_hybrid.pkl"
)

unsw_iforest = joblib.load(
    "isolation_forest.pkl"
)

# =========================
# CICIDS
# =========================

cicids_model = joblib.load(
    "cicids_model.pkl"
)

cicids_scaler = joblib.load(
    "cicids_scaler.pkl"
)

cicids_iforest = joblib.load(
    "cicids_iforest.pkl"
)

# =========================
# NSL-KDD
# =========================

nsl_model = joblib.load(
    "nslkdd_model.pkl"
)

nsl_scaler = joblib.load(
    "nslkdd_scaler.pkl"
)

nsl_iforest = joblib.load(
    "nslkdd_iforest.pkl"
)

print("All Models Loaded")

def predict_unsw(features):

    x = np.array(features).reshape(1,-1)

    x_scaled = unsw_scaler.transform(x)

    score = unsw_iforest.decision_function(
        x_scaled
    )

    anomaly_flag = (
        score < 0
    ).astype(int)

    x_hybrid = np.column_stack(
        (
            x_scaled,
            score,
            anomaly_flag
        )
    )

    pred = unsw_model.predict(
        x_hybrid
    )

    return pred[0]

def predict_cicids(features):

    x = np.array(features).reshape(1,-1)

    x_scaled = cicids_scaler.transform(x)

    score = cicids_iforest.decision_function(
        x_scaled
    )

    x_hybrid = np.column_stack(
        (
            x_scaled,
            score
        )
    )

    pred = cicids_model.predict(
        x_hybrid
    )

    return pred[0]

def predict_nsl(features):

    x = np.array(features).reshape(1,-1)

    x_scaled = nsl_scaler.transform(x)

    score = nsl_iforest.decision_function(
        x_scaled
    )

    x_hybrid = np.column_stack(
        (
            x_scaled,
            score
        )
    )

    pred = nsl_model.predict(
        x_hybrid
    )

    return pred[0]

def integrated_ids(
        dataset_type,
        features
):

    if dataset_type == "UNSW":

        return predict_unsw(
            features
        )

    elif dataset_type == "CICIDS":

        return predict_cicids(
            features
        )

    elif dataset_type == "NSL":

        return predict_nsl(
            features
        )

    else:

        raise ValueError(
            "Unknown Dataset"
        )
    
def draw_digital_twin(
        attacked_nodes
):

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
        ("IoT_1","Router"),
        ("IoT_2","Router"),
        ("Router","Firewall"),
        ("Firewall","Server"),
        ("Server","Database"),
        ("Client","Firewall")
    ]

    G.add_edges_from(edges)

    colors = []

    for node in G.nodes():

        if node in attacked_nodes:

            colors.append("red")

        else:

            colors.append("green")

    plt.figure(figsize=(10,7))

    pos = nx.spring_layout(
        G,
        seed=42
    )

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=3000
    )

    plt.title(
        "Integrated Hybrid IDS Digital Twin"
    )

    plt.show()

print("\nIntegrated IDS Ready")

draw_digital_twin(
    ["IoT_2", "Server"]
)