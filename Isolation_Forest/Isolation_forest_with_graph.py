

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.tree import _tree

# Load your dataset (update the file path to your dataset)
df = pd.read_csv('data_4var.csv')  # Update with your actual dataset path

# Select only the Population column for anomaly detection
X_population = df[['Population']].dropna()

# Train Isolation Forest on the population data
clf = IsolationForest(n_estimators=2, contamination=0.1, random_state=42)
clf.fit(X_population)

# Predict anomalies (-1 means anomaly, 1 means normal)
df['Anomaly'] = clf.predict(X_population)

# Save anomalies to a CSV file
anomalies_df = df[df['Anomaly'] == -1]
anomalies_df.to_csv('anomalies_detected.csv', index=False)
print(f"Anomalies saved to 'anomalies_detected.csv'. Total anomalies: {len(anomalies_df)}")

# Dynamically select the first anomaly and normal instance from the data
X_anomaly = X_population[df['Anomaly'] == -1].iloc[0].values.reshape(1, -1)  # First anomaly instance
X_normal = X_population[df['Anomaly'] == 1].iloc[0].values.reshape(1, -1)   # First normal instance

# Get decision paths for the selected anomaly and normal instance
anomaly_path = clf.estimators_[0].decision_path(X_anomaly).toarray().flatten()
normal_path = clf.estimators_[0].decision_path(X_normal).toarray().flatten()

# Function to plot a larger tree with dynamically selected anomaly and normal nodes
def plot_larger_tree(anomaly_path, normal_path):
    fig, ax = plt.subplots(figsize=(18, 8))  # Adjusting the figure size for a larger tree

    # Define more nodes (Root, Internal, External) with positions for a deeper tree
    nodes = {
        0: (0.5, 1),        # Root node
        1: (0.25, 0.85),    # Internal node 1
        2: (0.75, 0.85),    # Internal node 2
        3: (0.15, 0.7),     # Internal node 3 (left)
        4: (0.35, 0.7),     # Internal node 4 (right)
        5: (0.65, 0.7),     # Internal node 5 (left)
        6: (0.85, 0.7),     # Internal node 6 (right)
        7: (0.1, 0.55),     # Internal node 7
        8: (0.2, 0.55),     # Internal node 8
        9: (0.3, 0.55),     # Internal node 9
        10: (0.4, 0.55),    # Internal node 10
        11: (0.6, 0.55),    # Internal node 11
        12: (0.7, 0.55),    # Internal node 12
        13: (0.8, 0.55),    # Internal node 13
        14: (0.9, 0.55),    # Internal node 14
        15: (0.1, 0.4),     # External node (Anomalous)
        16: (0.2, 0.4),     # External node (Normal)
        17: (0.3, 0.4),     # External node (Normal)
        18: (0.4, 0.4),     # External node (Normal)
        19: (0.6, 0.4),     # External node (Normal)
        20: (0.7, 0.4),     # External node (Normal)
        21: (0.8, 0.4),     # External node (Normal)
        22: (0.9, 0.4),     # External node (Normal)
    }

    # Define more edges (connections between nodes)
    edges = [
        (0, 1), (0, 2),     # Root to internal nodes
        (1, 3), (1, 4),     # Left internal splits
        (2, 5), (2, 6),     # Right internal splits
        (3, 7), (3, 8),     # Deeper left internal nodes
        (4, 9), (4, 10),    # Deeper right internal nodes
        (5, 11), (5, 12),   # Deeper left internal nodes
        (6, 13), (6, 14),   # Deeper right internal nodes
        (7, 15), (8, 16),   # External nodes for anomaly and normal
        (9, 17), (10, 18),  # External nodes for normal
        (11, 19), (12, 20), # External nodes for normal
        (13, 21), (14, 22), # External nodes for normal
    ]

    # Plot edges
    for edge in edges:
        node_from, node_to = edge
        color = 'gray'  # Default edge color
        if anomaly_path[node_to]:  # If the path follows the anomaly
            color = 'red'
        elif normal_path[node_to]:  # If the path follows the normal point
            color = 'blue'
        ax.plot([nodes[node_from][0], nodes[node_to][0]],
                [nodes[node_from][1], nodes[node_to][1]], color=color, lw=2)

    # Plot nodes (root, internal, external) based on the example
    for node, (x, y) in nodes.items():
        if node in [15, 16, 17, 18, 19, 20, 21, 22]:  # External nodes should be squares
            ax.scatter(x, y, s=100, marker='s', color='black', zorder=3)
        else:  # Internal nodes should remain as circles
            ax.scatter(x, y, s=100, color='white', edgecolor='black', zorder=3)

    # Annotate nodes
    ax.text(0.5, 1.02, 'Root', ha='center')
    for node in range(1, 15):
        ax.text(nodes[node][0], nodes[node][1] + 0.02, 'T_in', ha='center')

    ax.text(0.1, 0.37, 'T_ex (Anomalous)', ha='center', color='red')  # Anomalous external node
    for i, pos in enumerate([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], start=16):
        ax.text(pos, 0.37, 'T_ex (Normal)', ha='center', color='blue')  # Normal external nodes

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legends
    red_line = plt.Line2D([0], [0], color='red', lw=3, label='A path of anomaly')
    blue_line = plt.Line2D([0], [0], color='blue', lw=3, label='A path of normal instance')
    internal_node = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markersize=10, label='Internal node (T_in)')
    external_node = plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='External node (T_ex)')
    
    ax.legend(handles=[red_line, blue_line, internal_node, external_node], loc='upper right')

    plt.title("Isolation Tree with Anomaly and Normal Paths")
    plt.savefig('isolation_forest_A2.png') 
    plt.show()

# Plot the larger tree with dynamically selected anomaly and normal paths
plot_larger_tree(anomaly_path, normal_path)


############ Graphs #######################
