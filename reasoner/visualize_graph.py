import re
import os
import networkx as nx
import matplotlib.pyplot as plt

# Input string with the given facts
input_str = ("link(support(node(afactor(7)),node(root(1)))) "
             "link(support(node(afactor(5)),node(root(1)))) "
             "link(support(node(afactor(6)),node(root(1)))) "
             "link(attack(node(afactor(6)),node(afactor(4)))) "
             "link(attack(node(afactor(6)),node(afactor(3)))) "
             "link(attack(node(afactor(6)),node(afactor(2)))) "
             "link(attack(node(afactor(6)),node(afactor(1)))) "
             "link(attack(node(afactor(5)),node(afactor(6)))) "
             "link(attack(node(afactor(4)),node(afactor(6)))) "
             "link(attack(node(afactor(3)),node(afactor(6)))) "
             "link(attack(node(afactor(2)),node(afactor(6)))) "
             "link(attack(node(afactor(1)),node(afactor(6))))")

# Regular expression to parse each fact:
#   Group 1: relation type (support or attack)
#   Group 2: source node type (afactor or root)
#   Group 3: source node number
#   Group 4: target node type (afactor or root)
#   Group 5: target node number
pattern = r'link\((support|attack)\(node\((afactor|root)\((\d+)\)\),node\((afactor|root)\((\d+)\)\)\)\)'

# Find all matches in the string
matches = re.findall(pattern, input_str)

# Create a directed graph
G = nx.DiGraph()

# Process each match and add an edge accordingly.
for rel_type, src_type, src_num, tgt_type, tgt_num in matches:
    # Construct node labels by concatenating type and number (e.g., "afactor7" or "root1")
    source = f"{src_type}{src_num}"
    target = f"{tgt_type}{tgt_num}"
    # Add the edge with a 'type' attribute to determine the color later
    G.add_edge(source, target, type=rel_type)

# Determine edge colors based on the 'type' attribute: green for support, red for attack.
edge_colors = [
    "green" if G[u][v]["type"] == "support" else "red"
    for u, v in G.edges()
]

# Compute positions for nodes in a visually appealing layout
pos = nx.spring_layout(G)

# Draw nodes, labels, and edges with their respective colors
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowstyle="-|>", arrowsize=15)

# Remove axes
plt.axis("off")

# Save the graph to a file named "graph.png" in the same folder as this code file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "graph.png")
plt.savefig(output_path)
plt.show()
