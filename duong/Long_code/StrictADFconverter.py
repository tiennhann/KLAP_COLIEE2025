from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",  # Changed to include provider prefix
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)

class ADFStructure(BaseModel):
    outcomes: List[str] = Field(description="Outcome is the agreement with article.")
    abstract_factors: List[str] = Field(description="A list of the abstract factors of the input article.")
    baselevel_factors: List[str] = Field(description="A list of the baselvel factors of the input article.")
    links: List[Tuple[str, Literal["support", "attack"], str]]= Field(description="A list of the links between the factors and the outcomes of the input article.")


class ADFConversionTask(dspy.Signature):
    """Signature for converting legal text to an ADF structure"""
    task_definition=dspy.InputField(desc="The definition for the task.")
    input_article=dspy.InputField(desc="The input article to convert into an ADF structure.")
    adf_structure:ADFStructure=dspy.OutputField()

class ADFConverter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.agent = dspy.ChainOfThought(ADFConversionTask)

    def forward(self, article_text, task_definition):
        return self.agent(input_article=article_text, task_definition=task_definition)

if __name__ == "__main__":
    with open("./framework.txt", "r") as f:
        task_definition = "".join(f.readlines())
    with open("./article_3.txt", "r")  as f:
        article = "".join(f.readlines())
    
    module = ADFConverter()
    output = module.forward(article_text=article, task_definition=task_definition)
    print(output)
    # Example structure of output (this should be replaced with actual output structure)
    output = {
        "nodes": {
            "baselevel_factors": output.adf_structure,
            "abstract_factors": ["node3", "node4"],
            "outcomes": ["node5"]
        },
        "edges": [
            {"source": "node1", "target": "node3", "type": "support"},
            {"source": "node2", "target": "node4", "type": "attack"},
            {"source": "node3", "target": "node5", "type": "support"}
        ]
    }

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with categories
    for category, nodes in output["nodes"].items():
        for node in nodes:
            G.add_node(node, category=category)

    # Add edges with types
    for edge in output["edges"]:
        G.add_edge(edge["source"], edge["target"], type=edge["type"])

    # Define colors for node categories

    # Define colors for edge types
    edge_colors = {
        "support": "green",
        "attack": "red"
    }


    # Get edge colors
    edge_color_list = [edge_colors[G.edges[edge]["type"]] for edge in G.edges]

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_shape='s', node_color='white',edge_color=edge_color_list, node_size=3000, font_size=10, font_weight='bold')
    plt.savefig("graph_output.png", format="png")
    # plt.show()