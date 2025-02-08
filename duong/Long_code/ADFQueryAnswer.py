from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field, validator
import networkx as nx
import matplotlib.pyplot as plt
import textwrap

# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)

class ADFStructure(BaseModel):
    outcomes: List[str] = Field(description="Main conclusions or outcomes.")
    abstract_factors: List[str] = Field(description="Intermediate level factors/conclusions.")
    baselevel_factors: List[str] = Field(description="Concrete facts or basic rules.")
    links: List[Tuple[str, Literal["support", "attack"], str]] = Field(
        description="Relationships between factors, showing support or attack connections."
    )

    class Config:
        validate_assignment = True

    @validator('links')
    def validate_links(cls, v, values):
        """Validate that links only connect existing nodes"""
        all_nodes = set(values.get('outcomes', []) + 
                        values.get('abstract_factors', []) + 
                        values.get('baselevel_factors', []))
        
        for source, link_type, target in v:
            if source not in all_nodes:
                raise ValueError(f"Source node '{source}' not found in factors or outcomes")
            if target not in all_nodes:
                raise ValueError(f"Target node '{target}' not found in factors or outcomes")
            if link_type not in ['support', 'attack']:
                raise ValueError(f"Invalid link type: {link_type}")
        return v

    def visualize(self):
        """Create an improved hierarchical visualization of the ADF structure"""
        G = nx.DiGraph()
        plt.figure(figsize=(15, 10))
        
        # Add nodes with different colors and positions
        pos = {}
        y_spacing = 1.0
        
        # Position outcomes at top
        for i, outcome in enumerate(self.outcomes):
            x_pos = (i - len(self.outcomes)/2) * 2
            pos[outcome] = (x_pos, 2 * y_spacing)
            G.add_node(outcome, node_type='outcome')
            
        # Position abstract factors in middle
        for i, factor in enumerate(self.abstract_factors):
            x_pos = (i - len(self.abstract_factors)/2) * 2
            pos[factor] = (x_pos, y_spacing)
            G.add_node(factor, node_type='abstract')
            
        # Position base factors at bottom
        for i, factor in enumerate(self.baselevel_factors):
            x_pos = (i - len(self.baselevel_factors)/2) * 2
            pos[factor] = (x_pos, 0)
            G.add_node(factor, node_type='baselevel')
        
        # Add edges with colors based on support/attack
        for source, link_type, target in self.links:
            G.add_edge(source, target, link_type=link_type)
        
        # Draw the graph
        node_colors = ['lightblue' if node in self.outcomes else
                        'lightgreen' if node in self.abstract_factors else
                        'lightcoral' for node in G.nodes()]
        
        edge_colors = ['green' if G[u][v]['link_type'] == 'support' else 'red'for u, v in G.edges()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20)
        
        # Draw labels with wrapped text
        labels = {node: '\n'.join(textwrap.wrap(node, width=20)) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                        label='Outcome', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                        label='Abstract Factor', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                        label='Base Factor', markersize=10),
            plt.Line2D([0], [0], color='green', label='Support'),
            plt.Line2D([0], [0], color='red', label='Attack')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("ADF Structure Visualization")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()

class ADFConversionTask(dspy.Signature):
    """Signature for converting legal text to an ADF structure"""
    task_definition = dspy.InputField(desc="The definition for the task.")
    input_article = dspy.InputField(desc="The input article to convert into an ADF structure.")
    adf_structure: ADFStructure = dspy.OutputField()

class ADFQueryCheckTask(dspy.Signature):
    """Signature for checking queries against ADF factors"""
    task_definition = dspy.InputField(desc="The task definition.")
    query = dspy.InputField(desc="The query to check against the factors.")
    factor = dspy.InputField(desc="The factor to check against.")
    output: Literal["Yes", "No", "Undecided"] = dspy.OutputField(desc="Query result")

class ADFConverter(dspy.Module):
    """Module for converting legal text to ADF structure"""
    def __init__(self):
        super().__init__()
        self.converter = dspy.ChainOfThought(ADFConversionTask)

    def forward(self, article_text: str, task_definition: str) -> ADFStructure:
        result = self.converter(
            input_article=article_text,
            task_definition=task_definition
        )
        return result.adf_structure

class ADFQueryChecker(dspy.Module):
    """Module for checking queries against ADF factors"""
    def __init__(self):
        super().__init__()
        self.query_checker = dspy.ChainOfThought(ADFQueryCheckTask)

    def check_factor(self, factor: str, query: str, task_definition: str) -> Literal["Yes", "No", "Undecided"]:
        result = self.query_checker(
            factor=factor,
            query=query,
            task_definition=task_definition
        )
        return result.output

    def check_all_factors(self, factors: List[str], query: str, task_definition: str) -> List[Tuple[str, str]]:
        return [(factor, self.check_factor(factor, query, task_definition)) 
                for factor in factors]

class ADFSystem:
    """Main system class for ADF processing"""
    def __init__(self):
        self.converter = ADFConverter()
        self.checker = ADFQueryChecker()

    def process_article_and_query(self, 
                                article_text: str, 
                                task_definition: str, 
                                query: str) -> Tuple[ADFStructure, List[Tuple[str, str]]]:
        # Convert article to ADF structure
        adf_structure = self.converter(article_text=article_text, task_definition=task_definition)
        
        # Check query against base-level factors
        query_results = self.checker.check_all_factors(
            factors=adf_structure.baselevel_factors,
            query=query,
            task_definition=task_definition
        )
        
        return adf_structure, query_results

def main():
    """Example usage of the ADF system"""
    # Sample data structure
    sample_adf = ADFStructure(
        outcomes=[
            "Foreign nationals have the right to enjoy private rights"
        ],
        abstract_factors=[
            "Private rights begin at birth",
            "Legal restrictions may limit foreign nationals' rights"
        ],
        baselevel_factors=[
            "Private rights are granted from the moment of birth",
            "Applicable laws can regulate foreign nationals' access",
            "Treaties may impose restrictions on foreign nationals",
            "Regulations may limit foreign nationals' private rights"
        ],
        links=[
            ("Private rights are granted from the moment of birth", "support", "Private rights begin at birth"),
            ("Applicable laws can regulate foreign nationals' access", "support", "Legal restrictions may limit foreign nationals' rights"),
            ("Treaties may impose restrictions on foreign nationals", "support", "Legal restrictions may limit foreign nationals' rights"),
            ("Regulations may limit foreign nationals' private rights", "support", "Legal restrictions may limit foreign nationals' rights"),
            ("Private rights begin at birth", "support", "Foreign nationals have the right to enjoy private rights"),
            ("Legal restrictions may limit foreign nationals' rights", "attack", "Foreign nationals have the right to enjoy private rights")
        ]
    )

    # Visualize the structure
    plt.figure(figsize=(30, 10))
    sample_adf.visualize()
    plt.show()

    # Initialize system and process actual article/query
    system = ADFSystem()
    
    # Load your actual data
    with open("./framework.txt", "r") as f:
        task_definition = f.read()
    with open("./article_3.txt", "r") as f:
        article = f.read()

    # Process article and query
    query = "An unborn child may not be given a gift on the donor's death."
    adf_structure, query_results = system.process_article_and_query(
        article_text=article,
        task_definition=task_definition,
        query=query
    )

    # Print results
    print("\nADF Structure:", adf_structure)
    print("\nQuery Results:", query_results)
    
    # Visualize the processed structure
    # adf_structure.visualize()
    # plt.show()

if __name__ == "__main__":
    main()