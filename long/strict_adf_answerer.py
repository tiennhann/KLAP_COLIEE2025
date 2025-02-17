from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
# Configure DSPy with the language model

from itertools import chain, combinations, permutations

import dspy
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",  # Changed to include provider prefix
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)

# from response import Response

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

class Factor(BaseModel):
    id:str=Field(description="an id of the  factor")
    content:str=Field(description="The detailed description of the factor, not conjunction")
    conjunction_factors:List[str]|None=Field(description="The list of factors'id that are in this conjunction")

class ADM:
    factors: List[Factor]
    links: List[Tuple[str, Literal["support", "attack"]|None, str]]
    def __init__(self, factors: List[Factor], links):
        self.factors = factors
        self.links = links


class ADMExtractionTask(dspy.Signature):
    task_definition=dspy.InputField(desc="The definition for the task.")
    text=dspy.InputField(desc="The input article to convert into an MDA structure.")
    factors:List[Factor]=dspy.OutputField()

class ADMLinkExtractionTask(dspy.Signature):
    task_definition=dspy.InputField(desc="The definition for the task.")
    text=dspy.InputField(desc="The input article to convert into an MDA structure.")
    factor1: Factor=dspy.InputField()
    factor2: Factor=dspy.InputField()
    link:Literal["support", "attack"]|None=dspy.OutputField()

class ADMCheckEquivalent(dspy.Signature):
    context1=dspy.InputField()
    text1=dspy.InputField()
    text2=dspy.InputField()
    context2=dspy.InputField()
    answer:bool=dspy.OutputField(desc="Answer if text1 is talking about the same thing as text2")

class ADMExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.factors_extractor = dspy.ChainOfThought(ADMExtractionTask)
        self.link_extractor = dspy.ChainOfThought(ADMLinkExtractionTask)
        with open("./extraction_task.txt", "r") as f:
            task_definition = "".join(f.readlines())
        self.task_definition = task_definition

    def forward(self, text):
        factors: List[Factor]= self.factors_extractor(task_definition=self.task_definition, text=text).factors
        links=[]
        for (factor1, factor2) in list(permutations(factors, 2)): 
            if factor2.conjunction_factors is not None:
                continue
            try:
                link = self.link_extractor(task_definition=self.task_definition, text=text, factor1=factor1, factor2=factor2).link
                links.append((factor1.id, link, factor2.id))
            except:
                continue
        return ADM(factors, links)

import networkx as nx
import matplotlib.pyplot as plt
import textwrap

def visualize_adm(adm: ADM, wrap_width=40):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each factor
    for factor in adm.factors:
        # Optionally wrap text so it fits better in the graph
        wrapped_content = "\n".join(textwrap.wrap(factor.content, width=wrap_width))
        G.add_node(factor.id, label=wrapped_content)

    # Add edges for the links
    for source, relation, target in adm.links:
        # Choose an edge color based on the relation
        if relation == "support":
            edge_color = "green"
        elif relation == "attack":
            edge_color = "red"
        else:
            continue
            
        G.add_edge(source, target, relation=relation, color=edge_color)

    for factor in adm.factors:
        # If there are conjunction factors, add edges from each conjunction factor to this one
        if factor.conjunction_factors:
            G.add_node(factor.id, label="conjunction")
            for conj_id in factor.conjunction_factors:
                G.add_edge(conj_id, factor.id, relation="conjunction", color="blue")

    # Compute a layout for the graph
    pos = nx.spring_layout(G)
    
    # Extract edge colors from edge attributes
    colors = [data["color"] for _, _, data in G.edges(data=True)]
    
    # Draw the graph **without** labels
    nx.draw(
        G, pos, with_labels=False, edge_color=colors,
        node_color="lightblue", node_size=2000,
        font_size=8, font_color="black"
    )
    
    # Prepare a label dictionary to include the full/wrapped text
    node_labels = {node: data["label"] for node, data in G.nodes(data=True)}
    
    # Now draw the labels separately so they aren't truncated
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    # Optionally draw edge labels to show "support"/"attack"/"conjunction"
    edge_labels = {(u, v): data['relation'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Show the plot
    plt.title("ADM Graph Visualization")
    plt.show()



if __name__ == "__main__":
    extractor = ADMExtractor()
    with open("./article.txt", "r") as f:
        articles= "".join(f.readlines())
    query = "Acceptance made by a minor that received an offer of gifts without burden without getting consent from his/her statutory agent may not be rescinded."
    visualize_adm(extractor.forward(text=articles))
    # visualize_adm(extractor.forward(text=query))