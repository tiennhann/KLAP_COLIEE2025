from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt

# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)

class ADFStructure(BaseModel):
    outcomes: List[str] = Field(description="Outcome is the agreement with article.")
    abstract_factors: List[str] = Field(description="A list of the abstract factors of the input article.")
    baselevel_factors: List[str] = Field(description="A list of the baselvel factors of the input article.")
    links: List[Tuple[str, Literal["support", "attack"], str]] = Field(description="A list of the links between the factors and the outcomes of the input article.")

    def visualize(self):
        """Method to visualize the ADF structure as a hierarchical graph"""
        G = nx.DiGraph()
        
        # Create lists to store nodes and their colors
        nodes = []
        colors = []
        
        # Calculate levels for hierarchical layout
        levels = {
            'outcomes': 0,
            'abstract': 1,
            'baselevel': 2
        }
        
        # Add nodes with their hierarchical positions
        pos = {}
        
        # Add outcomes (top level)
        outcome_spacing = 1.0 / (len(self.outcomes) + 1)
        for i, outcome in enumerate(self.outcomes, 1):
            G.add_node(outcome, node_type='outcome')
            nodes.append(outcome)
            colors.append('lightblue')
            pos[outcome] = (outcome_spacing * i, 1.0)  # Top level
        
        # Add abstract factors (middle level)
        abstract_spacing = 1.0 / (len(self.abstract_factors) + 1)
        for i, factor in enumerate(self.abstract_factors, 1):
            G.add_node(factor, node_type='abstract')
            nodes.append(factor)
            colors.append('lightgreen')
            pos[factor] = (abstract_spacing * i, 0.5)  # Middle level
        
        # Add baselevel factors (bottom level)
        baselevel_spacing = 1.0 / (len(self.baselevel_factors) + 1)
        for i, factor in enumerate(self.baselevel_factors, 1):
            G.add_node(factor, node_type='baselevel')
            nodes.append(factor)
            colors.append('lightcoral')
            pos[factor] = (baselevel_spacing * i, 0.0)  # Bottom level
        
        # Add edges with different colors for support/attack
        edges = []
        edge_colors = []
        for source, link_type, target in self.links:
            if source in G and target in G:
                G.add_edge(source, target)
                edges.append((source, target))
                edge_colors.append('green' if link_type == 'support' else 'red')
        
        # Draw the graph
        plt.figure(figsize=(20, 12))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                                nodelist=nodes,
                                node_color=colors,
                                node_size=3000)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, arrows=True, width=2,
                              arrowsize=20)
        
        # Draw labels with word wrapping
        labels = {}
        for node in G.nodes():
            # Wrap text at 20 characters
            words = node.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > 20:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            
            labels[node] = '\n'.join(lines)
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      label='Outcomes', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                      label='Abstract Factors', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                      label='Baselevel Factors', markersize=15),
            plt.Line2D([0], [0], color='green', label='Support', linewidth=2),
            plt.Line2D([0], [0], color='red', label='Attack', linewidth=2)
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add level labels
        plt.text(-0.1, 1.0, 'Outcomes', fontsize=12, fontweight='bold')
        plt.text(-0.1, 0.5, 'Abstract Factors', fontsize=12, fontweight='bold')
        plt.text(-0.1, 0.0, 'Baselevel Factors', fontsize=12, fontweight='bold')
        
        # Set margins and turn off axis
        plt.margins(x=0.2)
        plt.axis('off')
        
        return G

class ADFConversionTask(dspy.Signature):
    """Signature for converting legal text to an ADF structure"""
    task_definition = dspy.InputField(desc="The definition for the task.")
    input_article = dspy.InputField(desc="The input article to convert into an ADF structure.")
    query = dspy.InputField(desc="The query to check against the article.")
    adf_structure: ADFStructure = dspy.OutputField()
    entailment: Literal["Yes", "No"] = dspy.OutputField(desc="Whether the article entails the query")
    reasoning: str = dspy.OutputField(desc="Explanation for the entailment decision")

class ADFQueryCheckTask(dspy.Signature):
    """Signature for mapping a query into an ADF structure"""
    task_definition = dspy.InputField(desc="The task definition.")
    query = dspy.InputField(desc="The query to check against the baselevel factors.")
    factor = dspy.InputField(desc="A ADF factor.")
    output: Literal["Yes", "No"] = dspy.OutputField(desc="Query Justification")

class ArticleRelevanceTask(dspy.Signature):
    """Signature for checking article relevance to a query"""
    query = dspy.InputField(desc="The legal query to check")
    article = dspy.InputField(desc="The legal article text")
    is_relevant: bool = dspy.OutputField(desc="Whether the article is relevant to the query")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")
    reasoning: str = dspy.OutputField(desc="Explanation for the relevance assessment")

class ADFConverter(dspy.Module):
    """Module for converting legal text to ADF structure with entailment"""
    def __init__(self):
        super().__init__()
        self.converter = dspy.ChainOfThought(ADFConversionTask)

    def forward(self, article_text: str, query: str, task_definition: str) -> Tuple[ADFStructure, str, str, str]:
        """Convert article text to ADF structure and check entailment"""
        result = self.converter(
            input_article=article_text,
            query=query,
            task_definition=task_definition
        )
        
        # Extract reasoning about the entailment
        reasoning = [
            f"Article Analysis:",
            f"1. Main Statement: {result.adf_structure.outcomes[0]}",
            f"2. Key Implications:"
        ]
        
        # Add abstract factor implications
        for i, factor in enumerate(result.adf_structure.abstract_factors, 1):
            reasoning.append(f"   {i}. {factor}")
            
        # Add baselevel factor support
        reasoning.append(f"3. Supporting Evidence:")
        for i, factor in enumerate(result.adf_structure.baselevel_factors, 1):
            reasoning.append(f"   {i}. {factor}")
            
        # Add entailment conclusion
        reasoning.append(f"\nEntailment Decision: {result.entailment}")
        reasoning.append(f"Reasoning: {result.reasoning}")
        
        return result.adf_structure, result.entailment, "\n".join(reasoning)

class ADFQueryChecker(dspy.Module):
    """Enhanced module for legal textual entailment using ADF structure"""
    def __init__(self):
        super().__init__()
        self.query_checker = dspy.ChainOfThought(ADFQueryCheckTask)

    def check_factor(self, factor: str, query: str, task_definition: str) -> Tuple[str, float, str]:
        """Enhanced check for legal entailment with confidence and reasoning"""
        # Check direct entailment
        result = self.query_checker(
            factor=factor,
            query=query,
            task_definition=task_definition
        )
        
        # Generate confidence based on the output
        confidence = 0.8 if result.output == "Yes" else 0.2
        reasoning = f"Based on comparison between '{factor}' and query '{query}'"
        
        return result.output, confidence, reasoning

    def check_all_factors(self, adf_structure: ADFStructure, query: str, task_definition: str) -> dict:
        """Comprehensive legal entailment analysis"""
        results = {
            'baselevel_checks': [],
            'abstract_implications': [],
            'outcome_implications': [],
            'final_answer': "No",  # Default to No
            'confidence': 0.0,
            'reasoning': []
        }
        
        # Check baselevel factors
        total_confidence = 0.0
        relevant_factors = 0
        
        # First analyze the query against all factors
        results['reasoning'].append(f"Analyzing query: '{query}'")
        results['reasoning'].append("")
        
        for factor in adf_structure.baselevel_factors:
            result, confidence, reasoning = self.check_factor(factor, query, task_definition)
            results['baselevel_checks'].append((factor, result, confidence, reasoning))
            
            relevant_factors += 1
            total_confidence += confidence
            results['reasoning'].append(f"Factor: '{factor}'")
            results['reasoning'].append(f"Assessment: {result} (Confidence: {confidence:.2f})")
            results['reasoning'].append(f"Reasoning: {reasoning}")
            results['reasoning'].append("")

            # Track implications if result is Yes
            if result == "Yes" and confidence > 0.5:
                current_factor = factor
                chain = [current_factor]
                
                while current_factor:
                    next_link = None
                    for source, link_type, target in adf_structure.links:
                        if source == current_factor:
                            impact = confidence if link_type == "support" else -confidence
                            results['abstract_implications'].append(
                                (current_factor, link_type, target, confidence)
                            )
                            chain.append(f"--({link_type})--> {target}")
                            next_link = target
                            break
                    current_factor = next_link
                
                if chain:
                    results['reasoning'].append("Implication chain: " + " ".join(chain))
                    results['reasoning'].append("")

        # Calculate final confidence and answer
        if relevant_factors > 0:
            # Average confidence from base factors
            base_confidence = total_confidence / relevant_factors
            
            # Determine final answer and confidence
            yes_count = sum(1 for _, result, _, _ in results['baselevel_checks'] if result == "Yes")
            no_count = sum(1 for _, result, _, _ in results['baselevel_checks'] if result == "No")
            
            # Calculate final confidence
            if yes_count + no_count > 0:
                results['confidence'] = base_confidence
                
                # Determine answer based on majority and confidence
                if yes_count > no_count and base_confidence > 0.5:
                    results['final_answer'] = "Yes"
            
            results['reasoning'].append("Final Analysis:")
            results['reasoning'].append(f"- Total factors checked: {relevant_factors}")
            results['reasoning'].append(f"- Yes responses: {yes_count}")
            results['reasoning'].append(f"- No responses: {no_count}")
            results['reasoning'].append(f"- Final confidence: {results['confidence']:.2f}")
            results['reasoning'].append(f"- Final answer: {results['final_answer']}")
        
        return results

class ArticleRelevanceChecker(dspy.Module):
    """Module for checking article relevance to queries"""
    def __init__(self):
        super().__init__()
        self.relevance_checker = dspy.ChainOfThought(ArticleRelevanceTask)
    
    def check_relevance(self, query: str, article: str) -> Tuple[bool, float, str]:
        """Check if an article is relevant to a query"""
        result = self.relevance_checker(
            query=query,
            article=article
        )
        return result.is_relevant, result.confidence, result.reasoning

class ADFSystem:
    """Legal textual entailment system using ADF"""
    def __init__(self):
        self.converter = ADFConverter()
        self.checker = ADFQueryChecker()
        self.relevance_checker = ArticleRelevanceChecker()

    def process_legal_query(self, article_text: str, query: str, task_definition: str) -> Tuple[str, float, List[str], ADFStructure]:
        """Process a legal query and determine entailment"""
        # Convert article to ADF and get entailment
        adf_structure, entailment, reasoning = self.converter(
            article_text=article_text,
            query=query,
            task_definition=task_definition
        )
        
        # Calculate confidence based on structure support
        confidence = self._calculate_confidence(adf_structure, entailment)
        
        return entailment, confidence, reasoning.split("\n"), adf_structure

    def _calculate_confidence(self, adf_structure: ADFStructure, entailment: str) -> float:
        """Calculate confidence based on supporting evidence"""
        # Count supporting factors
        total_factors = len(adf_structure.abstract_factors) + len(adf_structure.baselevel_factors)
        supporting_links = sum(1 for _, link_type, _ in adf_structure.links if link_type == "support")
        
        # Calculate base confidence
        base_confidence = supporting_links / max(total_factors, 1)
        
        # Adjust confidence based on entailment
        confidence = base_confidence * (0.8 if entailment == "Yes" else 0.6)
        
        return min(confidence, 1.0)

def main():
    """Main function demonstrating legal textual entailment"""
    # Load task definition and article
    with open("./framework.txt", "r") as f:
        task_definition = f.read()
    with open("./article_3.txt", "r") as f:
        article = f.read()

    # Initialize system
    system = ADFSystem()
    
    # First get and print the ADF structure
    print("\nArticle Text:")
    print("-" * 50)
    print(article)
    
    print("\nADF Structure Analysis")
    print("=" * 50)
    
    # Process each query
    test_queries = [
        "An unborn child may not be given a gift on the donor's death.",
    ]
    
    # Store ADF structure for visualization
    first_adf = None
    
    # Process each query
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Process query against the article
        answer, confidence, reasoning, adf_structure = system.process_legal_query(
            article_text=article,
            query=query,
            task_definition=task_definition
        )
        
        if first_adf is None:
            first_adf = adf_structure
        
        # Print results
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence:.2f}")
        print("\nReasoning:")
        for step in reasoning:
            print(f"  - {step}")

        # Print ADF Structure components
        print("\nADF Structure Components:")
        print("-" * 20)
        print("\nOutcomes:")
        for i, outcome in enumerate(adf_structure.outcomes, 1):
            print(f"{i}. {outcome}")
        
        print("\nAbstract Factors:")
        for i, factor in enumerate(adf_structure.abstract_factors, 1):
            print(f"{i}. {factor}")
        
        print("\nBaselevel Factors:")
        for i, factor in enumerate(adf_structure.baselevel_factors, 1):
            print(f"{i}. {factor}")
        
        print("\nLinks:")
        for i, (source, link_type, target) in enumerate(adf_structure.links, 1):
            print(f"{i}. {source}")
            print(f"   --({link_type})-->")
            print(f"   {target}")
            print()

    # Visualize the ADF structure
    if first_adf:
        print("\nGenerating ADF Structure Visualization...")
        G = first_adf.visualize()
        plt.show()

if __name__ == "__main__":
    main()