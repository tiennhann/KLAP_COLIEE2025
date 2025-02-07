import dspy
import json
import sys
import os

# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama/llama3.2",  # Changed to include provider prefix
        api_base="http://localhost:11434",
        api_type="ollama"  # Added to specify the API type
    )
)

class ADFStructure(dspy.Signature):
    """Signature for converting legal text to ADF structure"""
    article_text = dspy.InputField(desc="The legal article text to analyze")
    query = dspy.InputField(desc="The query to evaluate")
    context = dspy.InputField(desc="The ADF analysis instructions")
    root_decision = dspy.OutputField(desc="The main decision/conclusion (root node)")
    internal_nodes = dspy.OutputField(desc="List of intermediate factors with their acceptance conditions")
    base_factors = dspy.OutputField(desc="List of leaf nodes with their truth values")
    supporting_links = dspy.OutputField(desc="List of relationships that support higher nodes")
    attacking_links = dspy.OutputField(desc="List of relationships that oppose higher nodes")

class ADFConverter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(ADFStructure)

    def forward(self, query, article_text):
        adf_prompt = f"""
        Convert this legal analysis into an ADF structure following these rules:

        1. Root Decision Node:
        - Identify the main decision from the query
        - Format: "Decision: [Y/N] - [Description]"

        2. Internal Nodes (Intermediate Factors):
        - List factors that influence the decision
        - Include acceptance conditions
        - Format each node as:
            "Factor: [name]
            Condition: Accept if [condition], Reject if [condition]"

        3. Base-level Factors (Leaf Nodes):
        - List concrete facts/rules from the article
        - Indicate if true/false based on given information
        - Format: "Base Factor: [fact/rule] - [True/False]"

        4. Links:
        - Supporting Links: List relationships that support higher nodes
        - Attacking Links: List relationships that oppose higher nodes
        - Format: "[Source] -> [Target] : [Support/Attack]"

        Example Output Structure:
        Root Decision: "Y - Minor can accept gifts without burden"
        Internal Nodes: ["Legal Capacity - Accept if acquiring rights only"]
        Base Factors: ["Gift without burden - True", "Consent requirement - False"]
        Supporting Links: ["Gift without burden -> Legal Capacity"]
        Attacking Links: []

        Legal Article:
        {article_text}

        Query to Analyze:
        {query}
        """
        
        # Pass both the article text and query along with the prompt
        result = self.predictor(
            article_text=article_text,
            query=query,
            context=adf_prompt
        )
        
        return result

def process_test_cases(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    module = ADFConverter()
    results = []

    for case in test_cases:
        try:
            print(f"\nProcessing case {case['id']}...")
            adf_structure = module.forward(case['query'], case['paragraphs'])
            
            result = {
                'id': case['id'],
                'expected_label': case['label'],
                'query': case['query'],
                'article': case['paragraphs'],
                'adf_structure': {
                    'root_decision': adf_structure.root_decision,
                    'internal_nodes': adf_structure.internal_nodes,
                    'base_factors': adf_structure.base_factors,
                    'supporting_links': adf_structure.supporting_links,
                    'attacking_links': adf_structure.attacking_links
                }
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing case {case['id']}: {e}")
    
    output_file = 'adf_structures.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nADF structures saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ADFconverter.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    process_test_cases(input_file)   