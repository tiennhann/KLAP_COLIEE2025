from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GenAngelic():
    def __init__(self):
        model_name = "deepseek-ai/deepseek-r1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bf16 for efficiency
            device_map="auto",  # Auto place model on available GPU(s)
        )

# Enable evaluation mode (no gradient updates)
        pass
    def chat(self, article, query):
        


        prompt = f"""
        Given the following articles:
        Article:
        {article}

        Query: 
        {query}

        Role:
        You are an expert legal assistant.

        Task:
        1. Generate the ANGELIC structure (a framework in the Law AI domain) based on the provided articles while ensuring the following conditions are met:

        Structural Requirements:
            •	Nodes:
                - Include the following types of nodes: Root (R), Abstract Factors (A), Base-Level Factors (B).
            •	Directed Links:
                - Support Link: Indicates that the source node validates the truth of the target node.
                - Attack Link: Indicates that the source node invalidates the truth of the target node.
            •   Hierarchy Constraints:
                - The Root (R) nodes do not have directed links to Abstract Factors (A) or Base-Level Factors (B).

        Naming Conventions:
            •	Root Nodes: R1 for the first root, R2 for the second root (if needed), and so on.
            •	Abstract Factors: A1 for the first abstract factor, A2 for the second, and so forth.
            •	Base-Level Factors: B1 for the first base-level factor, B2 for the second, and so on.
            •	Link Notation: Each directed link must be explicitly denoted with a unique number.
        
        2. Generate the entailment relationship between the given query (treated as a hypothesis) and each Base-Level Factor (B). The query entails a Base-Level Factor if the context of the Base-Level Factor is strongly related to the hypothesis. Clearly determine and justify whether each Base-Level Factor is entailed by the hypothesis.
        
        Post-Processing Step:

        After generating the ANGELIC structure, assume that the Base-Level Factors (B), Root Nodes (R), and Abstract Factors (A) are correctly identified. Carefully review and re-evaluate all links to ensure logical consistency and justification.
        """
        inputs = self.tokenizer(prompt, return_tensor="pt").to("cuda")
        output = self.model.generate(**inputs, max_length = 2000)
        # Generate the response using the Hugging Face model
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)