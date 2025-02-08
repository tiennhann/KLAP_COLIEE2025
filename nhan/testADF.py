import dspy

# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)
class LegalEntailment(dspy.Module):
    def __init__(self):
        super().__init__()
        self.entailment = dspy.ChainOfThought("query, article -> entailment")
    
    def forward(self, query, article):
        return self.entailment(query = query, article = article)

model = LegalEntailment()

# Example query & retrieved article
query = "An unborn child may not be given a gift on the donor’s death."
article = "Article 3 (1) The enjoyment of private rights commences at birth. (2) Unless otherwise prohibited by applicable laws, regulations, or treaties, foreign nationals enjoy private rights."

output = model.forward(query=query, article=article)
print(output)