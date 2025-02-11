from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field

# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",  # Changed to include provider prefix
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)

class ADFStructure(BaseModel):
    issues: List[str] = Field(description="A list of the issues of the input article.")
    factors: List[str] = Field(description="A list of the factors of the input article.")
    links: List[Tuple[str, Literal["support", "attack"], str]]= Field(description="A list of the links bewteen the factors and the outcomes of the input article.")


class ADFConversionTask(dspy.Signature):
    """Signature for converting legal text to an ADF structure"""
    task_definition=dspy.InputField(desc="The definition for the task.")
    input_article=dspy.InputField(desc="The input article to convert into an ADF structure.")
    adf_structure:ADFStructure=dspy.OutputField()

class ADFConverter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.agent = dspy.ChainOfThought(ADFConversionTask)

    def forward(self, article_text, task_definition, query):
        return self.agent(input_article=article_text, task_definition=task_definition, query=query)

if __name__ == "__main__":
    with open("./framework.txt", "r") as f:
        task_definition = "".join(f.readlines())
    with open("./articles.txt", "r")  as f:
        article = "".join(f.readlines())
    
    module = ADFConverter()
    
    print(module.forward(article_text=article, task_definition=task_definition))