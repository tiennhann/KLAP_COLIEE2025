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


class  ADFQueryCheckTask(dspy.Signature):
    """Signature for mapping a query into an ADF structure"""
    query=dspy.InputField(desc="The query to check against the baselevel factors.")
    factor=dspy.InputField(desc="A ADF factor.")
    output:Literal["Yes", "No", "Undecided"]=dspy.OutputField(desc="Query Justification")

class ADFConverter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.adf_converter = dspy.ChainOfThought(ADFConversionTask)
        self.factor_checker= dspy.ChainOfThought(ADFQueryCheckTask) 

    def forward(self, article_text, task_definition,  query):
        adf_structure:ADFStructure = self.adf_converter(input_article=article_text, task_definition=task_definition).adf_structure
        factors = adf_structure.baselevel_factors  
        result=[]
        for factor in factors:
            result.append((factor, self.factor_checker(factor=factor, query=query, task_definition=task_definition).output))
        return (adf_structure, result)

if __name__ == "__main__":
    with open("./framework.txt", "r") as f:
        task_definition = "".join(f.readlines())
    with open("./article_3.txt", "r")  as f:
        article = "".join(f.readlines())
    
    module = ADFConverter()
    query="An unborn child may not be given a gift on the donor's death."
    output = module.forward(article_text=article, task_definition=task_definition,  query=query)
    print(output[0])
    print(output[1])


   