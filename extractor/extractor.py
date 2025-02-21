from pathlib import Path
from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt

# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3:8b",  # Changed to include provider prefix
        api_base="http://localhost:11434",
        max_tokens=20000,
    )
)

class MDATask(dspy.Signature):
    """Signature for converting legal text to an ADF structure"""
    task_definition = dspy.InputField(desc="The definition for the task.")
    input_text = dspy.InputField(
        desc="The input text from which to extract the MDA structure."
    )
    answer:str= dspy.OutputField()


class MDAExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.task = dspy.ChainOfThought(MDATask)
        file_path = Path(__file__).parent / "extraction_task.txt"
        with file_path.open("r") as file:
            self.task_definition = "".join(file.readlines())

    def forward(self, text):
        converted = self.task(input_text=text, task_definition=self.task_definition)
        return converted.answer


if __name__ == "__main__":
    with open("./article.txt", "r") as f:
        article = "".join(f.readlines())

    module = MDAExtractor()
    print(module.forward(text=article))
