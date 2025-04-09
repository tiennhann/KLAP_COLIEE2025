from pathlib import Path
from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt

class MDATask(dspy.Signature):
    """Signature for converting legal context into Answer Set Programming facts."""
    task_definition = dspy.InputField(desc="The definition for the task.")
    retry= dspy.InputField(desc="Whether or not a previous attempt of the task failed. And the current attempt is a retry. The failed answer will be included.")
    failed_answer= dspy.InputField(desc="The failed answer if this is a retry attempt.")
    input_text = dspy.InputField(
        desc="The input text from which to convert context to the Answer Set Programming facts."
    )
    answer:List[str]= dspy.OutputField()


class MDAExtractor(dspy.Module):
    def __init__(self, file_path_input = ""):
        super().__init__()
        self.task = dspy.ChainOfThought(MDATask)
        if file_path_input == "":
            file_path = Path(__file__).parent / "extraction_task.txt"
        else: file_path = Path(__file__).parent / file_path_input
        # print("extraction prompt are using:", file_path)
        with file_path.open("r") as file:
            self.task_definition = "".join(file.readlines())
        self.prompt_path = file_path

    def forward(self, text, failed_answer=None):
        retry = False
        if failed_answer != None:
            retry = True

        converted = self.task(input_text=text, task_definition=self.task_definition, retry=retry, failed_answer=failed_answer)
        atoms = converted.answer
        for i in range(len(atoms)):
            atom = atoms[i]
            if not atom.endswith("."):
                atom += "."
                atoms[i] = atom
        return "".join(atoms)


if __name__ == "__main__":
    # Configure DSPy with the language model
    dspy.settings.configure(
        lm=dspy.LM(
            model="ollama_chat/llama3:8b",  # Changed to include provider prefix
            api_base="http://localhost:11434",
            max_tokens=20000,
        )
    )

    with open("./article.txt", "r") as f:
        article = "".join(f.readlines())

    module = MDAExtractor()
    print(module.forward(text=article))
