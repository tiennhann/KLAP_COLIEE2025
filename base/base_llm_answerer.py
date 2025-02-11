from typing import List, Literal, Tuple
import dspy

# Configure DSPy with the language model
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",  # Changed to include provider prefix
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)

class QueryEntailmentCheckTask(dspy.Signature):
    task_definition=dspy.InputField(desc="The definition for the task.")
    query=dspy.InputField(desc="A sentence, a question, or a statement.")
    input_articles=dspy.InputField(desc="The input articles that are relevant to the query.")
    answer:Literal["Y", "N"]=dspy.OutputField(desc="Answer if the query logically follows from the input articles.")

class BaseLLMAnswerer(dspy.Module):
    def __init__(self):
        with open("./base/base_llm_task_definition.txt", "r") as f:
            task_definition = "".join(f.readlines())
        super().__init__()
        self.agent = dspy.ChainOfThought(QueryEntailmentCheckTask)
        self.task_definition = task_definition

    def forward(self, articles, query):
        return self.agent(input_articles=articles, task_definition=self.task_definition, query=query)

if __name__ == "__main__":
    with open("./base_llm_task_definition.txt", "r") as f:
        task_definition = "".join(f.readlines())
    with open("./articles.txt", "r")  as f:
        article = "".join(f.readlines())
    
    module = BaseLLMAnswerer(task_definition=task_definition)
    query="Acceptance made by a minor that received an offer of gifts without burden without getting consent from his/her statutory agent may not be rescinded."
    print(module.forward(articles=article, query=query))