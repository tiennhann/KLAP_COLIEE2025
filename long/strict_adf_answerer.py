from typing import List, Literal, Tuple
import dspy
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
# Configure DSPy with the language model


from itertools import chain, combinations

from response import Response

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

class ADFStructure(BaseModel):
    outcomes: List[str] = Field(description="Outcome of the input article.")
    factors: List[str] = Field(description="A list of the factors that cause the outcome of the input article.")

class ADFConversionTask(dspy.Signature):
    """Signature for converting legal text to an ADF structure"""
    task_definition=dspy.InputField(desc="The definition for the task.")
    input_article=dspy.InputField(desc="The input article to convert into an MDA structure.")
    adf_structure:ADFStructure=dspy.OutputField()

class ADFQuestionTask(dspy.Signature):
    task_definition=dspy.InputField(desc="The definition for the task.")
    outcome=dspy.InputField(desc="The outcome to check the relationship of the input group to.")
    input_article=dspy.InputField(desc="The input article where the group of factors were generated from.")
    factor_group: List[str]=dspy.InputField(desc="The input group of factors.")
    answer: Literal["leads_to", "does_not_lead_to", "unknown"]=dspy.OutputField(desc="Answer of whether or not the input group of factor leads to the outcome or not.")

class ADFChecker(dspy.Signature): 
    query=dspy.InputField()
    factor=dspy.InputField(desc="A factor in the MDA structure.")
    answer:bool=dspy.OutputField(desc="Is the query applicable to the factor?")

class ADFAnswerer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.agent = dspy.ChainOfThought(ADFConversionTask)
        with open("./long/task_definition.txt", "r") as f:
            task_definition = "".join(f.readlines())
        self.task_definition = task_definition

    def forward(self, articles, query):
        answerer = dspy.ChainOfThought(ADFQuestionTask)
        checker = dspy.ChainOfThought(ADFChecker)
        converted = self.agent(input_article=articles, task_definition=self.task_definition).adf_structure
        factors = converted.factors
        factor_groups = list(powerset(converted.factors))
        outcomes = converted.outcomes
        rules = []
        checkers = []

        for factor_group in factor_groups:
            for outcome in outcomes:
                rule = answerer(input_article=articles, task_definition=self.task_definition, outcome=converted.outcomes[0], factor_group=factor_group).answer
                rules.append((outcome, factor_group, rule))
        
        truth_values = {}
        for factor in factors:
            truth_value = checker(query=query, factor=factor).answer
            truth_values[factor] = truth_value

            checkers.append((factor, truth_value))

        answers = []
        for rule in rules:
            rel = rule[2]
            factor_group = rule[1]
            answer = False
            if rel == "leads_to":
                answer = True 
                for factor in factor_group:
                    answer = answer and truth_values[factor]
            elif rel == "does_not_lead_to":
                answer = False 
                for factor in factor_group:
                    answer = answer or not truth_values[factor]
            if answer:
                return Response("Y")
        if len(answers) == 0:
            return Response("N")

        