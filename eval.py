import json

from base.base_llm_answerer import BaseLLMAnswerer
from long.strict_adf_answerer import ADFAnswerer

import dspy
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.2",  # Changed to include provider prefix
        api_base="http://localhost:11434",
        max_tokens=20000
    )
)

cases = []
with open('testing_data.json') as f:
    cases = json.load(f)

correct = 0
module = ADFAnswerer()
for c in cases:
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]
    l= module.forward(articles=articles, query=query).answer
    if l == expected:
        correct += 1

print(correct, "/", len(cases))
