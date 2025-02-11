import json

from base.base_llm_answerer import BaseLLMAnswerer
from long.loose_adf_answerer import LooseADFAnswerer
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
with open('train_as_test.json') as f:
    cases = json.load(f)

correct = 0
wrong=0
module = BaseLLMAnswerer()
for c in cases:
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]
    l= module.forward(articles=articles, query=query).answer
    if l == expected:
        correct += 1
    else:
        wrong += 1

    print(correct, "correct", wrong, "wrongs", correct + wrong, "out of", len(cases), "tested")

print(correct, "/", len(cases))
