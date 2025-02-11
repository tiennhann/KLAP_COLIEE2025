import json

from base.base_llm_answerer import BaseLLMAnswerer 

cases = []
with open('testing_data.json') as f:
    cases = json.load(f)

correct = 0
module = BaseLLMAnswerer()
for c in cases:
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]
    l= module.forward(articles=articles, query=query).answer
    if l == expected:
        correct += 1

print(correct, "/", len(cases))
