import json

from answerer import Answerer

import dspy
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.3",  # Changed to include provider prefix
        api_base="http://localhost:11435",
        max_tokens=20000
    )
)

cases = []
with open('testing_data.json') as f:
    cases = json.load(f)

correct = 0
wrong=0
error = 0
module = Answerer(debug=True)
for c in cases:
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]
    try:
        l= module.answer(article=articles, query=query)
    except:
        error += 1
        continue
    l = "Y" if l else "N"
    if l == expected:
        correct += 1
    else:
        wrong += 1
    print("true label:", expected, " - predicted label:", l)
    print(correct, "correct", wrong, "wrongs", correct + wrong, "out of", len(cases), "tested")

print(correct, "/", len(cases), "with", error, "errors")
