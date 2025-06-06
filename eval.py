import json
import sys
from answerer import Answerer
import random
import dspy
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.3",  # Changed to include provider prefix
        api_base="http://localhost:11435",
        max_tokens=20000
    )
)
# extractor_prompt_filename = "extraction_task.txt"
extractor_prompt_filename = "extraction_query_arguments.txt"
theory_filename = "theory_4.lp"
prompt_filename = "./duong/init_prompt_angelic_7.2.txt"
cases = []
with open('test_data.json') as f:
    cases = json.load(f)

cases = random.sample(cases, 100)

correct = 0
wrong=0
error = 0
module = Answerer(extractor_prompt_filename=extractor_prompt_filename, theory_filename=theory_filename, prompt_filename=prompt_filename, debug=True)
# module = Answerer(debug=True)
for c in cases:
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]
    print("article:", articles)
    print("query:", query)
    try:
        l= module.answer(article=articles, query=query)
    except KeyboardInterrupt:
        print("\nLoop interrupted. Exiting...")
        sys.exit()
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
