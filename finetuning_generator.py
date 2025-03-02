import json

from answerer import Answerer, ExtractorException

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

module = Answerer(debug=False)
# print("Generate finetuning data for extractor...")
# data = []
# for i, c in enumerate(cases):
#     query=c["query"]
#     articles=c["paragraphs"]
#     expected=c["label"]

#     failed_answer = ""
#     count = 20
#     while True:
#         if count < 0:
#             print("ignore this case, failed to many times")
#             break
#         count -= 1
#         try:
#             chat, extracted, l= module.retry_extractor(article=articles, query=query, failed_answer=failed_answer)
#             l = "Y" if l else "N"
#             if l == expected:
#                 data.append({"chat": chat, "extracted": extracted})
#                 print("Generated for case", i)
#                 break
#             else:
#                 print("retrying... wrong answer")
#                 failed_answer = extracted
#                 continue
#         except ExtractorException as e:
#             print("retrying...  failed")
#             failed_answer = e.args[0]
#             continue
#         except:
#             raise
# with open("finetuning_extractor_data.json", "w") as f:
#     f.writelines(json.dumps(data))
# print("Done extractor...")

print("Generate finetuning data for chat...")
data = []
for i, c in enumerate(cases):
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]

    failed_answer = ""
    while True:
        try:
            chat, l= module.retry_chat(article=articles, query=query, failed_answer=failed_answer)
            l = "Y" if l else "N"
            if l == expected:
                data.append({"query": query, "chat": chat})
                print("Generated for case", i)
                break
            else:
                print("retrying... wrong answer")
                failed_answer = chat 
                continue
        except ExtractorException as e:
            print("retrying...  failed")
            failed_answer = e.args[0]
            continue
        except:
            raise
with open("finetuning_chat_data.json", "w") as f:
    f.writelines(json.dumps(data))
print("Done chat...")