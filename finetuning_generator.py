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
import logging

logging.getLogger("requests").setLevel(logging.CRITICAL)

cases = []
with open('train_as_test.json') as f:
    cases = json.load(f)

module = Answerer()
print("Generate finetuning data for extractor...")
data = []
data_count = 2
for i, c in enumerate(cases[11:]):
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]

    failed_answer = None
    while True:
        try:
            chat, extracted, l= module.retry_extractor(article=articles, query=query, failed_answer=failed_answer)
            l = "Y" if l else "N"
            if l == expected:
                data.append({"chat": chat, "extracted": extracted})
                print("Generated for case", i)
                break
            else:
                print("retrying... wrong answer")
                failed_answer = f"The following answer failed because it did not give the right result: {extracted}"
                continue
        except ExtractorException as e:
            print("retrying...  failed")
            failed_answer = f"The following answer failed because it's not the right Clingo syntax: {e.args[0]}"
            continue
        except Exception as e:
            print(e)
            continue
    if len(data) == 5: 
        with open(f"finetuning_extractor_data_{data_count}.json", "w") as f:
            f.writelines(json.dumps(data))
        data_count+= 1
        data = []
if len(data) > 0:
    with open(f"finetuning_extractor_data_{data_count}.json", "w") as f:
        f.writelines(json.dumps(data))
print("Done extractor...")

# print("Generate finetuning data for chat...")
# data = []
# data_count = 0
# for i, c in enumerate(cases):
#     query=c["query"]
#     articles=c["paragraphs"]
#     expected=c["label"]

#     failed_answer = ""
#     while True:
#         try:
#             chat, l= module.retry_chat(article=articles, query=query, failed_answer=failed_answer)
#             l = "Y" if l else "N"
#             if l == expected:
#                 data.append({"query": query, "articles": articles, "chat": chat})
#                 print("Generated for case", i)
#                 break
#             else:
#                 print("retrying... wrong answer")
#                 failed_answer = chat 
#                 continue
#         except ExtractorException as e:
#             print("retrying...  failed")
#             failed_answer = e.args[0]
#             continue
#         except:
#             raise
#     if len(data) == 10: 
#         with open(f"finetuning_chat_data_{data_count}.json", "w") as f:
#             f.writelines(json.dumps(data))
#         data_count+= 1
#         data = []
# if len(data) > 0:
#     with open(f"finetuning_chat_data_{data_count}.json", "w") as f:
#         f.writelines(json.dumps(data))
# print("Done chat...")