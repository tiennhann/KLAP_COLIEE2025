import json
import re
from answerer import Answerer
from minh.deepseek import chat
#from budget_forcing_dif import chat
import random
from turtle import pd
import xml.etree.ElementTree as ET


def parse_xml_file(file_path):
    tree = ET.parse(file_path)    
    root = tree.getroot()    
    records = []

    for pair in root.findall('pair'):
        query_id = pair.attrib["id"]
        verdict = pair.attrib["label"]
        query = pair.find('t2')
        query = query.text if query is not None else None
        paragraphs  = pair.find('t1')
        paragraphs  = paragraphs.text if query is not None else None

        query =  query.strip()
        paragraphs = paragraphs.strip()
        query_id = query_id.strip()

        record_data = {
            'ID': query_id,
            "verdict": verdict,
            'query': query,
            'articles': paragraphs 
        }

        records.append(record_data)

    return records

def records_to_dataframe(records):
    df = pd.DataFrame(records)
    return df

raw_file_name = "riteval_H21_en" 
file_name = raw_file_name + ".xml"
parsed_records = parse_xml_file(file_name)
# Print results (or process further)
for r in parsed_records:        
    first_art = r["articles"].split("\nA")[0]
    s = [first_art]
    s += ["A" + e for e in r["articles"].split("\nA")[1:] if e]
    r["articles"] = s   

cases = []
with open('train_as_test.json') as f:
    cases = json.load(f)
cases = random.sample(cases, 1000)
data = []
correct = 0
wrong=0
error = 0
module = Answerer(debug=True)
for c in cases:
    query=c["query"]
    articles=c["paragraphs"]
    expected=c["label"]
    ID = c["id"]
    input_article = articles
#    for article in articles:
#        input_article += article + "\n"
    print("ID:", ID)
#    print("article:", input_article)
#    print("query:", query)
    
#    l= module.answer(article=articles, query=query)
    l = chat(article = input_article, query=query)
    # extract_text = re.sub(r"<think>.*?</think>", "", l, flags=re.DOTALL)
    # output = extract_text.replace("\n", "")
    outputs = l
    print("answer:\n", outputs, "\n\n")
    print("expected:\n", expected, "\n\n")
    
    datapoint = {
           "id": ID,
           "query": query,
           "articles":input_article,
           "output": outputs,
           "expected": expected
    }
    data.append(datapoint)

with open("100_random_test" + ".json", "w") as f:
    json.dump(data, f, indent=4)

#    l = "Y" if l else "N"
#    if l == expected:
#        correct += 1
#    else:
#        wrong += 1
    print("end of one case______________________________")
#    print(correct, "correct", wrong, "wrongs", correct + wrong, "out of", len(cases), "tested")

#print(correct, "/", len(cases), "with", error, "errors")
