import json
import sys
from answerer_2 import Answerer
import random
import dspy
import time
import subprocess

def start_ollama_server():
    """Start the Ollama server on port 11435 using GPU 1."""
    print("Starting Ollama server on port 11435 with GPU 1...")
    env = {
        "OLLAMA_HOST": "127.0.0.1:11435",
        "CUDA_VISIBLE_DEVICES": "1"
    }
    server_process = subprocess.Popen(
        ["./ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**env, **dict(subprocess.os.environ)},  # Merge with existing environment variables
        cwd="/home/anguyen/ollama/bin"  # Set the working directory
    )
    time.sleep(5)  # Wait for the server to start
    return server_process

def stop_ollama_server(server_process):
    """Stop the Ollama server."""
    print("Stopping Ollama server...")
    server_process.terminate()
    server_process.wait()
    
    
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3.3",  # Changed to include provider prefix
        api_base="http://localhost:11435",
        max_tokens=20000
    )
)
# extractor_prompt_filename = "extraction_task.txt"
# train_as_test.json
# all_ins_article_698.json
# testing_data.json
# article_698_all_queries
# 
cases = []
with open('test_data/all_query_of_articles.json') as f:
    cases = json.load(f)

instance_run = 400
# cases = random.sample(cases, 400)
# run_1 = 8
correct = 0
wrong=0
error = 0
error_get_link = 0
# Start the Ollama server
llm_correct = 0
llm_wrong = 0
asp_correct_llm_wrong = 0
llm_correct_asp_wrong = 0

module = Answerer(debug=True)
# module = Answerer(debug=True)
instance_count = 0
for c in cases:
    if instance_count > instance_run: break
    instance_count +=1
    if 'run_1' in globals():
        if run_1:
            if instance_count < run_1: continue
            if instance_count > run_1: break
    query=c["query"]
    articles=c["paragraph"]
    if len(query) < 2: continue
    query_test = query.pop(random.randrange(len(query)))
    expected = query_test[1]
    query_examples = [query.pop(random.randrange(len(query))) for _ in range(2)] if len(query) > 2 else query
    # expected=c["label"]
    if 'server_process' in locals():
        stop_ollama_server(server_process)
    server_process = start_ollama_server()
    try:
        l, llm_answer= module.answer_2(article=articles, queries_examples=[q[0] for q in query_examples], labels_examples=[q[1] for q in query_examples], query_test=query_test[0], label_test=query_test[1])
    except KeyboardInterrupt:

        print("\nLoop interrupted. Exiting...")
        sys.exit()
    except:
        error += 1
        continue
    if l==True:
        l = "Y"
    elif  l==False:
        l = "N"
    if l == expected:
        correct += 1
    elif l == 'E':
        error += 1
    else:
        wrong += 1
    if llm_answer == expected:
        llm_correct +=1
    else: llm_wrong +=1
    
    if llm_answer == expected and l!= expected and l != 'E':
        llm_correct_asp_wrong +=1
    if llm_answer != expected and l==expected:
        asp_correct_llm_wrong +=1
        
    stop_ollama_server(server_process)
    print('instance num:', instance_count)
    print('instance id', c['id'])
    print("article:", articles)
    print("query:", query)
    print("true label:", expected, " - asp predicted label:", l, " - llm predicted label:", llm_answer)
    print(correct, "correct", wrong, "wrongs", correct + wrong, "out of", len(cases), "tested")
    print(error, " errors")
    print("accurary of asp_law=", round((correct/(wrong+correct))*100, 2))
    print("accuracy of llm=", round((llm_correct/(llm_correct+llm_wrong))*100, 2))
    print(f'asp_law correct , llm wrong: {asp_correct_llm_wrong}')
    print(f'llm correct, asp_law wrong:{llm_correct_asp_wrong}')


print(correct, "/", len(cases), "with", error, "errors")
