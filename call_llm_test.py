import json
import sys
from answerer_2 import Answerer, ExtractFactorFromText
import random
import dspy
import time
import subprocess
from duong.deepseek_2 import CallLLM
import signal
import os

def kill_ollama_gpu_task():
    """Find and kill the 'ollama' task running on the GPU using nvidia-smi."""
    try:
        # Run nvidia-smi to list GPU processes
        result = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        for line in result.splitlines():
            if "ollama" in line:
                # Extract the PID (Process ID)
                pid = int(line.split()[4])  # PID is usually in the 5th column
                print(f"Killing 'ollama' GPU task with PID: {pid}")
                os.kill(pid, signal.SIGKILL)
                print(f"Process {pid} killed successfully.")
                return
        print("No 'ollama' GPU task found.")
    except Exception as e:
        print(f"Error while killing 'ollama' GPU task: {e}")
        
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

# c = {
#     "id": "H30-28-E",
#     "label": "N",
#     "query": "When A was on a long-term business trip and was absent, a part of a hedge of A's house collapsed due to a strong wind. Afterward, B who has a house on a land next to A's house performed an act for A without any obligation. If B started repair of the hedge by him/herself but the hedge has all withered because B left it in the middle of the repair, A may not claim the damage of the withered hedge.",
#     "paragraphs": "Article 698\nIf a manager engages in benevolent intervention in another's business in order to allow a principal to escape imminent danger to the principal's person, reputation, or property, the manager is not liable to compensate for damage resulting from this unless the manager has acted in bad faith or with gross negligence."
# }

c = {
    "id": "H30-28-E",
    "label": "N",
    "query": [
              ["When A was on a long-term business trip and was absent, a part of a hedge of A's house collapsed due to a strong wind. Afterward, B who has a house on a land next to A's house performed an act for A without any obligation. If B started repair of the hedge by him/herself but the hedge has all withered because B left it in the middle of the repair, A may not claim the damage of the withered hedge.", 'N'], 
              ["In cases where an individual rescues another person from getting hit by a car by pushing that person out of the way, causing the person's luxury kimono to get dirty, the rescuer does not have to compensate damages for the kimono.", 'Y'],  
              ["A manager must engage in management exercising care identical to that he/she exercises for his/her own property.",'N'],
              ["Unless a Manager engages in the Management of Business in order to allow a principal to escape imminent danger to the principal's person, reputation or property, the manager must manage the business with due care of a prudent manager.", "Y"],
              ["If a manager engages in benevolent intervention in another's business in order to allow a principal to escape imminent danger to the principal's property, the manager is not liable to compensate for damage resulting from this unless the manager has acted in bad faith or with gross negligence.", "Y"],
              ["In cases where A found a collared dog whose owner is unknown, and took care of it for the unknown owner; if A took the dog to his/her house and takes care of it, it shall be sufficient if he/she takes care of it exercising care identical to that he/she exercises for his/her own property.", "N"]
              ],
    "paragraphs": "Article 698\nIf a manager engages in benevolent intervention in another's business in order to allow a principal to escape imminent danger to the principal's person, reputation, or property, the manager is not liable to compensate for damage resulting from this unless the manager has acted in bad faith or with gross negligence."
}
# module = Answerer(debug=True)
# module = Answerer(debug=True)
module = CallLLM()
# query=[q[0] for q in c["query"]]
# labels = [q[1] for q in c["query"]]
# query_list = ""
# label_list = ""
# for q,i in zip(query, range(0, len(query))):
#     query_list += f"Query {i+1}: {q}\n"
    
# for l,i in zip(labels, range(0, len(labels))):
#     label_list += f"correct label for query {i+1} is: {l}\n"
# query = c["query"]
# label_list = c["label"]

articles=c["paragraphs"]
expected=c["label"]
kill_ollama_gpu_task()
server_process = start_ollama_server()


try:
    # response = module.GenerateExamplesFromQuery(article=articles, queries=query_list, nb_of_factors=3, labels=label_list)
    cases = module.GenerateExamples(article=articles,  nb_of_factors=3)
    print(f"=============cases are: \n{cases}")
    factors_extracted = ExtractFactorFromText(cases)
    # print(f"=============factors are: \n{factors_extracted}")
    for q in c["query"]:
        if q[1] == "Y": 
            label = "true"
        else: label= "false"
        response = module.GenerateFactorFromReasoning(article=articles, labels=label, queries=q[0], nb_of_factors=3, factors=cases)
        print("============= Analyse factors of llm reasoning:")
        print('label is:', q[1])
        print(q[0])
        print(module.CheckPledged(article=articles, query=q[0]))
        print(f" \n{response}")

    # print(f'true label is: {expected}')
    
    # asp_label, llm_label = module.answer(article=articles, query=query)
    # print(f'asp_label:{asp_label} - llm_label: {llm_label}')
    
    stop_ollama_server(server_process)
except KeyboardInterrupt:
    stop_ollama_server(server_process)
    print("\nLoop interrupted. Exiting...")
    sys.exit()


