import json
import sys
import time
from answerer import Answerer
import random
import dspy
import traceback

# Configure DSPy
dspy.settings.configure(
    lm=dspy.LM(
        model="ollama_chat/llama3:70b",  # Changed to include provider prefix
        api_base="http://localhost:11435",
        max_tokens=20000
    )
)

# Configuration files
extractor_prompt_filename = "extraction_task.txt"
theory_filename = "theory_2.lp"
prompt_filename = "./duong/init_prompt_angelic_4.1.txt"

# Output file for results
output_file = "R02.task4.H1"  # Change this to your desired output filename

# Load test cases
cases = []
with open('test_R02.json') as f:
    cases = json.load(f)

# Initialize counters
correct = 0
wrong = 0
error = 0
max_retries = 3  # Maximum number of retry attempts
retry_delay = 5   # Seconds to wait between retries

# Dictionary to store results
results = {}

# Initialize the answerer module
module = Answerer(extractor_prompt_filename=extractor_prompt_filename, 
                 theory_filename=theory_filename, 
                 prompt_filename=prompt_filename, 
                 debug=True)

# Process each test case
for idx, c in enumerate(cases):
    query = c["query"]
    articles = c["paragraph"]
    case_id = c["id"]
    
    print(f"\n--- Case {idx+1}/{len(cases)}: {case_id} ---")
    print("Article:", articles[:100] + "..." if len(articles) > 100 else articles)
    print("Query:", query)
    
    # Initialize retry counter
    retries = 0
    success = False
    label = "N"  # Default label if all attempts fail
    
    # Try processing the case with retries
    while retries < max_retries and not success:
        try:
            if retries > 0:
                print(f"Retry attempt {retries}/{max_retries}...")
            
            l = module.answer(article=articles, query=query)
            print("Answer:", l)
            
            # Store the result
            label = l
            results[case_id] = label
            
            # If we get here, the call was successful
            success = True
            
            # Save to file immediately after each successful answer
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{case_id} {label}\n")
            print(f"Result for case {case_id} saved to {output_file}")
            
        except KeyboardInterrupt:
            print("\nLoop interrupted. Exiting...")
            # Save the results we have so far before exiting
            with open(output_file, 'w', encoding='utf-8') as f:
                for id, lbl in results.items():
                    f.write(f"{id} {lbl}\n")
            sys.exit()
            
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed after {max_retries} attempts for case {case_id}. Error: {str(e)}")
                error += 1
                print(traceback.format_exc())
                
                # Store N result for failed cases
                results[case_id] = label
                
                # Also save to file after maximum retries are exhausted
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{case_id} {label}\n")