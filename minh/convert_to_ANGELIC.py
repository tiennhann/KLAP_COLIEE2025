from ollama import Client

prompt = ""
prompt_angelic_path = "./duong/init_prompt_angelic_4.txt"
prompt_query_path ="./duong/prompt_query_entail_bfactor.txt"
model="deepseek-r1:70b"
with open(f"{prompt_angelic_path}", "r") as f:
    prompt_angelic = "".join(f.readlines())
    
with open(f"{prompt_query_path}", "r") as f:
    prompt_query = "".join(f.readlines())
def chat(article, query):
    # Create a client that connects to the Ollama server running on localhost.
    client = Client(host='http://localhost:11435')
    print("run with angelic structure conversion prompt:", prompt_angelic_path)
    print("run with query entailment prompt:", prompt_query_path)
    print("run with llm:", model)
    # Initialize an empty conversation history list
    conversation = []

    prompt = f"""
    {article} \n
    {prompt_angelic}
    """
    # Append the user's message to the conversation history
    conversation.append({'role': 'user', 'content': prompt})

    # Send the entire conversation history to the model
    response = client.chat(
            model=model,
        messages=conversation,
        stream=False
    )

    # Extract and print the model's reply
    angelicSt = response['message']['content']
    
    # Initialize an empty conversation history list
    conversation = []

    prompt = f"""
    query is:
    {query} \n
    Base-level factors is includes here:
    {angelicSt}
    {prompt_query}
    """
    # Append the user's message to the conversation history
    conversation.append({'role': 'user', 'content': prompt})
    
    response = client.chat(
            model=model,
        messages=conversation,
        stream=False
    )

    # Extract and print the model's reply
    queryEntails = response['message']['content']
    
    reply = angelicSt + queryEntails
    return reply
