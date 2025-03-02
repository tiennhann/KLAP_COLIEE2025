from ollama import Client

prompt = ""
prompt_path = "./duong/init_prompt_angelic_4.txt"
model="llama3.2"
with open(f"{prompt_path}", "r") as f:
    prompt = "".join(f.readlines())
    
def chat(article, query, failed_response=""):
    # Create a client that connects to the Ollama server running on localhost.
    client = Client(host='http://localhost:11434')
    # print("run with prompt:", prompt_path)
    # print("run with llm:", model)
    # Initialize an empty conversation history list
    conversation = []

    failed_message = ""
    if failed_response != "":
        failed_message = f"Retry the above task again. The following that you gave me wasn't a correct answer: {failed_response}"

    user_input = f"""
    Articles: {article} \n
    Query: {query}
    {prompt}
    {failed_message}
    """
    # Append the user's message to the conversation history
    conversation.append({'role': 'user', 'content': user_input})

    # Send the entire conversation history to the model
    response = client.chat(
            model=model,
        messages=conversation,
        stream=False
    )

    # Extract and print the model's reply
    reply = response['message']['content']
    return reply
