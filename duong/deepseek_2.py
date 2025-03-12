from ollama import Client


prompt_path = "./duong/init_prompt_angelic_7.txt"
model="llama3.3"



def chat(article, query, prompt_input_path=prompt_path ,failed_response=""):
    with open(f"{prompt_input_path}", "r") as f:
        prompt = "".join(f.readlines())
    # Create a client that connects to the Ollama server running on localhost.
    client = Client(host='http://localhost:11435')
    # print("run with prompt:", prompt_path)
    # print("run with llm:", model)
    # Initialize an empty conversation history list
    conversation = []

    failed_message = ""
    if failed_response != "":
        failed_message = f"Reconstruct the structure again. The one following haves incorrect directed links: {failed_response}"

    user_input = f"""
    Given the civil legal article context
    : {article} \n
    and query: {query}

    {failed_message}
    {prompt}
    
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
