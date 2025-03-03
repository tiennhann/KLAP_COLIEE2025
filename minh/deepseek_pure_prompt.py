from ollama import Client

def forward(conversation):
    client = Client(host='http://localhost:8080')
    return client.chat(
            model='llama3:70b',
            messages=conversation,
            options={"temperature":2}
    )

def chat(article, query):
    # Create a client that connects to the Ollama server running on localhost.

    # Initialize an empty conversation history list
    conversation = []
    print("Start Answering_________________________________________________")
    user_input = f"""For each pair below, please determine whether the sentence is logically entailed by the given information. If the sentence logically follows from the information, respond with "entails". If it does not logically follow, respond with "does not entail". Only give answer and nothing else.

Example 1: Information:
(1) If the owner of a first thing attaches a second thing that the owner owns to the first thing to serve the ordinary use of the first thing, the thing that the owner attaches is an appurtenance.
(2) An appurtenance is disposed of together with the principal thing if the principal thing is disposed of.

Sentence: Extended parts of the building shall be regarded as appurtenance.

Expected Answer: does not entail

Example 2: Information:
(1) The exercise of a right of retention does not preclude the running of extinctive prescription of claims.

Sentence: Even while the holder of a right to retention continues the possession of the retained property, extinctive prescription runs for its secured claim.

Expected Answer: entails

Information:
{article}

Sentence:
{query}

Answer:""".format(article=article, query=query)
#    print("user_input:\n",user_input)
    conversation.append({'role': 'user', 'content': user_input})
    # Append the user's message to the conversation history
    response = []
    loop_num = 5
    for i in range(loop_num):
        response.append(forward(conversation)['message']['content'])
       # Send the entire conversation history to the model

    # Extract and print the model's reply
    return response
