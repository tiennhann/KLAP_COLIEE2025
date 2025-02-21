from ollama import Client

# Create a client that connects to the Ollama server running on localhost.
client = Client(host='http://localhost:11434')

# Initialize an empty conversation history list
conversation = []

print("Type your messages below (type 'exit' to quit):")
user_input = """
given the following articles:
Article:
If a manager engages in benevolent intervention in another's business in order to allow a principal to escape imminent danger to the principal's person, reputation, or property, the manager is not liable to compensate for damage resulting from this unless the manager has acted in bad faith or with gross negligence.

Task definition: From those articles, generate the ANGELIC structure (ANGELIC is a framework in law AI domain)  which must satisfy:
1/ includes the root node, abstract factors, base-level factors and directed link between them (attack or support type).
2/ There is no directed link from root node to abstract factors or base-level factors.
3/ There is no directed link between abstract factors.
4/ There is no directed link between base-level factors.
5/ There is no directed link from abstract factors to base-level factors.
6/ Denote abstract factor by A1 for first abstract factor, A2 for second and etc.
7/ Denote base-level factor by B1 for first base-level factor, B2 for second one and etc.
8/ Denote root by R1 for first root and R2 for second root if needed.
9/ mention denoted number for each link.
10/ There is directed link from query to each base-level factors.
"""
while True:
    # user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Append the user's message to the conversation history
    conversation.append({'role': 'user', 'content': user_input})

    # Send the entire conversation history to the model
    response = client.chat(
            model='llama3:70b',
        messages=conversation
    )

    # Extract and print the model's reply
    reply = response['message']['content']
    print("AI:", reply)

    # Append the model's reply to the conversation history
    conversation.append({'role': 'assistant', 'content': reply})
    break
