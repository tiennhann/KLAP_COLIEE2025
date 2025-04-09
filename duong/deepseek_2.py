from ollama import Client

model="llama3.3"

class CallLLM:
    def __init__(self):
        self.prompt_path_factor = '/home/anguyen/KLAP_COLIEE2025/duong/prompt_generate_factor_Angelic-based.txt'
        self.prompt_path_example = '/home/anguyen/KLAP_COLIEE2025/duong/prompt_generate_examples_Angelic-based.txt'
        self.prompt_path_query_entailment = '/home/anguyen/KLAP_COLIEE2025/duong/prompt_query_entail_factor.txt'

    def CallOneShotLLM(self, article, query_list):
      # Create a client that connects to the Ollama server running on localhost.
        client = Client(host='http://localhost:11435')
        # Initialize an empty conversation history list
        conversation = []
        # print("Start Answering_________________________________________________")
        user_input = f"""For given legal article and query pair below, please determine whether the query is legally valid with the context the article. If the query logically follows the legal context, respond with "Y". If it does not logically follow, respond with "N". Explain why you response with that answer by first factoring the article into factors, then reasoning what factor appear or missing that cause the query validate or invalidate the article. Be skeptical, if there is a small evidence, still consider contain that factor.

    Example 1:
    Article 87:
    (1) If the owner of a first thing attaches a second thing that the owner owns to the first thing to serve the ordinary use of the first thing, the thing that the owner attaches is an appurtenance. 
    (2) An appurtenance is disposed of together with the principal thing if the principal thing is disposed of.
    Query: Extended parts of the building shall be regarded as appurtenance.
    Expected Answer: N

    Example 2:
    Article 300:
    (1) The exercise of a right of retention does not preclude the running of extinctive prescription of claims.
    Query: Even while the holder of a right to retention continues the possession of the retained property, extinctive prescription runs for its secured claim.
    Expected Answer: Y

    Provided article:
    {article}

    Query:
    {query_list}

    Answer:""".format(article=article, query=query_list)
    #    print("user_input:\n",user_input)
        conversation.append({'role': 'user', 'content': user_input})
        response = client.chat(
                model=model,
            messages=conversation,
            stream=False
            )
        reply = response['message']['content']
        # Send the entire conversation history to the model

        # Extract and print the model's reply
        return reply        

    def GetAnswerLLM(self, article, query):
        # Create a client that connects to the Ollama server running on localhost.
        client = Client(host='http://localhost:11435')
        # Initialize an empty conversation history list
        conversation = []
        print("Start Answering_________________________________________________")
        user_input = f"""For each pair below, please determine whether the sentence is logically entailed by the given information. If the sentence logically follows from the information, respond with "Y". If it does not logically follow, respond with "N". Only give answer and nothing else.

    Example 1: Information: 
    Article 87:
    (1) If the owner of a first thing attaches a second thing that the owner owns to the first thing to serve the ordinary use of the first thing, the thing that the owner attaches is an appurtenance. 
    (2) An appurtenance is disposed of together with the principal thing if the principal thing is disposed of.

    Sentence: Extended parts of the building shall be regarded as appurtenance.

    Expected Answer: N

    Example 2: Information: 
    Article 300:
    (1) The exercise of a right of retention does not preclude the running of extinctive prescription of claims.

    Sentence: Even while the holder of a right to retention continues the possession of the retained property, extinctive prescription runs for its secured claim.

    Expected Answer: Y

    Information: 
    {article}

    Sentence:
    {query}

    Answer:""".format(article=article, query=query)
    #    print("user_input:\n",user_input)
        conversation.append({'role': 'user', 'content': user_input})
        response = client.chat(
                model=model,
            messages=conversation,
            stream=False
            )
        reply = response['message']['content']
        # Send the entire conversation history to the model

        # Extract and print the model's reply
        return reply

    def GenerateFactor(self, article, query, nb_of_factors=4, prompt_input_path='/home/anguyen/KLAP_COLIEE2025/duong/prompt_gen_factor_Angelic-based.txt'):

        with open(f"{prompt_input_path}", "r") as f:

            prompt = "".join(f.readlines())
        # print("prompt to generate factors:", prompt_input_path)
        # print("model to generate first structure:", model)
        # Create a client that connects to the Ollama server running on localhost.
        client = Client(host='http://localhost:11435')
        # print("run with prompt:", prompt_path)
        # print("run with llm:", model)
        # Initialize an empty conversation history list
        conversation = []


        user_input = f"""
        {article} \n
        Queries is:
        {query} \n
        For given legal article and query below, the query logically follows the legal context, respond with "Y". If it does not logically follow, respond with "N". Explain why you response with that answer by first factoring the article into {nb_of_factors} factors, then reasoning what factor cause the query logical or not.
            Example 1: Information: 
            Article 87:
            (1) If the owner of a first thing attaches a second thing that the owner owns to the first thing to serve the ordinary use of the first thing, the thing that the owner attaches is an appurtenance. 
            (2) An appurtenance is disposed of together with the principal thing if the principal thing is disposed of.

            Query: Extended parts of the building shall be regarded as appurtenance.

            Expected Answer: N

            Example 2:
            Article 300:
            (1) The exercise of a right of retention does not preclude the running of extinctive prescription of claims.

            Query: Even while the holder of a right to retention continues the possession of the retained property, extinctive prescription runs for its secured claim.

            Expected Answer: Y

        Requirement:
        1. If multiple articles are provided, maintain a single factor order rather than separating them by article.

        2. Present the number of factors in the format: <Number of factors : X> where X is the total number of factors.

        3. Representing it in the format:
        ### Begin Factoring
        <here is the list of factors>
        ### End of Factoring
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

    def GenerateExamplesFromQuery(self, article, queries, nb_of_factors, labels):
        client = Client(host='http://localhost:11435')
        conversation = []

        user_input =f"""
        You are a legal-reasoning assistant with expertise in analyzing statutory and regulatory language. Given following:
        {article}\n
        Queries:\n
        {queries}\n
        # Task 1 (Reasoning):
        Given the legal article, its factor list, and the list of queries:
        1) Factorization: Break down the provided article into {nb_of_factors} distinct factors.

        2) Query Analysis: For each query:
            - The article 698 implicitly impose the legal provision to the queries.
            - identifying whether each factors from the provided factor list present within the query. Evaluate your confidence value (0-100%) for your justification of each factors.
            - Clearly determine if the query explicitly or implicitly concludes that the juridical acts (or obligations) should be charged or not charged based on the article. Evaluate your confidence value (0-100%) of your justification(charged and not charged).
            - You'll be provided with the correct logical response (Y or N) for the query.

        A query is logical (Y) if the identified factors from the query align coherently with its conclusion (charged or not charged); otherwise, it's not logical (N).

        Provide concise and coherent justification for your assessment of the identified factors and the final decision of each query.
        
        
        # (Formatted Response):
        You need to output each query's analyse into this requirement format:

        1) **(query Q entails X)** if query Q has factor X, X is the factor number.
        2) **(query Q Charged)** if according to the article, the query explicitly or implicitly determinse that the juridical acts (or obligations) will be charged; or as **(query Q Not Charged)** if the query explicitly or implicitly determines that the juridical acts (or obligations) will not being charged.
        3) **(query Q N)** if reponse is N or **(query Q Y)** for response Y.
        4) You should put the output in the end of each query for better read.
        

        """
        # # Task 4(Reevaluate the analyse with correct response)
        # Given the correct response: 
        # You change the factors that query contains, so that your response match the provided response (which is true). It could be irrelevant, so find the sub-optimal explanation.
        
        
        # print(user_input)
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
    def ReturnLabelGivenNbOfFactors(self, article, queries, nb_of_factors):
        client = Client(host='http://localhost:11435')
        conversation = []

        user_input =f"""
        
        {article} \n
        Queries is:
        {queries} \n
        For given legal article and query pair below, please determine whether the query is legally valid with the context the article. If the query logically follows the legal context, respond with "Y". If it does not logically follow, respond with "N". Explain why you response with that answer by first factoring the article into {nb_of_factors} factors, then reasoning what factor appear or missing that cause the query validate or invalidate the article. Be skeptical, if there is a small evidence, still consider contain that factor.

            Example 1: Information: 
            Article 87:
            (1) If the owner of a first thing attaches a second thing that the owner owns to the first thing to serve the ordinary use of the first thing, the thing that the owner attaches is an appurtenance. 
            (2) An appurtenance is disposed of together with the principal thing if the principal thing is disposed of.

            Query: Extended parts of the building shall be regarded as appurtenance.

            Expected response: N

            Example 2:
            Article 300:
            (1) The exercise of a right of retention does not preclude the running of extinctive prescription of claims.

            Query: Even while the holder of a right to retention continues the possession of the retained property, extinctive prescription runs for its secured claim.

            Expected response: Y

        Requirement:
        1) If multiple articles are provided, maintain a single factor order rather than separating them by article.
        2) Present the number of factors in the format: <Number of factors : X> where X is the total number of factors.
        3) Representing it in the format:
        ### Begin Factoring
        <here is the list of factors>
        ### End of Factoring
        
        
        
        1) **(query Q entails X)** if query Q has factor X, X is the factor number.
        2) **(query Q Charged)** if according to the article, the query explicitly or implicitly decide that the juridical acts (or obligations) will be charged; or as **(query Q Not Charged)** if the query explicitly or implicitly determines that the juridical acts (or obligations) will not being charged, 
        3) **(query Q N)** if reponse is N or **(query Q Y)** for response Y.
        4) You should put the output in the end of each query for better read.
        
        """

        # print(user_input)
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
    
    def GenerateExamples(self, article, number_of_cases=10, prompt_input_path='/home/anguyen/KLAP_COLIEE2025/duong/prompt_gen_examples_Angelic-based.txt'):

        with open(f"{prompt_input_path}", "r") as f:

            prompt = "".join(f.readlines())
        # print("prompt to generate examples:", prompt_input_path)
        # print("model to generate first structure:", model)
        # Create a client that connects to the Ollama server running on localhost.
        client = Client(host='http://localhost:11435')
        # print("run with prompt:", prompt_path)
        # print("run with llm:", model)
        # Initialize an empty conversation history list
        conversation = []


        user_input = f"""
        legal context is: {article} \n

        You are a legal-reasoning assistant with expertise in analyzing statutory and regulatory language. Be concise.

        Instructions:
        Task 1: Factor the provided article(s) into set of factors that collectively cover all relevant aspects necessary to determine whether the case should be charged or not charged. Aim to minimize the number of factors while maintaining completeness and clarity.
        Requirement:
        1. If multiple articles are provided, maintain a single factor order rather than separating them by article.
        2. Representing it in the format:
        ### Begin Factoring
        
        <here is the list of factors>
        <Number of factors : X>
        
        ### End of Factoring
        
        where X is the total number of factors.
        Task 2:
        Given that set of factors from the article, generate {number_of_cases} case scenarios. For each case scenario:

            1. Assume that a subset of these factors occurred. Be concise, use only given factors. Justifying the juridical act (or obligations) will be charged (i.e., "pledged") or not charged based on the articles and the occurrence of these factors. Then, present it using the following formal output format:
            Ex <i>: (<factor x>, <factor y>, ... | true)
            where:
                1) x, y, ... denote the factor numbers present in that case.
                2) true indicates juridical act (or obligations) will be charged (i.e., "pledged")(use false if not) 
                3) i is the order of example.
                4) If the factor not present then you do not need to include it in the output format.

            2. Provide a justification determining whether 
            3. Ensure that no two cases use the same combination of factors.

        
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
    def GenerateFactorFromReasoning(self, article, labels, queries, factors):
        client = Client(host='http://localhost:11435')
        conversation = []

        user_input =f"""
        Given the legal context:
        {article} \n
        The factors are:\n
        {factors}
        The statement:
        {queries} \n
        is:
        {labels}\n
        - Explain why the statement is {labels} by indicating what factors are the evidence.\n

        - Reasoning what factors in the factors list entailed by query in the explanation. Entailment means factor is considered as an evidence (not an assumption, or condition). Answer only **(query entails X)** if query entails factor X or **(query does not entails X)** if query does not entails X, where X is the factor number.\n
        
        Requirement:
        You put all formatted output into the box:
        ### Begin analysis
        
        <formatted output, only formatted output>
        
        ### End analysis
        """
            # 4) You should put the output in the end of each query for better read.
        # - Identify what the query implies: the juridical acts will be charged or not charged. Then, you answer with only: **(query Charged)** if the query explicitly or implicitly determinse that the juridical acts (or obligations) will be charged; or **(query Not Charged)** if the query explicitly or implicitly determines that the juridical acts (or obligations) will not being charged.\n
        # print(user_input)
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
    def CheckPledged(self, article, query):
        client = Client(host='http://localhost:11435')
        conversation = []

        user_input =f"""
        Given the legal context:
        {article} \n
        The statement is: 
        {query}\n
        Does the statement decide the juridical acts is pledged or not pledged?. reply in **Pledged** or **Not Pledged**. If you can not decide reply **Unknown**.


        ### End analysis
        """

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
    
    def GenerateQueryEntailments(self, factors, query, article, examples, prompt_input_path="/home/anguyen/KLAP_COLIEE2025/duong/prompt_gen_entailment_of_query_factor.txt"):

        with open(f"{prompt_input_path}", "r") as f:

            prompt = "".join(f.readlines())
        # print("prompt to generate query entailments:", prompt_input_path)
        # print("model to generate first structure:", model)
        # Create a client that connects to the Ollama server running on localhost.
        client = Client(host='http://localhost:11435')
        # print("run with prompt:", prompt_path)
        # print("run with llm:", model)
        # Initialize an empty conversation history list
        conversation = []


        user_input = f"""
        Article is: {article}
        and factor included in:
        {factors} \n
        and query: {query}

        Task Definition:

        Given a query and a list of factors, Reasoning what factors in the factors list entailed by query in the explanation. Entailment means factor is considered as an evidence (not an assumption, or condition). Maintain a skeptical perspective; even minimal evidence should be enough to consider the presence of that factor.
        Output format: "(Query entails X)" or "(Query not entails X)" where X is the order number of factor in the factor list.
        
        Examples:\n
        {examples}
        
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