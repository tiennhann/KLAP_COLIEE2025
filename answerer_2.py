# from duong.gen_angelic_st import GenAngelic
# from duong.deepseek_2 import GenerateFactor, GenerateExamples, GenerateQueryEntailments
import json
from extractor.extractor import MDAExtractor
from reasoner.reasoner import ASPSolver
import dspy
from duong.deepseek_2 import CallLLM
import re
from rouge_score import rouge_scorer
# chat = GenAngelic().init()
class Answerer:
    def __init__(self, debug=False):
        self.debug = debug
        self.extractor_examples = MDAExtractor(file_path_input='extraction_example_fact_task_definition.txt')
        self.extractor_query_entailment = MDAExtractor(file_path_input='extraction_query_fact_task_definition.txt')
        self.solver = ASPSolver()
        self.chat = CallLLM()
        print(f'prompt to extract factors: {self.chat.prompt_path_factor}')
        print(f'prompt to extract examples: {self.chat.prompt_path_example}')
        print(f'prompt to extract asp fact of examples: {self.extractor_examples.prompt_path}')
        print(f'asp program to generate Angelic-based link: {self.solver.getlink_path}')
        print(f'prompt to extract query entailment: {self.chat.prompt_path_query_entailment}')
        print(f'prompt to extract asp fact of query entailment: {self.extractor_query_entailment.prompt_path}')
        print(f'asp program to predict label: {self.solver.getlabel_path}')
        
    def answer(self, article, query):
        answer_from_factors_generator = self.chat.GenerateFactor(article=article, query=query)
        if self.debug:
            print("======== Answer from Angelic-based factors generator: ========\n", answer_from_factors_generator)
        factors = ExtractFactorFromText(answer_from_factors_generator)
        if self.debug:
            print("======== Factors:\n", factors)
        examples = self.chat.GenerateExamples(article=article, nb_of_factors=4)
        if self.debug:
            print("======== Generate Examples:\n", examples)
        
        link = 'UNSAT'
        count_link_generation = 0
        while link == 'UNSAT':
            count_link_generation +=1
            facts = self.extractor_examples.forward(text=factors + "\n" + examples)
            link = self.solver.GetLink(facts)
            print(f'generating link process {count_link_generation}th')
            if count_link_generation > 5: break
        if self.debug:
            print("======= ASP facts from factors and examples ========\n", facts)
        print("====== link generated:\n", link)

        query_entailments = self.chat.GenerateQueryEntailments(factors=factors, query=query, article=article)
        if self.debug:
            print("======= Query Entaiments:\n", query_entailments)
            
        query_entailm_facts = self.extractor_query_entailment.forward(text=query_entailments)
        # if self.debug:
        print("======= ASP facts from query entailments:\n", query_entailm_facts)
        
        # get label
        if link == 'UNSAT':
            label = 'E'
        else:
            label = self.solver.GetLabel(link+query_entailm_facts)
        
        return label, self.chat.GetAnswerLLM(article=article, query=query)

    def answer_2(self, article, queries_examples, labels_examples,query_test, label_test):
        # generate factors/cases
        examples = self.chat.GenerateExamples(article=article)
        if self.debug:
            print("========Examples:")
            print(examples)
        facts = self.extractor_examples.forward(text=examples)
        if self.debug:
            print("========ASP facts of Examples:")
            print(facts)
        link = self.solver.GetLink(facts)
        if self.debug:
            print("========Angelic-based link:")
            print(link)
        query_entailments_examples =""
        
        query_entailments_examples += ''.join(
            self.chat.GenerateFactorFromReasoning(article=article, labels=labels_examples[i], queries=queries_examples[i], factors=examples)
            for i in range(0, len(queries_examples))
        )
        if self.debug:
            print("======== Query Entailment Example:")
            print(query_entailments_examples)
        query_entailments = self.chat.GenerateQueryEntailments(factors=ExtractFactorFromText(examples), query=query_test, article=article, examples=query_entailments_examples)
        if self.debug:
            print("======= Query entailments of test query is:")
            print(query_entailments)
        pledged_check = self.chat.CheckPledged(article=article, query=query_test)
        if self.debug:
            print("=======  Pledge Check of test query is:")
            print(pledged_check)
        query_entailments_asp_facts = self.extractor_query_entailment.forward(text=query_entailments+pledged_check)
        if self.debug:
            print("======= Query entailments ASP facats:")
            print(query_entailments_asp_facts)
        label = self.solver.GetLabel(link+query_entailments_asp_facts)
        
        return label, self.chat.GetAnswerLLM(article=article, query=query_test)
    
    def CallOneShotLLM(self, article, query_list):
        answer = self.chat.Ask(article, query_list)
        return answer
    
def ExtractFactorFromText(text):
    # This pattern captures everything from Begin to End of Factoring, including the number of factors line
    pattern = r"(### Begin Factoring.*?### End of Factoring)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def EvaluateFactorSet(article, factors):
    """
    Evaluate the summary by comparing it to the reference text using ROUGE metrics.
    
    Parameters:
        reference_text (str): The original text to compare against.
        summary_text (str): The summary produced from the original text.
    
    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = rouge.score(factors, article)
    return scores


if __name__ == "__main__":
    dspy.settings.configure(
        lm=dspy.LM(
            model="ollama_chat/llama3.3",  # Changed to include provider prefix
            api_base="http://localhost:11435",
            max_tokens=20000
        )
    )

    # answerer = Answerer(debug=True)
    # answer = answerer.answer("Article 356\n The pledgee of immovables may use and profit from the immovables that are the subject matter of a pledge in line with the way the relevant immovables are used.", "A  pledgee of immovable peroperty may use and receive the profits from the immovable property that is the subject matter of a pledge, in accordance with the method of its use, but may not collect fruits derived by the real property.")

    # print(answer)
    
    print(EvaluateFactorSet("Article 698\nIf a manager engages in benevolent intervention in another's business in order to allow a principal to escape imminent danger to the principal's person, reputation, or property, the manager is not liable to compensate for damage resulting from this unless the manager has acted in bad faith or with gross negligence.", "1. The manager's intervention was benevolent and aimed at allowing the principal to escape imminent danger.\n2. The manager acted in bad faith or with gross negligence during the intervention.\n3. Damage resulted from the manager's intervention."))
    
    
    