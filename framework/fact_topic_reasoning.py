import json
import os
import fire
from openai import OpenAI
import umap.umap_ as UMAP
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from data_process import dataset_process
from utils import openai_async_inference
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer

os.environ['OPENAI_API_KEY'] = 'API_KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def ground_truth_prompt(question, intent_conversation, llm_model_name):
    client = OpenAI()
    prompt = f'''
    Given the conversation: {intent_conversation} as background information.
    Can you answer the question: {question}?
    Please output the answer without any explaination, and answer should be less than 40 words. 
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model=llm_model_name,
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def get_fact(text):
    fact_list = []
    if text.find('(1). ') != -1:
        facts = text.split('(1). ')[1].split('). ')
        for fact in facts:
            fact_list.append(fact.split('\n')[0])
    elif text.find('(1) ') != -1:
        facts = text.split('(1) ')[1].split(') ')
        for fact in facts:
            fact_list.append(fact.split('\n')[0])
    elif text.find('1. ') != -1:
        facts = text.split('1. ')[1].split('. ')
        for fact in facts:
            fact_list.append(fact.split('\n')[0])
    else:
        fact_list.append(text)
    return fact_list


def get_distinct_facts(speaker_fact, threshold, model):
    embeddings = model.encode(speaker_fact)
    similarities = model.similarity(embeddings, embeddings)   
    sim_list = []
    list_pair = []
    for i, sim_score in enumerate(similarities):
        for j in range(len(sim_score)):
            if sim_score[j] > threshold and sim_score[j] < 0.99:
                if i < j:
                    sim = [i, speaker_fact[i],j , speaker_fact[j], sim_score[j]]
                    sim_list.append(sim)
                    list_pair.append([speaker_fact[i],  speaker_fact[j]])

    for info in sim_list:
        if info[1] in speaker_fact:
            speaker_fact.remove(info[1])
    return speaker_fact

class HierarchicalClusteringSummarizer:
    def __init__(self, emb_model, model_type: str = 'gpt-4o-mini'):
        """
        Initialize the hierarchical clustering summarizer.
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings
        """

        self.hierarchy: Dict[str, Any] = {}
        self.current_level = 0
        self.model_type = model_type
        self.emb_model = emb_model

    def summarized_topic_prompt(self, text):
        client = OpenAI()

        prompt = f'''
        Can you summarize {text} in one sentence to only contain the high-level information? 
        Please only output the summary without anything else.
        '''
        completion = client.chat.completions.create(
            # model="gpt-4-turbo-preview",
            model= self.model_type,
            messages=[
                {"role": "system", "content": "You are an expert in generate high level personal question."},
                {"role": "user", "content": prompt }
            ]
        )
        response = str(completion.choices[0].message.content)
        return response  


    def get_clustered_facts(self, fact_list):
        cluster_size = len(fact_list) // 6
        embeddings = self.emb_model.encode(fact_list)
        reducer = UMAP.UMAP(n_neighbors=15, min_dist=0.1, n_components= min(50, embeddings.shape[0]-5), random_state=42)
        reduced_embedding_umap = reducer.fit_transform(embeddings)
        clustering_model = GaussianMixture(n_components = cluster_size, random_state=42)
        clustering_model.fit(reduced_embedding_umap)
        predict_label = clustering_model.predict(reduced_embedding_umap)
        key_dic = {}
        for i, index in enumerate(predict_label):
            if index not in key_dic:
                key_dic[index] = [fact_list[i]]
            else:
                key_dic[index].append(fact_list[i])
        
        return key_dic

    def build_hierarchy(self, fact_list):
        current_fact_list = fact_list
        while(len(current_fact_list) > 15):
            cluster_groups = self.get_clustered_facts(current_fact_list)
            new_sentences = []
            new_dict = {}
            for cluster_label, cluster_sentences in cluster_groups.items():
                summary =self.summarized_topic_prompt(cluster_sentences)
                new_sentences.append(summary)
                new_dict[summary] = cluster_sentences
            self.hierarchy[f"level_{self.current_level}"] = {
                    "summaries": new_dict,
                }
            # Prepare for next iteration
            current_fact_list = new_sentences
            self.current_level += 1
            
        print(f"Level {self.current_level} completed")
        
        return self.hierarchy

    def select_fact_batch(self, question, facts):
        prompt = '''
        Given the information: {fact}.
        Could that affect the answer of the question the question that {question}?
        Please output only 'YES' or 'NO'.
        '''
        prompts = [prompt.format(fact=fact, question=question) for fact in facts]
        messages = []
        for inp in prompts:
            messages.append([
                {"role": "system", "content": "You are an expert in answering questions."},
                {"role": "user", "content": inp }
            ])
        checked_list = openai_async_inference(messages,
                                    model_name = self.model_type,
                                    tqdm_description="Calling OpenAI API",
                                    batch_size=-1) 
        
        sel_facts = []
        for i, response in enumerate(checked_list):
            if response.lower().find('yes') != -1:
                sel_facts.append(facts[i])
        return sel_facts
    
    # def select_fact(self, question, topic_list):
    #     client = OpenAI()
    #     prompt = f'''
    #     Given question {question}, can you select all topics below that are potentially related to the question?
    #     topics: {topic_list}
    #     Please only output the names of topics without changing any words. Each topic should be separated by a semicolon.
    #     '''
    #     completion = client.chat.completions.create(
    #         # model="gpt-4-turbo-preview",
    #         model= self.model_type,
    #         messages=[
    #             {"role": "system", "content": "You are an expert in selecting relevant topics."},
    #             {"role": "user", "content": prompt }
    #         ]
    #     )
    #     select_topic = str(completion.choices[0].message.content)
    #     select_keys  = select_topic.split(';')

    #     select_facts = []
    #     candidate_embeddings = self.emb_model.encode(topic_list)

    #     for key in select_keys:
    #         target_embedding = self.emb_model.encode(key)
    #         similarities = self.emb_model.similarity(target_embedding, candidate_embeddings)[0]
    #         best_match_idx = similarities.argmax()
    #         select_facts.append(topic_list[best_match_idx])

    #     return select_facts
    
    def find_related_facts(self, question):
        related_facts = []
        for key in self.hierarchy.keys():
            print(int(key.split('_')[1]))
        current_level = max(int(key.split('_')[1]) for key in self.hierarchy.keys())
        print(f'max_level is {current_level}')
        current_summaries = list(self.hierarchy[f"level_{current_level}"]["summaries"].keys())
        current_related = self.select_fact_batch(question, current_summaries)
        print(f'beginning_current_related {current_related}')
        while current_level > 0:
            next_level = current_level - 1
            next_summaries = []

            for summary in current_related:
                children = self.hierarchy[f"level_{current_level}"]["summaries"].get(summary, [])
                if children:
                    related_children = self.select_fact_batch(question, children)
                    next_summaries.extend(related_children)
                print(f'related_children {related_children}, summary {summary}, children {children}')

            
            current_related = next_summaries
            current_level = next_level
            print(len(current_related))
            print(f'current_related {current_related}')

            if current_level == 0:
                for summary in current_related:
                    related_facts.extend(self.hierarchy[f"level_{current_level}"]["summaries"][summary])
                break

        return list(set(related_facts)), current_related

    def get_hierarchy_stats(self):
        stats = {}
        max_level = max(int(key.split('_')[1]) for key in self.hierarchy.keys())
        for level in range(max_level + 1):
            level_key = f"level_{level}"
            if level_key in self.hierarchy:
                stats[level_key] = len(self.hierarchy[level_key]["summaries"])
        return stats

def rag_original(question, summarization_list, original_list, model, top_k=5):
    question_embeddings = model.encode(question)
    summary_embeddings = model.encode(summarization_list)
    original_embeddings = model.encode(original_list)
    sum_similarities = model.similarity(question_embeddings, summary_embeddings)[0]   
    org_similarities = model.similarity(question_embeddings, original_embeddings)[0]
    top_sum = sum_similarities.argsort(descending=True)[:top_k]
    top_org = org_similarities.argsort(descending=True)[:top_k]
    sel_sum_list = []
    for idx in top_sum:
        sel_sum_list.append(summarization_list[idx])
    sel_org_list = []
    for idx in top_org:
        sel_org_list.append(original_list[idx])
    return sel_sum_list, top_sum.tolist(), sel_org_list, top_org.tolist()

def fact_prompt(question, speaker1_fact, llm_model_name):
    client = OpenAI()
    prompt = f'''
    Speaker1 has the following personal traits: {speaker1_fact}.
    Given these traits as background information, can you answer the question: {question} based on the useful traits?
    Your answer should be aligned with the personal traits of speaker1.
    Please output the answer without any explaination, and answer should be less than 40 words. 
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model=llm_model_name,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def long_context_prompt(question, conversation, llm_model_name):
    client = OpenAI()
    prompt = f'''
    Given the previous conversations {conversation} as background information, can you answer the question: {question}?
    Your answer should also recall the previous conversation between Speaker1 and Assistant.
    Please output the answer without any explaination, and answer should be less than 40 words. 
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model=llm_model_name,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def rag_sumy_prompt(question, summary, llm_model_name):
    client = OpenAI()
    prompt = f'''
    Given the related summary of Speaker1 and Assistant's conversation: {summary} as background information, can you answer the question: {question}?
    Your answer should be aligned with the personal traits of Speaker1 and Assistant.
    Your answer should also recall the previous conversation between Speaker1 and Assistant.
    Please output the answer without any explaination, and answer should be less than 40 words. 
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model=llm_model_name,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def rag_original_prompt(question, original, llm_model_name):
    client = OpenAI()
    prompt = f'''
    Given the related conversation of Speaker1 and Assistant: {original} as background information, can you answer the question: {question}?
    Your answer should be aligned with the personal traits of Speaker1 and Assistant.
    Your answer should also recall the previous conversation between Speaker1 and Assistant.
    Please output the answer without any explaination, and answer should be less than 40 words. 
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model=llm_model_name,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response
   

def main(
        home_dir = './datasets/impConv',
        dataset_name = 'syn_reasoning',
        model_type = 'gpt-4o-mini',
        summy_info = 'summarized_reasoning_facts.json',
        output_response_file = 'reasoning_full_response.json',
        output_retrieve_file = 'reasoning_retrieve_text.json'
    ):

    conversations, questions = dataset_process(home_dir, dataset_name)
    with open(f'{home_dir}/{dataset_name}/{summy_info}', 'r') as f:
        sum_fact = json.load(f)

    print(dataset_name)
    if os.path.exists(f'{home_dir}/{dataset_name}/{output_response_file}'):
        with open(f'{home_dir}/{dataset_name}/{output_response_file}', 'r') as f:
            final_response = json.load(f)
    else:
        final_response = []

    if os.path.exists(f'{home_dir}/{dataset_name}/{output_retrieve_file}'):
        with open(f'{home_dir}/{dataset_name}/{output_retrieve_file}', 'r') as f:
            retrieve_text = json.load(f)
    else:
        retrieve_text = []
        
    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    for index, uni_fact in tqdm(enumerate(sum_fact[len(final_response):500]), desc='Processing conversations'):
        speaker1_fact = []
        for conv in uni_fact:
            speaker1 = conv['Speaker1_fact']
            speaker1_fact.extend(get_fact(speaker1))

        speaker1_fact = get_distinct_facts(speaker1_fact, 0.9, model)  

        summarizer = HierarchicalClusteringSummarizer(model, model_type)
        hierarchy = summarizer.build_hierarchy(speaker1_fact)

        summarization_list = []
        for item in uni_fact:
            summarization_list.append(item['summary'])

        prev_conversation = ''
        for conv in conversations[index]:
            prev_conversation += conv


        response_texts = []
        retrevial_texts = []
        for qa_pair in questions[index]:
            all_response_dic = {'generate_answer': ground_truth_prompt(qa_pair['question'], qa_pair['target_conv'], model_type)}
            all_response_dic['question'] = qa_pair['question']
            all_response_dic['ground_truth'] = qa_pair['answer']
            all_retrevial_dic = {'question': qa_pair['question']}
            all_retrevial_dic['ground_truth_conv'] = qa_pair['target_conv']
            all_retrevial_dic['ground_truth_answer'] = qa_pair['answer']
            select_facts1, select_summary1 = summarizer.find_related_facts(qa_pair['question'])

            sel_summary_response_text = fact_prompt(qa_pair['question'], select_summary1, model_type)
            all_response_dic['select_summary'] = sel_summary_response_text
            all_retrevial_dic['select_summary'] = select_summary1
            print(f'sel_summary_response_text {sel_summary_response_text}')

            sel_fact_response_text = fact_prompt(qa_pair['question'], select_facts1, model_type)
            all_response_dic['select_fact'] = sel_fact_response_text
            all_retrevial_dic['select_fact'] = select_facts1
            print(f'sel_fact_response_text {sel_fact_response_text}')

            all_fact_response_text = fact_prompt(qa_pair['question'], speaker1_fact, model_type)
            all_response_dic['all_fact'] = all_fact_response_text
            all_retrevial_dic['all_fact'] = speaker1_fact
            print(f'all_fact_response_text {all_fact_response_text}')

            long_content_response_text = long_context_prompt(qa_pair['question'], prev_conversation, model_type)
            all_response_dic['long_content'] = long_content_response_text
            print(f'long_content_response_text {long_content_response_text}')

            sel_sum_list, sum_index, sel_org_list, rag_index = rag_original(qa_pair['question'], summarization_list, conversations[index], model, top_k=5)
            rag_sumy_response_text = rag_sumy_prompt(qa_pair['question'], sel_sum_list, model_type)
            all_response_dic['rag_sumy'] = rag_sumy_response_text
            all_retrevial_dic['rag_sumy'] = sel_sum_list
            all_retrevial_dic['rag_sumy_index'] = sum_index
            print(f'rag_sumy_response_text {rag_sumy_response_text}')

            rag_org_response_text = rag_original_prompt(qa_pair['question'], sel_org_list, model_type)
            all_response_dic['rag_org'] = rag_org_response_text
            all_retrevial_dic['rag_org'] = sel_org_list
            all_retrevial_dic['rag_org_index'] = rag_index
            print(f'rag_org_response_text {rag_org_response_text}')

            response_texts.append(all_response_dic)
            retrevial_texts.append(all_retrevial_dic)

        final_response.append(response_texts)
        retrieve_text.append(retrevial_texts)

        with open(f'{home_dir}/{dataset_name}/{output_response_file}', 'w') as f:
            json.dump(final_response, f, indent=4)

        with open(f'{home_dir}/{dataset_name}/{output_retrieve_file}', 'w') as f:
            json.dump(retrieve_text, f, indent=4)
    


if __name__ == "__main__":
    fire.Fire(main)