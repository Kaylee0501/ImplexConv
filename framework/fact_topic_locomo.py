import json
import os
import fire
from openai import OpenAI
from sklearn.mixture import GaussianMixture
from data_process import dataset_process
from utils import evaluate_bleu, calculate_rouge_l
from sentence_transformers import SentenceTransformer

os.environ['OPENAI_API_KEY'] = ' REMOVED'
os.environ['SAMBANOVA_API_KEY'] = 'bfeb45e5-df9c-4193-b249-73fdbb6b78e1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_fact(text):
    fact_list = []
    if '(1).' not in text:
        return fact_list
    facts = text.split('(1). ')[1].split('). ')
    for fact in facts:
        fact_list.append(fact.split('\n')[0])
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

def summarized_topic(text, model_type = 'gpt-4o-mini'):
    client = OpenAI()
    prompt = f'''
    Can you summarize {text} in one high-level topic for a person?
    Please only output the topic and not the details.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model= model_type,
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response 

def select_facts(question, topic_list, topic_dic, encode_model, model_type = 'gpt-4o-mini'):
    client = OpenAI()
    prompt = f'''
    Given question {question}, can you select topics below that are possibly related to the answer to the question?
    topics: {topic_list}
    Please only output the full names of topics without changing any words. Each topic should be separated by a semicolon.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model= model_type,
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    select_topic = str(completion.choices[0].message.content)
    select_keys  = select_topic.split(';')

    select_facts = []
    #list all the keys in topic_dic:
    all_keys = []
    for key in topic_dic.keys():
        all_keys.append(key)
    candidate_embeddings = encode_model.encode(all_keys)

    for key in select_keys:
        target_embedding = encode_model.encode(key)
        candidate_embeddings = encode_model.encode(all_keys)
        similarities = encode_model.similarity(target_embedding, candidate_embeddings)[0]
        best_match_idx = similarities.argmax()
        best_match_string = all_keys[best_match_idx]
        select_facts.extend(topic_dic[best_match_string])

    return select_facts

def get_clustered_facts(speaker_fact, model, cluster_num, model_type):
    if len(speaker_fact) < cluster_num:
        cluster_num = len(speaker_fact)
    embeddings = model.encode(speaker_fact)
    clustering_model = GaussianMixture(n_components = cluster_num, random_state=0)
    clustering_model.fit(embeddings)
    predict_label = clustering_model.predict(embeddings)
    key_dic = {}
    for i, index in enumerate(predict_label):
        if index not in key_dic:
            key_dic[index] = [speaker_fact[i]]
        else:
            key_dic[index].append(speaker_fact[i])
    summarized_facts = {}   
    for key in key_dic.keys():
        summarized_facts[key] = summarized_topic(key_dic[key], model_type)
    topic_dic = {}
    for key, value in key_dic.items():
        topic_dic[summarized_facts[key]] = value

    return summarized_facts, topic_dic

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
        if sum_similarities[idx] > 0.35:
            sel_sum_list.append(summarization_list[idx])
    sel_org_list = []
    for idx in top_org:
        if org_similarities[idx] > 0.35:
            sel_org_list.append(original_list[idx])
    return sel_sum_list, sel_org_list

def fact_prompt(question, speaker1_name, speaker1_fact, speaker2_name, speaker2_fact):
    client = OpenAI()
    prompt = f'''
    {speaker1_name} has the following personal traits: {speaker1_fact}.
    {speaker2_name} has the following personal traits: {speaker2_fact}.
    Can you answer the question: {question} based on the useful traits?
    Your answer should be aligned with the personal traits of {speaker1_name} and {speaker2_name}.
    The response should be in one sentence with no more than 15 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def long_context_prompt(conversation, question, speaker1, speaker2):
    client = OpenAI()
    prompt = f'''
    Given the previous conversations {conversation} as background information, can you answer the question: {question}?
    Your answer should also recall the previous conversation between {speaker1} and {speaker2}.
    The response should be in one sentence with no more than 15 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def rag_sumy_prompt(summary, question, speaker1, speaker2):
    client = OpenAI()
    prompt = f'''
    Given the related summary of {speaker1} and {speaker2}'s conversation: {summary} as background information, can you answer the question: {question}?
    Your answer should be aligned with the personal traits of {speaker1} and {speaker2}.
    Your answer should also recall the previous conversation between {speaker1} and {speaker2}.
    The response should be in one sentence with no more than 15 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def rag_original_prompt(original, question, speaker1, speaker2):
    client = OpenAI()
    prompt = f'''
    Given the related conversation of {speaker1} and {speaker2}: {original} as background information, can you answer the question: {question}?
    Your answer should be aligned with the personal traits of {speaker1} and {speaker2}.
    Your answer should also recall the previous conversation between {speaker1} and {speaker2}.
    The response should be in one sentence with no more than 15 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response


def sel_fact_prompt(question, speaker_fact, speaker_name):
    client = OpenAI()
    prompt = f'''
    Given {speaker_name} has the following personal traits: {speaker_fact}. Can you answer the question: {question} based on the useful traits?
    Your answer should be aligned with the personal traits of {speaker_name}.
    The response should be in one sentence with no more than 15 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response
   

def main(
        home_dir = './datasets',
        dataset_name = 'locomo',
        model_type = 'gpt-4o-mini',
        summy_info = 'summarized_facts.json',
        output_response_file = 'full_response.json',
        output_retrieve_file = 'retrieve_text.json'
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

    for index, uni_fact in enumerate(sum_fact[len(final_response):]):        
        user_name = []
        for name in uni_fact[0]:
            if name != 'summary':
                user_name.append(name[:-5])
        
        speaker1_fact = []
        speaker2_fact = []
        for conv in uni_fact:
            speaker1 = conv[f'{user_name[0]}_fact']
            speaker2 = conv[f'{user_name[1]}_fact']
            speaker1_fact.extend(get_fact(speaker1))
            speaker2_fact.extend(get_fact(speaker2))

        speaker1_fact = get_distinct_facts(speaker1_fact, 0.9, model)
        speaker2_fact = get_distinct_facts(speaker2_fact, 0.9, model)

        print(f'{user_name[0]}_fact {speaker1_fact}')
        print(f'{user_name[1]}_fact {speaker2_fact}')

        summarized_facts1, topic_dic1 = get_clustered_facts(speaker1_fact, model, 15, model_type)
        summarized_facts2, topic_dic2 = get_clustered_facts(speaker2_fact, model, 15, model_type)

        summarization_list = []
        for item in uni_fact:
            summarization_list.append(item['summary'])

        prev_conversation = ''
        for conv in conversations[index]:
            prev_conversation += conv

        response_texts = []
        retrevial_texts = []
        for qa_pair in questions[index]:
            for key in qa_pair:
                if 'answer' in key:
                    all_response_dic = {key: qa_pair[key]}
            all_response_dic['question'] = qa_pair['question']
            all_retrevial_dic = {'question': qa_pair['question']}
            flag = 0
            if user_name[0] in qa_pair['question']:
                sel_facts = select_facts(qa_pair['question'], summarized_facts1, topic_dic1, model, model_type)
                select_name = user_name[0]
            elif user_name[1] in qa_pair['question']:
                sel_facts = select_facts(qa_pair['question'], summarized_facts2, topic_dic2, model, model_type)
                select_name = user_name[1]
            else:
                sel_facts1 = select_facts(qa_pair['question'], summarized_facts1, topic_dic1, model, model_type)
                sel_facts2 = select_facts(qa_pair['question'], summarized_facts2, topic_dic2, model, model_type)
                flag = 1

            if flag == 0:
                sel_fact_response_text = sel_fact_prompt(qa_pair['question'], sel_facts, select_name)
                all_retrevial_dic['select_fact'] = sel_facts
            else:
                sel_fact_response_text = fact_prompt(qa_pair['question'], user_name[0], sel_facts1, user_name[1], sel_facts2)
                all_retrevial_dic['select_fact'] = sel_facts1 + sel_facts2
            all_response_dic['select_fact'] = sel_fact_response_text

            print(f'sel_fact_response_text {sel_fact_response_text}')

            all_fact_response_text = fact_prompt(qa_pair['question'], user_name[0], speaker1_fact, user_name[1], speaker2_fact)
            all_retrevial_dic['all_fact'] = speaker1_fact + speaker2_fact
            all_response_dic['all_fact'] = all_fact_response_text
            print(f'all_fact_response_text {all_fact_response_text}')

            long_content_response_text = long_context_prompt(prev_conversation, qa_pair['question'], user_name[0], user_name[1])
            all_retrevial_dic['long_content'] = prev_conversation
            all_response_dic['long_content'] = long_content_response_text
            print(f'long_content_response_text {long_content_response_text}')

            sel_sum_list, sel_org_list = rag_original(qa_pair['question'], summarization_list, conversations[index], model, top_k=5)
            rag_sumy_response_text = rag_sumy_prompt(sel_sum_list, qa_pair['question'], user_name[0], user_name[1])
            all_retrevial_dic['rag_sumy'] = sel_sum_list
            all_response_dic['rag_sumy'] = rag_sumy_response_text
            print(f'rag_sumy_response_text {rag_sumy_response_text}')

            rag_org_response_text = rag_original_prompt(sel_org_list, qa_pair['question'], user_name[0], user_name[1])
            all_retrevial_dic['rag_org'] = sel_org_list
            all_response_dic['rag_org'] = rag_org_response_text
            print(f'rag_org_response_text {rag_org_response_text}')

            retrevial_texts.append(all_retrevial_dic)
            response_texts.append(all_response_dic)

        final_response.append(response_texts)
        retrieve_text.append(retrevial_texts)

        with open(f'{home_dir}/{dataset_name}/{output_response_file}', 'w') as f:
            json.dump(final_response, f, indent=4)
        
        with open(f'{home_dir}/{dataset_name}/{output_retrieve_file}', 'w') as f:
            json.dump(retrieve_text, f, indent=4)
    




if __name__ == "__main__":
    fire.Fire(main)