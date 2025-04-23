import json
import os
import fire
from openai import OpenAI
from sklearn.mixture import GaussianMixture
from data_process import dataset_process
from utils import evaluate_bleu, calculate_rouge_l
from sentence_transformers import SentenceTransformer
from utils import openai_async_inference

os.environ['OPENAI_API_KEY'] = 'API-KEY'

def get_fact(text):
    fact_list = []
    print(f'summarized {text}')
    if text.find('(1). ') != -1:
        facts = text.split('(1). ')[1].split('). ')
        for fact in facts:
            fact_list.append(fact.split(' (')[0])
    elif text.find('(1) ') != -1:
        facts = text.split('(1) ')[1].split(') ')
        for fact in facts:
            fact_list.append(fact.split(' (')[0])
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

def summarized_topic(text, model_type = 'gpt-4o-mini'):
    client = OpenAI()

    prompt = f'''
        Can you summarize {text} in one sentence to only contain the high-level information? 
        Please only output the summary without anything else.
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

def select_fact_batch(question, facts, model_type = 'gpt-4o-mini'):
    prompt = '''
    Given the personal information: {fact}, do you think it may possibly related to the {question}?
    Please output only 'YES' or 'NO'.
    '''
    summaries = [cur_fact for cur_fact in facts]
    prompts = [prompt.format(fact=fact, question=question) for fact in summaries]
    messages = []
    for inp in prompts:
        messages.append([
            {"role": "system", "content": "You are an expert in answering questions."},
            {"role": "user", "content": inp }
        ])
    checked_list = openai_async_inference(messages,
                                model_name = model_type,
                                tqdm_description="Calling OpenAI API",
                                batch_size=-1) 
    
    sel_summary = []
    sel_facts = []
    for i, response in enumerate(checked_list):
        if response.lower().find('yes') != -1:
            sel_summary.append(summaries[i])
            sel_facts.extend(facts[summaries[i]])
    return sel_summary, sel_facts

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
    for key, value in key_dic.items():
        summary = summarized_topic(value, model_type)
        summarized_facts[summary] = value

    return summarized_facts

def rag_original(question, summarization_list, original_list, model, top_k=2):
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

def fact_prompt(conversation, question, speaker1_fact, speaker2_fact, model_type = 'gpt-4o-mini'):
    client = OpenAI()
    prompt = f'''
    Speaker1 has the following personal traits: {speaker1_fact}.
    Speaker2 has the following personal traits: {speaker2_fact}.
    Given their previous conversations {conversation} as background information, can you generate a response answer for speaker2 when speaker1 says {question}?
    Your answer should be aligned with the personal traits of speaker1 and speaker2.
    Your answer should also recall the previous conversation between speaker1 and speaker2.
    The response should be in one sentence with no more than 30 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model= model_type,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def long_context_prompt(conversation, question, model_type = 'gpt-4o-mini'):
    client = OpenAI()
    prompt = f'''
    Given their previous conversations {conversation} as background information, can you generate a response answer for speaker2 when speaker1 says {question}?
    Your answer should also recall the previous conversation between speaker1 and speaker2.
    The response should be in one sentence with no more than 30 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model= model_type,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def rag_sumy_prompt(conversation, summary, question, model_type = 'gpt-4o-mini'):
    client = OpenAI()
    prompt = f'''
    There is the related summary of speaker1 and speaker2's conversation: {summary}.
    Given their previous conversations {conversation} as background information, can you generate a response answer for speaker2 when speaker1 says {question}?
    Your answer should be aligned with the personal traits of speaker1 and speaker2.
    Your answer should also recall the previous conversation between speaker1 and speaker2.
    The response should be in one sentence with no more than 30 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model= model_type,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def rag_original_prompt(conversation, original, question, model_type = 'gpt-4o-mini'):
    client = OpenAI()
    prompt = f'''
    There is the related conversation of speaker1 and speaker2: {original}.
    Given their previous conversations {conversation} as background information, can you generate a response answer for speaker2 when speaker1 says {question}?
    Your answer should be aligned with the personal traits of speaker1 and speaker2.
    Your answer should also recall the previous conversation between speaker1 and speaker2.
    The response should be in one sentence with no more than 30 words.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model= model_type,
        messages=[
            {"role": "system", "content": "You are an expert in answering personal questions."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response
   

def main(
        home_dir = './datasets',
        dataset_name = 'CC',
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
            conversations = conversations[len(final_response):]
            questions = questions[len(final_response):]
            sum_fact = sum_fact[len(final_response):]
    else:
        final_response = []

    if os.path.exists(f'{home_dir}/{dataset_name}/{output_retrieve_file}'):
        with open(f'{home_dir}/{dataset_name}/{output_retrieve_file}', 'r') as f:
            retrieve_text = json.load(f)
    else:
        retrieve_text = []
    
    print(f'conversation length: {len(conversations)}, question length: {len(questions)}, sum_fact length: {len(sum_fact)}')
        
    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    for index, uni_fact in enumerate(sum_fact[:310]):
        speaker1_fact = []
        speaker2_fact = []
        for conv in uni_fact[:4]:
            speaker1 = conv['SPEAKER_1_fact']
            speaker2 = conv['SPEAKER_2_fact']
            speaker1_fact.extend(get_fact(speaker1))
            speaker2_fact.extend(get_fact(speaker2))

        speaker1_fact = get_distinct_facts(speaker1_fact, 0.9, model)
        speaker2_fact = get_distinct_facts(speaker2_fact, 0.9, model)

        print(f'speaker1_fact {speaker1_fact}')
        print(f'speaker2_fact {speaker2_fact}')

        summarized_facts1 = get_clustered_facts(speaker1_fact, model, 5, model_type)
        summarized_facts2 = get_clustered_facts(speaker2_fact, model, 5, model_type)

        summarization_list = []
        for item in uni_fact[:4]:
            summarization_list.append(item['summary'])

        prev_conversation = ''
        for conv in conversations[index][:4]:
            prev_conversation += conv

        cur_turn = ''
        long_content_turn = prev_conversation
        response_texts = []
        retrevial_texts = []
        for qa_pair in questions[index]:
            all_response_dic = {'ground_truth': qa_pair['answer']}
            all_response_dic['question'] = qa_pair['question']
            all_retrevial_dic = {'question': qa_pair['question']}
            all_retrevial_dic['ground_truth'] = qa_pair['answer']

            select_summary1, select_facts1 = select_fact_batch(qa_pair['question'], summarized_facts1, model_type)
            select_summary2, select_facts2 = select_fact_batch(qa_pair['question'], summarized_facts2, model_type)

            print(f'select_summary1 {select_summary1}')
            print(f'select_facts1 {select_facts1}')
            print(f'select_summary2 {select_summary2}')
            print(f'select_facts2 {select_facts2}')

            sel_summary_response_text = fact_prompt(cur_turn, qa_pair['question'], select_summary1, select_summary2, model_type)
            all_response_dic['select_summary'] = sel_summary_response_text
            all_retrevial_dic['select_summary'] = select_summary1 + select_summary2
            print(f'sel_summary_response_text {sel_summary_response_text}')

            sel_fact_response_text = fact_prompt(cur_turn, qa_pair['question'], select_facts1, select_facts2, model_type)
            all_response_dic['select_fact'] = sel_fact_response_text
            all_retrevial_dic['select_fact'] = select_facts1 + select_facts2
            print(f'sel_fact_response_text {sel_fact_response_text}')

            all_fact_response_text = fact_prompt(cur_turn, qa_pair['question'], speaker1_fact, speaker2_fact, model_type)
            all_response_dic['all_fact'] = all_fact_response_text
            all_retrevial_dic['all_fact'] = speaker1_fact + speaker2_fact
            print(f'all_fact_response_text {all_fact_response_text}')

            long_content_response_text = long_context_prompt(long_content_turn, qa_pair['question'], model_type)
            all_response_dic['long_content'] = long_content_response_text
            all_retrevial_dic['long_content'] = long_content_turn
            print(f'long_content_response_text {long_content_response_text}')

            sel_sum_list, sel_org_list = rag_original(qa_pair['question'], summarization_list, conversations[index][:4], model, top_k=2)
            rag_sumy_response_text = rag_sumy_prompt(cur_turn, sel_sum_list, qa_pair['question'], model_type)
            all_response_dic['rag_sumy'] = rag_sumy_response_text
            all_retrevial_dic['rag_sumy'] = sel_sum_list
            print(f'rag_sumy_response_text {rag_sumy_response_text}')

            rag_org_response_text = rag_original_prompt(cur_turn, sel_org_list, qa_pair['question'], model_type)
            all_response_dic['rag_org'] = rag_org_response_text
            all_retrevial_dic['rag_org'] = sel_org_list
            print(f'rag_org_response_text {rag_org_response_text}')

            cur_turn += 'SPEAKER_1: ' + qa_pair['question'] + '\n' + 'SPEAKER_2: ' + qa_pair['answer'] + '\n'
            long_content_turn += 'SPEAKER_1: ' + qa_pair['question'] + '\n' + 'SPEAKER_2: ' + qa_pair['answer'] + '\n'

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