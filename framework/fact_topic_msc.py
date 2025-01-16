import json
import os
import fire
from openai import OpenAI
from sklearn.mixture import GaussianMixture
from data_process import dataset_process
from utils import evaluate_bleu, calculate_rouge_l
from sentence_transformers import SentenceTransformer

os.environ['OPENAI_API_KEY'] = 'API-KEY'
os.environ['SAMBANOVA_API_KEY'] = 'API-KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    Given question {question}, can you select all possible topics below that are related to the question?
    topics: {topic_list}
    Please only output the names of topics without changing any words. Each topic should be separated by a semicolon.
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

def fact_prompt(conversation, question, speaker1_fact, speaker2_fact):
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def long_context_prompt(conversation, question):
    client = OpenAI()
    prompt = f'''
    Given their previous conversations {conversation} as background information, can you generate a response answer for speaker2 when speaker1 says {question}?
    Your answer should also recall the previous conversation between speaker1 and speaker2.
    The response should be in one sentence with no more than 30 words.
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

def rag_sumy_prompt(conversation, summary, question):
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generate high level personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    return response

def rag_original_prompt(conversation, original, question):
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
        dataset_name = 'MSC',
        model_type = 'gpt-4o-mini',
        summy_info = 'summarized_facts.json',
        output_file = 'full_response.json',
    ):

    conversations, questions = dataset_process(home_dir, dataset_name)
    with open(f'{home_dir}/{dataset_name}/{summy_info}', 'r') as f:
        sum_fact = json.load(f)

    print(dataset_name)

    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    final_response = []
    for index, uni_fact in enumerate(sum_fact[:100]):
        speaker1_fact = []
        speaker2_fact = []
        for conv in uni_fact[:4]:
            speaker1 = conv['speaker1_fact']
            speaker2 = conv['speaker2_fact']
            speaker1_fact.extend(get_fact(speaker1))
            speaker2_fact.extend(get_fact(speaker2))

        speaker1_fact = get_distinct_facts(speaker1_fact, 0.9, model)
        speaker2_fact = get_distinct_facts(speaker2_fact, 0.9, model)

        print(f'speaker1_fact {speaker1_fact}')
        print(f'speaker2_fact {speaker2_fact}')

        summarized_facts1, topic_dic1 = get_clustered_facts(speaker1_fact, model, 5, model_type)
        summarized_facts2, topic_dic2 = get_clustered_facts(speaker2_fact, model, 5, model_type)

        summarization_list = []
        for item in uni_fact[:4]:
            summarization_list.append(item['summary'])

        prev_conversation = ''
        for conv in conversations[index][:4]:
            prev_conversation += conv

        cur_turn = ''
        long_content_turn = prev_conversation
        response_texts = []
        for qa_pair in questions[index]:
            all_response_dic = {'ground_truth': qa_pair['answer']}
            all_response_dic['question'] = qa_pair['question']
            select_facts1 = select_facts(qa_pair['question'], summarized_facts1, topic_dic1, model, model_type)
            select_facts2 = select_facts(qa_pair['question'], summarized_facts2, topic_dic2, model, model_type)

            sel_fact_response_text = fact_prompt(cur_turn, qa_pair['question'], select_facts1, select_facts2)
            all_response_dic['select_fact'] = sel_fact_response_text
            print(f'sel_fact_response_text {sel_fact_response_text}')

            all_fact_response_text = fact_prompt(cur_turn, qa_pair['question'], speaker1_fact, speaker2_fact)
            all_response_dic['all_fact'] = all_fact_response_text
            print(f'all_fact_response_text {all_fact_response_text}')

            long_content_response_text = long_context_prompt(long_content_turn, qa_pair['question'])
            all_response_dic['long_content'] = long_content_response_text
            print(f'long_content_response_text {long_content_response_text}')

            sel_sum_list, sel_org_list = rag_original(qa_pair['question'], summarization_list, conversations[index][:4], model, top_k=2)
            rag_sumy_response_text = rag_sumy_prompt(cur_turn, sel_sum_list, qa_pair['question'])
            all_response_dic['rag_sumy'] = rag_sumy_response_text
            print(f'rag_sumy_response_text {rag_sumy_response_text}')

            rag_org_response_text = rag_original_prompt(cur_turn, sel_org_list, qa_pair['question'])
            all_response_dic['rag_org'] = rag_org_response_text
            print(f'rag_org_response_text {rag_org_response_text}')

            cur_turn += 'SPEAKER_1: ' + qa_pair['question'] + '\n' + 'SPEAKER_2: ' + qa_pair['answer'] + '\n'
            long_content_turn += 'SPEAKER_1: ' + qa_pair['question'] + '\n' + 'SPEAKER_2: ' + qa_pair['answer'] + '\n'

            response_texts.append(all_response_dic)
        final_response.append(response_texts)

        with open(f'{home_dir}/{dataset_name}/{output_file}', 'w') as f:
            json.dump(final_response, f, indent=4)
    




if __name__ == "__main__":
    fire.Fire(main)