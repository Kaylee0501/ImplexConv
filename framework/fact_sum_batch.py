import json
import time
import os
import fire
#from openai import OpenAI
import openai
from tqdm import tqdm
from data_process import dataset_process
from utils import sum_fact_all_batch, sum_fact_reasoning, sum_fact_reasoning_batch

os.environ['OPENAI_API_KEY'] = 'API_KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(
        home_dir = './datasets',
        data_dir = 'impConv',
        dataset_name = 'syn_reasoning',
        model_type = 'gpt-4o-mini',
        output_file = 'summarized_reasoning_facts0.json',
    ):
    # if file exist,  load it
    conversations, qa_pairs = dataset_process(home_dir, dataset_name)
    if os.path.exists(f'{home_dir}/{data_dir}/{output_file}'):
        with open(f'{home_dir}/{data_dir}/{output_file}', 'r') as f:
            conversation_list = json.load(f)
        conversations = conversations[len(conversation_list):]        
    else:
        conversation_list = []
    print(f'conversation length: {len(conversation_list)}')

    for i, conversation in tqdm(enumerate(conversations[:500]), desc='Processing conversations'):
        dialog_list = []
        batch_size = 60
        for j in range(0, len(conversation), batch_size):
            batch = conversation[j:j+batch_size] # list of string dialogs
            batch_responses = [None for _ in range(len(batch))]
            batch_finished = [False for _ in range(len(batch))]
            while True:
                print('batch_finished:', batch_finished)
                unfinished_batch_indices = [i for i, finished in enumerate(batch_finished) if not finished]
                unfinished_batch = [batch[i] for i in unfinished_batch_indices]               
                responses = sum_fact_reasoning_batch(unfinished_batch, model_type)
                for i, response in enumerate(responses):
                    if "<<Summary:>>" in response and \
                    "<<Speaker1:>>" in response:
                        batch_responses[unfinished_batch_indices[i]] = response
                        batch_finished[unfinished_batch_indices[i]] = True
                if all(batch_finished):
                    break
            for response in batch_responses:
                dialog_dic = {}
                dialog_dic['summary'] = response.split('<<Summary:>>')[1].split(f'<<Speaker1:>>')[0].strip()
                dialog_dic['Speaker1_fact'] = response.split(f'<<Speaker1:>>')[1].strip()
                dialog_list.append(dialog_dic)
    
        conversation_list.append(dialog_list)
    
        with open(f'{home_dir}/{data_dir}/{output_file}', 'w') as f:
            json.dump(conversation_list, f, indent=4)





if __name__ == "__main__":
    fire.Fire(main)