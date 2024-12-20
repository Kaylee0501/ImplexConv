import json
import time
from datasets import load_dataset
import os
import fire
#from openai import OpenAI
import openai

##conversations: list of list of strings, each string is a dialog session
##qa_pairs: list of list of dictionaries, each dictionary has 'question' and 'answer' keys

def dataset_process(home_dir, dataset_name):
    if dataset_name == 'MSC':
        input_file = f'{home_dir}/msc/MSC/sequential_msc.json'

        with open(input_file, 'r') as f:
            data = json.load(f)
        
        ## conversations include the 1-5 sessions
        ## qa_pairs include the 5th session questions and answers
        conversations = []
        qa_pairs = []
        for items in  data:
            sec_dialog = []
            qa = []
            for i, session in enumerate(items):
                dialogs = session['dialog']
                context = ''
                for dialog in dialogs:
                    context += 'SPEAKER_1: ' + dialog['SPEAKER_1'] + '\n'
                    context += 'SPEAKER_2: ' + dialog['SPEAKER_2'] + '\n'
                    if i == 4:
                        pair_dict = {'question': dialog['SPEAKER_1'], 'answer': dialog['SPEAKER_2']}
                        qa.append(pair_dict)
                sec_dialog.append(context)
            conversations.append(sec_dialog)
            qa_pairs.append(qa)
        return conversations, qa_pairs

    elif dataset_name == 'CC':
        cc = load_dataset("jihyoung/ConversationChronicles")
        data = cc['test']
        ## conversations include the 1-5 sessions
        ## qa_pairs include the 5th session questions and answers
        conversations = []
        qa_pairs = []
        for items in data:
            dialogs = []
            dialog_1 = ''
            for i ,item in enumerate(items['first_session_dialogue']):
                if i % 2 == 0:
                    dialog_1 += 'SPEAKER_1: ' + item + '\n'
                else:
                    dialog_1 += 'SPEAKER_2: ' + item + '\n'
            dialogs.append(dialog_1)
            dialog_2 = ''
            for i ,item in enumerate(items['second_session_dialogue']):
                if i % 2 == 0:
                    dialog_2 += 'SPEAKER_1: ' + item + '\n'
                else:
                    dialog_2 += 'SPEAKER_2: ' + item + '\n'
            dialogs.append(dialog_2)
            dialog_3 = ''
            for i ,item in enumerate(items['third_session_dialogue']):
                if i % 2 == 0:
                    dialog_3 += 'SPEAKER_1: ' + item + '\n'
                else:
                    dialog_3 += 'SPEAKER_2: ' + item + '\n'
            dialogs.append(dialog_3)
            dialog_4 = ''
            for i ,item in enumerate(items['fourth_session_dialogue']):
                if i % 2 == 0:
                    dialog_4 += 'SPEAKER_1: ' + item + '\n'
                else:
                    dialog_4 += 'SPEAKER_2: ' + item + '\n'
            dialogs.append(dialog_4)
            dialog_5 = ''
            qa = []
            for i ,item in enumerate(items['fifth_session_dialogue']):
                if i % 2 == 0:
                    pair_dict = {}
                    dialog_5 += 'SPEAKER_1: ' + item + '\n'
                    pair_dict['question'] = item
                else:
                    dialog_5 += 'SPEAKER_2: ' + item + '\n'
                    pair_dict['answer'] = item
                    qa.append(pair_dict)
            dialogs.append(dialog_5)
            conversations.append(dialogs)
            qa_pairs.append(qa)
        return conversations, qa_pairs
    
    elif dataset_name == 'dailydialog':
        test_file = f'{home_dir}/dailydialog/test.jsonl'
        valid_file = f'{home_dir}/dailydialog/valid.jsonl'
        data = []
        with open(test_file, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        with open(valid_file, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        ## single session dialog
        ## qa_pairs include question and answer from the only session. Here we want to test the performance for a single question without history info
        conversations = []
        qa_pairs = []
        for items in data:
            dialog = ''
            qa = []
            for i, content in enumerate(items['dialogue']):
                if i % 2 == 0:
                    pair_dict = {}
                    dialog += 'SPEAKER_1: ' + content['text'] + '\n'
                    pair_dict['question'] = content['text']
                else:
                    dialog += 'SPEAKER_2: ' + content['text']  + '\n'
                    pair_dict['answer'] = content['text']
                    qa.append(pair_dict)
            conversations.append([dialog])
            qa_pairs.append(qa)
        return conversations, qa_pairs
    
    elif dataset_name == 'locomo':
        input_file = f'{home_dir}/locomo/locomo10.json'
        with open(input_file, 'r') as f:
            data = json.load(f)
        ## conversations include the 1-35 sessions
        ## qa_pairs from the locomo dataset 
        conversations = []
        qa_pairs = []
        for items in data:
            chat_dic = items['conversation']
            speaker_1 = chat_dic['speaker_a']
            speaker_2 = chat_dic['speaker_b']
            i  = 1
            dialog_session = []
            while True:
                if f'session_{i}' in chat_dic:
                    dialogs = ''
                    for text in chat_dic[f'session_{i}']:
                        if text['speaker'] == speaker_1:
                            dialogs += 'SPEAKER_1: ' + text['text'] + '\n'
                        else:
                            dialogs += 'SPEAKER_2: ' + text['text'] + '\n'
                    dialog_session.append(dialogs)
                    i += 1
                else:
                    break
            conversations.append(dialog_session)
            qa_pairs.append(items['qa'])
        return conversations, qa_pairs