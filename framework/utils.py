import json
import os
import openai
from openai import OpenAI
from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)  # pip install nltk

from rouge_score import rouge_scorer

os.environ['OPENAI_API_KEY'] = 'API-KEY'
os.environ['SAMBANOVA_API_KEY'] = 'API-KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def sum_fact_old(scenario, model_type):
    speaker1 = scenario.split('\n')[0].split(':')[0]
    speaker2 = scenario.split('\n')[1].split(':')[0]
    
    if model_type == 'Meta-Llama-3.1-405B-Instruct':
        client = openai.OpenAI(
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )
    else:
        client = OpenAI()

    prompt = f''' Given {scenario}.
    Can you first summarize the conversation to only contain the main information with the format <<Summary:>>,
    and then find all the summarized key personalization facts for both {speaker1} and {speaker2} that already happened and only include long-term effects. 
    Please only output facts without any other reasons or further explanation with the format:
    <<{speaker1}:>> (1). (2). (3)....
    <<{speaker2}:>> (1). (2). (3)....
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
    return response, speaker1, speaker2

def sum_fact(scenario, model_type):
    speaker1 = scenario.split('\n')[0].split(':')[0]
    speaker2 = scenario.split('\n')[1].split(':')[0]
    
    if model_type == 'Meta-Llama-3.1-405B-Instruct':
        client = openai.OpenAI(
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )
    else:
        client = OpenAI()

    prompt = f''' Given {scenario}.
    Can you first summarize the conversation to only contain the main information with the format <<Summary:>>,
    and then find all the key personalization facts for both {speaker1} and {speaker2} from the raw conversation. 
    Please only output facts without any other reasons or further explanation with the format:
    <<{speaker1}:>> (1). (2). (3)....
    <<{speaker2}:>> (1). (2). (3)....
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
    return response, speaker1, speaker2

def evaluate_bleu(reference, response):
    # Convert reference and response to list of tokens
    reference_tokens = reference.split()
    response_tokens = response.split()

    # Calculate BLEU scores for n-grams from 1 to 3
    bleu_1 = sentence_bleu(
        [reference_tokens],
        response_tokens,
        weights=(1, 0, 0),
        smoothing_function=SmoothingFunction().method1,
    )
    bleu_2 = sentence_bleu(
        [reference_tokens],
        response_tokens,
        weights=(0.5, 0.5, 0),
        smoothing_function=SmoothingFunction().method1,
    )
    bleu_3 = sentence_bleu(
        [reference_tokens],
        response_tokens,
        weights=(0.33, 0.33, 0.33),
        smoothing_function=SmoothingFunction().method1,
    )

    return bleu_1, bleu_2, bleu_3


def calculate_rouge_l(candidate, reference):
    # Create a scorer for ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Compute ROUGE-L
    scores = scorer.score(reference, candidate)
    return scores['rougeL']