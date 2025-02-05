import json
import os
import openai
from openai import OpenAI
from openai import AsyncOpenAI
import asyncio
import time
from tqdm import tqdm
from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)  # pip install nltk

from rouge_score import rouge_scorer

os.environ['OPENAI_API_KEY'] = 'API-KEY'
os.environ['SAMBANOVA_API_KEY'] = 'API-KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def sum_fact_main(scenario, model_type):
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

def openai_async_inference(messages, tqdm_description="Calling OpenAI API", model_name=None, parse_json=True, batch_size=20):
    """get response with async openai api"""

    if 'gpt' in model_name:
        client = AsyncOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        )
    else:
        client = AsyncOpenAI(
            api_key=os.environ.get('TOGETHER_API_KEY'),
            base_url=os.environ.get('TOGETHER_BASE_URL'),
        )
    
    async def get_response(msg, index):
        completion = await client.chat.completions.create(
            model=model_name,
            messages=msg,
        )
        return completion.choices[0].message.content, index
    
    async def get_all_responses(msgs):
        # Create a list of TaskWrapper objects
        tasks = [get_response(msg, i) for i, msg in enumerate(msgs)]

        results = [None] * len(msgs) # Initialize results list with the same length as input
        with tqdm(total=len(tasks), desc=tqdm_description, position=1, leave=False) as pbar:
            for future in asyncio.as_completed(tasks):
                result = await future
                results[result[1]] = result[0] # index the result with the original index
                pbar.update(1)

        return results
    
    if batch_size == -1:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        # Run the async function within the existing event loop
        completions = loop.run_until_complete(get_all_responses(messages))
        time.sleep(1)
    else:
        # rate limit for together api
        completions = []
        for i in range(0, len(messages), batch_size):
            # Get the current event loop
            loop = asyncio.get_event_loop()
            # Run the async function within the existing event loop
            completions.extend(loop.run_until_complete(get_all_responses(messages[i:i+batch_size])))
            # time.sleep(5)

    final_outputs = []
    for completion in completions:
        final_outputs.append(completion.strip())

    return final_outputs


def sum_fact_all(scenario, model_type):
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


def sum_fact_all_batch(scenarios, model_name):
    
    prompt = ''' Given {scenario}.
    Can you first summarize the conversation to only contain the main information with the format <<Summary:>>,
    and then find all the key personalization facts for both {speaker1} and {speaker2} from the raw conversation. 
    Please only output facts without any other reasons or further explanation with the format:
    <<{speaker1}:>> (1). (2). (3)....
    <<{speaker2}:>> (1). (2). (3)....
    '''
    prompts = []
    for scenario in scenarios:
        speaker1 = scenario.split('\n')[0].split(':')[0]
        speaker2 = scenario.split('\n')[1].split(':')[0]
        prompts.append(prompt.format(scenario=scenario, speaker1=speaker1, speaker2=speaker2))
        
    messages = []
    for inp in prompts:
        messages.append([
            {"role": "system", "content": "You are an expert in generate high level personal traits."},
            {"role": "user", "content": inp }
        ])
    return openai_async_inference(messages,
                                  model_name = model_name,
                                  tqdm_description="Calling OpenAI API",
                                  batch_size=-1)


def sum_fact_reasoning(scenario, model_type):
    
    if model_type == 'Meta-Llama-3.1-405B-Instruct':
        client = openai.OpenAI(
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )
    else:
        client = OpenAI()

    prompt = f''' Given {scenario}.
    Can you first summarize the conversation to only contain the main information with the format <<Summary:>>,
    and then find the summarized key personalization facts for Speaker1 that already happened and only include long-term effects. 
    Please only output facts without any other reasons or further explanation with the format:
    <<Speaker1:>> (1). (2). (3)....
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

def sum_fact_reasoning_batch(scenarios, model_name):
    
    prompt = ''' Given {scenario}.
    Can you first summarize the conversation to only contain the main information with the format <<Summary:>>,
    and then find the summarized key personalization facts for Speaker1 that already happened and only include long-term effects. 
    Please only output facts without any other reasons or further explanation with the format:
    <<Speaker1:>> (1). (2). (3)....
    '''
    prompts = [prompt.format(scenario=scenario) for scenario in scenarios]
    messages = []
    for inp in prompts:
        messages.append([
            {"role": "system", "content": "You are an expert in generate high level personal traits."},
            {"role": "user", "content": inp }
        ])
    return openai_async_inference(messages,
                                  model_name = model_name,
                                  tqdm_description="Calling OpenAI API",
                                  batch_size=-1)


def select_reasoning_question_batch(facts, question, model_name):
    
    prompt = '''
    Given the persona traits: {fact}.
    Can the above traits directly affect the answer of question: {question}?
    Please output only 'YES' or 'NO'.
    '''
    prompts = [prompt.format(fact=fact, question=question) for fact in facts]
    messages = []
    for inp in prompts:
        messages.append([
            {"role": "system", "content": "You are an expert in generate high level personal traits."},
            {"role": "user", "content": inp }
        ])
    return openai_async_inference(messages,
                                  model_name = model_name,
                                  tqdm_description="Calling OpenAI API",
                                  batch_size=-1)


def implic_reason_batch(facts, question, model_name):
    prompt = '''
    Given the information: {fact}.
    Could that implys the question that {question}?
    Then answer shoud only be <<YES>> or <<NO>> without anything else.
    '''
    prompts = [prompt.format(fact=fact, question=question) for fact in facts]
    messages = []
    for inp in prompts:
        messages.append([
            {"role": "system", "content": "You are an expert in answering questions."},
            {"role": "user", "content": inp }
        ])
    return openai_async_inference(messages,
                                  model_name = model_name,
                                  tqdm_description="Calling OpenAI API",
                                  batch_size=-1)


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

def process_sessions(file_path):
    sessions = []
    current_session = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.isdigit():  # Check if the line is an index number
                if current_session:
                    sessions.append("\n".join(current_session))  # Join turns with newlines
                    current_session = []  # Reset for the next session
            else:
                # Replace User: and Agent: with Speaker: and Assistant:
                formatted_line = line.replace("User:", "Speaker1:").replace("Agent:", "Assistant:")
                current_session.append(formatted_line)  # Add the formatted line to the session

    # Add the last session to the list if it exists
    if current_session:
        sessions.append("\n".join(current_session))

    return sessions


def process_redial(home_dir):
    name_list = ['Books', 'Electronics', 'Movie', 'Sports']
    session_list = []
    for name in name_list:
        file_path = f'{home_dir}/LLM_Redial/{name}/Conversation.txt'
        session_list.extend(process_sessions(file_path))
    return session_list