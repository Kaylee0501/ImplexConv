import json
import os
import fire
import numpy as np
import tqdm
import random
import openai
import json
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['SAMBANOVA_API_KEY'] = 'KEY'


def user_inq_1(per_info, reason_info, trait_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    
    prompt = f'''
    {per_info} However, they cannot do that activity recently because {reason_info}. Can you come up with a high-level question with an answer other than {trait_info}? 
    The question should be vague.
    Ask the question in the first person.
    The question should not mention that something is wrong.
    The question should not mention that they want something other than their usual activity.
    The question should not include words like "recently, other, new".
    Make sure all the words in the answer and reason do not exist in the question. 
    Please output the inquiry with the format in one line without any additional sentences: 
    User:
    '''
    completion = client.chat.completions.create(
    model='Meta-Llama-3.1-405B-Instruct',
    messages=[
                {"role":"system","content":"You are a helpful assistant"},
                {"role":"user","content": prompt}
              ],
    temperature =  1,
    top_p = 0.9
    )
    response = str(completion.choices[0].message.content)
    print(response)
    # Extract the output from the response if“User” eixits, delete the "User:" and any leading/trailing spaces
    if 'User:' in response:
        response = response.split('User:')[1].strip()
    
    return response

def conversation(scenario):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    prompt = f'''
    There are two speakers. Speaker 1 encounters the scenario that "{scenario}". Speaker 2 is the AI assistant.
    Based on the information. Can you generate a conversation with at least 10 turns?
    Speaker 1 shouldn't mention the scenario too early. It must be mentioned in the later section.
    Speaker 1 is exactly the person who encounters the scenario.
    The beginning turns should serve as a warm-up to introduce the scenario in a natural way.
    The conversation should be centered around the scenario without any irrelevant or extra information that is not related to the scenario.
    For Spearker 1, please do not start the conversation by saying something similar to "I'm feeling a bit overwhelmed lately." or use the same format as this sentence.
    Include diverse styles like detailed explanations, step-by-step guidance, casual small talk, humor, storytelling, and problem-solving. 
    The conversation should feel realistic and flow naturally. 
    Aim for a balance of formality and informality, capturing nuanced exchanges that go beyond simple responses.
    Please output the conversation in the following format:
    Speaker1: ...
    Assistant: ...
    
    Speaker1: ...
    Assistant: ...
    '''
    completion = client.chat.completions.create(
    model='Meta-Llama-3.1-405B-Instruct',
    messages=[
                {"role":"system","content":"You are a helpful assistant"},
                {"role":"user","content": prompt}
              ],
    temperature =  1,
    top_p = 0.9
    )
    return str(completion.choices[0].message.content)

def select_reason(per_info, implicit_reason):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    str_reason = ''
    for i, reason in enumerate(implicit_reason):
        str_reason += f'{i+1}. {reason}\n'

    prompt = f'''
    {per_info}. Here are potential implicit reasons why this person is unable to follow this trait: {str_reason}. 
    Could you select the reason that is both the most logically sound and subtly implied?
    Please select only from the provided options and output the reason only.
    '''
    completion = client.chat.completions.create(
    model='Meta-Llama-3.1-405B-Instruct',
    messages=[
                {"role":"system","content":"You are a helpful assistant"},
                {"role":"user","content": prompt}
              ],
    temperature =  1,
    top_p = 0.9
    )
    return str(completion.choices[0].message.content)

def select_question(model, reason, inquiries, query_prompt_name = "s2s_query"):
    query_embeddings = model.encode([reason], prompt_name=query_prompt_name)
    sentence_embeddings = model.encode(inquiries)

    similarities = model.similarity(query_embeddings, sentence_embeddings)
    min_index = similarities.argmin()  # Assuming similarities is a numpy array or tensor
    
    # Select the question with the lowest similarity
    selected_question = inquiries[min_index]
    
    return selected_question

def main(
        home_dir = './datasets/impl_reasoning',
        input_file = 'sel_implic_reason2.json',
        output_file = 'sel_implic_full2.jsonl'
    ):
    with open(f'{home_dir}/{input_file}', 'r') as file:
        data = json.load(file)

    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    for line in data:
        persona = line['persona']
        impli_reas = line['reason_extreme']
        traits = ' '.join(persona.split()[3:])

        print(f'persona: {persona}')
        print(f'reasons: {impli_reas}')
        print(f'traits: {traits}')

        while True:
            try:
                best_reason = select_reason(persona, impli_reas)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass

        inquiries = []
        for reason in impli_reas:
            while True:
                try:
                    inquiry = user_inq_1(persona, reason, traits)
                    break
                except Exception as e:
                    if e == KeyboardInterrupt:
                        raise e
                    else:
                        pass
            print(inquiry)
            inquiries.append(inquiry)

        best_question = select_question(model, best_reason, inquiries)

        while True:
            try:
                trait_conv = conversation(persona)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass
        
        while True:
            try:
                rea_conv = conversation(best_reason)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass
        
        result = {
            'persona' : persona,
            'reason_candidates': impli_reas,
            'best_reason': best_reason,
            'question': inquiries,
            'selected_question': best_question,
            'conversation_trait': trait_conv,
            'conversation_reasoning': rea_conv
        }

        with open(f'{home_dir}/{output_file}', "a") as outfile:
            json.dump(result, outfile, indent=4)
            outfile.write("\n")



if __name__ == "__main__":
    fire.Fire(main)