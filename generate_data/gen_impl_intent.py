import json
import os
import fire
import numpy as np
import tqdm
import random
import openai
import json
from sentence_transformers import SentenceTransformer
os.environ['SAMBANOVA_API_KEY'] = 'API_KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def implic_reason(per_info, traits_info):

    prompt_1 = f'''
        {per_info} Can you give me at least 15 implicit reason information that supports this claim? Therefore, if I ask you, "Does {per_info}?", you have to answer "yes". 
        The reason information should be completely different from each other and belong to different categories.
        The reason should be specific with detailed information, like why it happens.
        The reason cannot include words related to "{traits_info}"
        Please explain the reasoning in only one sentence. Please only output the reasons with the format:
        1: 
        2: 
    '''

    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    response = client.chat.completions.create(
        model='Meta-Llama-3.1-405B-Instruct',
        messages=[
                    {"role":"system","content":"You are a helpful assistant"},
                    {"role":"user","content": prompt_1}
                ],
        temperature =  1,
        top_p = 0.9
    )
    reason_result = response.choices[0].message.content
    print(reason_result)

    lines = reason_result.splitlines()

    # Initialize an empty list to store the reasons
    reasons = []

    # Loop through each line
    for line in lines:
        # Split the line at the colon and check if it has a reason part
        if ":" in line:
            reason = line.split(":")[1].strip()  # Extract the part after colon and remove any leading/trailing spaces
            reasons.append(reason)
    top_reason = reasons[1:]

    return top_reason

def select_unique_reasons(similarity_matrix, threshold):
    # Number of reasons
    num_reasons = similarity_matrix.shape[0]
    
    # To keep track of selected reasons
    selected_reasons = []
    
    # Create a list of reasons (0 to num_reasons-1)
    remaining_reasons = list(range(num_reasons))
    
    # Loop through all reasons
    while len(remaining_reasons) >= 2:
        # Find the two reasons with the lowest similarity score
        min_score = float('inf')
        best_pair = (None, None)
        
        for i in range(len(remaining_reasons)):
            for j in range(i + 1, len(remaining_reasons)):
                reason1 = remaining_reasons[i]
                reason2 = remaining_reasons[j]
                if similarity_matrix[reason1, reason2] < min_score:
                    min_score = similarity_matrix[reason1, reason2]
                    best_pair = (reason1, reason2)
        
        # Add the best pair to the selected reasons
        selected_reasons.extend(best_pair)

        
        # Remove the reasons that are similar to either of the two selected reasons
        new_remaining_reasons = []
        for reason in remaining_reasons:
            if similarity_matrix[best_pair[0], reason] < threshold and similarity_matrix[best_pair[1], reason] < threshold:
                new_remaining_reasons.append(reason)
        
        remaining_reasons = new_remaining_reasons

    if len(remaining_reasons) == 1:
        selected_reasons.append(remaining_reasons[0])

    print(selected_reasons)
    
    return selected_reasons

def select_reason(per_info, implicit_reason):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    str_reason = ''
    for i, reason in enumerate(implicit_reason):
        str_reason += f'{i+1}. {reason}\n'

    prompt = f'''
    {per_info}. Here are potential implicit reason information that can support this claim: {str_reason}. 
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

def gen_question(per_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    prompt = f'''
    Given {per_info} Can you change it into a question? The answer should be "yes" or 'no'.
    For example. The original sentence is "This person loves sports." The question should be "Does this person love sports?"
    Please only output the question in one line:
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

def noisy_scenario(persona, question, traits_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    prompt = f'''
    Given question: {question}
    Can you give me 5 implicit scenarios for this person that supports this question? Therefore, if I ask you, "Does {persona}?", you have to answer "yes". 
    The reason information should be completely different from each other and belong to different categories.
    The reason cannot include words related to "{traits_info}".
    The scenarios should contain only one sentence.
    Please output the scenarios only with the index number using the format below.
    1. ...
    2. ...
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
    result = str(completion.choices[0].message.content)
    final_result = []
    for a in result.split('. ')[1:]:
        final_result.append(a.split('\n')[0])

    return final_result

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


def main(
        home_dir = './datasets/impl_reasoning',
        input_file = 'hobbies.json',
        output_file = 'final_syn_intent_conv3.jsonl'
    ):
    with open(f'{home_dir}/{input_file}', 'r') as file:
        #data = json.load(file)
        data = json.load(file)[25020:30000]

    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    for line in data:
        persona = line
        traits = ' '.join(line.split()[3:])
        print(f'persona: {persona}')


        while True:
            try:
                impli_rea = implic_reason(persona, traits)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass
        print(f'implicit reasons: {impli_rea}')
        embeddings = model.encode(impli_rea)
        similarities = model.similarity(embeddings, embeddings)
        #print(similarities)
        selected_indices = select_unique_reasons(similarities, 0.55)
        selected_reasons = [impli_rea[i] for i in selected_indices]

        while True:
            try:
                best_reason = select_reason(persona, selected_reasons)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass

        print(f'best reason: {best_reason}')
        while True:
            try:
                question = gen_question(persona)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass
        
        print(f'question: {question}')
        while True:
            try:
                reason_conv = conversation(best_reason)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass

        print(f'conversation reasoning: {reason_conv}')

        while True:
            try:
                noisy_scenarios = noisy_scenario(persona, question, traits)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass
        print(f"Noisy scenarios: {noisy_scenarios}")

        noisy_convs = []
        for scen in noisy_scenarios:
            while True:
                try:
                    noisy_conv = conversation(scen)
                    break
                except Exception as e:
                    if e == KeyboardInterrupt:
                        raise e
                    else:
                        pass
            print(f"Noisy conversation: {noisy_conv}")
            noisy_convs.append(noisy_conv)
        
        result = {
            'persona' : persona,
            'reason': best_reason,
            'inquiry': question,
            "noisy_scenarios": noisy_scenarios,
            "syn_reasoning_conv": reason_conv,
            "noisy_conv": noisy_convs,
            }
        
        with open(f'{home_dir}/{output_file}', "a") as outfile:
            json.dump(result, outfile)
            outfile.write("\n")



if __name__ == "__main__":
    fire.Fire(main)