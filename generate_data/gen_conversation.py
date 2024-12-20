import json
import time
import os
import fire
#from openai import OpenAI
import openai
os.environ['SAMBANOVA_API_KEY'] = 'API_KEY'

def user_inq(per_info, reason_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    
    prompt = f'''
    Here's the conversation between a user(speaker 1) and a chatbot assistant.
    Speaker 1 has the following persona trait: {per_info}. However, speaker 1 cannot do the trait due to the reason that {reason_info}. 
    Now, speaker 1 ask you a question related to the trait. {reason_info} affect your answer to this question.
    You should tell speark 1 they cannot do the trait due to the reason.
    The trait should be mentioned in the question.
    The question itself should not mention the reason or affect by the reason.
    Questions should be asked in the first person. Include "I".
    The question should not be a yes/no question. 
    The question needs to be diverse.

    Please only output the question in the format of less than 20 words without any additional sentences.

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



def noisy_scenario(persona, question, traits_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    prompt = f'''
    Consider a person with specific personality traits {persona} that could serve as responses to a given question {question}. 
    Can you generate additional scenarios that reflect or align with these personality traits to support the question?
    Please output 5 scenarios that are relevant to the given traits and question.
    The scenarios should contain only one sentence.
    The scenarios can talk about both {traits_info} or other stuff that is related to {traits_info} but do not have to be the same.
    Please output the scenarios only with the index number.

    For example:

    Trait: I love sports
    Question: I'm bored; can you give me some suggestions?
    Scenarios:
    1. I love playing basketball.
    2. My favorite basketball player is Stephen Curry.
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



def main(
        home_dir = './datasets/impl_reasoning',
        input_file = 'sel_implic_full2.json',
        output_file = 'final_syn_conversation2.jsonl'
    ):
    with open(f'{home_dir}/{input_file}', 'r') as file:
        data = json.load(file)

    for line in data[2071:]:
        persona = line['persona']
        impli_reas = line['best_reason']
        traits = ' '.join(persona.split()[3:])

        print(f'persona: {persona}')
        print(f'reasons: {impli_reas}')

        while True:
            try:
                inquiry = user_inq(persona, impli_reas)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass
        print(f'inquiry: {inquiry}')

        while True:
            try:
                noisy_scenarios = noisy_scenario(persona, inquiry, traits)
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
            'reason': impli_reas,
            "question": inquiry,
            "noisy_scenarios": noisy_scenarios,
            "syn_trait_conv": line['conversation_trait'],
            "syn_reasoning_conv": line['conversation_reasoning'],
            "noisy_conv": noisy_convs,
        }

        with open(f'{home_dir}/{output_file}', "a") as outfile:
            json.dump(result, outfile)
            outfile.write("\n")



if __name__ == "__main__":
    fire.Fire(main)