from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)  # pip install nltk
from rouge import Rouge  # pip install rouge
from langchain_groq import ChatGroq  # pip install langchain-groq


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


def evaluate_rouge(reference, response):
    rouge = Rouge()
    scores = rouge.get_scores(response, reference)[0]
    rouge_score = scores["rouge-l"]["f"]

    return rouge_score


def evaluate_llm_judge(llm, question, response, reason):
    evaluation_prompt = f"""
    Question: {question}
    Response: {response}
    Reason: {reason}

    Evaluate the response based on the following criteria:
    1. Relevance to the reason.
    2. Quality and clarity of the advice.
    Provide a 0/1 for the response based on the reason. Just return 0 or 1 with reasoning
    """
    while True:
        try:
            impli_rea = llm.invoke(evaluation_prompt)
            break
        except Exception as e:
            if e == KeyboardInterrupt:
                raise e
            else:
                pass
    return impli_rea


def evaluate_prompt(llm, prompt, question, reason):
    evaluation_prompt = f"""
    Prompt: {prompt}
    Question: {question}
    Reason: {reason}

    Evaluate the response based on the following criteria:
    1. Does the prompt have the relevant reason
    2. Quality and clarity of the retrieval of the information.
    Provide a 0/1 for the response based on the reason. Just return 0 or 1 with reasoning
    """
    while True:
        try:
            impli_rea = llm.invoke(evaluation_prompt)
            break
        except Exception as e:
            if e == KeyboardInterrupt:
                raise e
            else:
                pass
    return impli_rea


if __name__ == "__main__":
    # Example usage
    reference = "The cat is on the mat"
    response = "The cat is sitting on the mat"

    bleu_1, bleu_2, bleu_3 = evaluate_bleu(reference, response)
    rouge_score = evaluate_rouge(reference, response)

    question = "What sports can I play to improve my overall fitness and have fun at the same time?"
    response = "You can play sports like basketball, soccer, or tennis to improve your fitness and have fun."
    reason = "I have ankle pain and need to avoid high-impact sports."


    # # Evaluate the response using the LLM
    llm = ChatGroq(
        temperature=0, model_name="llama-3.1-70b-versatile", groq_api_key=GROQ_API_KEY
    )
    llm_response = evaluate_llm_judge(llm, question, response, reason)
    llm_score = int(llm_response.content[0])

    retrieved_prompt = "Speaker 1: I am planning to visit Paris next month. Can you recommend some places to visit? Assistant: Sure! You should visit the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
    llm_response_prompt = evaluate_prompt(llm, retrieved_prompt, question, reason)
    llm_prompt_score = int(llm_response_prompt.content[0])
    print(llm_response_prompt)

    print("BLEU-1:", bleu_1)
    print("BLEU-2:", bleu_2)
    print("BLEU-3:", bleu_3)
    print("ROUGE:", rouge_score)
    print("LLM as a judge Score:", llm_score)
    print("LLM Prompt relevance Score:", llm_prompt_score)
