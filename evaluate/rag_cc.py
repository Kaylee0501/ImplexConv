import json
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)  # pip install nltk
from rouge import Rouge  # pip install rouge

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


def format_document(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def split_conversation_chronicles(dataset):
    documents = []
    
    for episode in dataset:
        # Episode-level document
        episode_doc = Document(
            page_content=f"{episode['first_session_dialogue']} {episode['second_session_dialogue']} {episode['third_session_dialogue']} {episode['fourth_session_dialogue']}",
            metadata={
                "type": "episode",
                "id": episode["dataID"],
                "relationship": episode["relationship"],
                "time_intervals": episode["time_interval"]
            }
        )
        documents.append(episode_doc)
        
        # Session-level documents
        for i, session in enumerate(["first", "second", "third", "fourth"], 1):
            session_doc = Document(
                page_content=episode[f"{session}_session_dialogue"][0],
                metadata={
                    "type": "session",
                    "id": f"{episode['dataID']}_session_{i}",
                    "relationship": episode["relationship"],
                    "time_interval": episode["time_interval"][i-1] if i > 1 else None
                }
            )
            documents.append(session_doc)
        
        # Summary document
        summary_doc = Document(
            page_content=episode["summary"][0],
            metadata={
                "type": "summary",
                "id": f"{episode['dataID']}_summary",
                "relationship": episode["relationship"]
            }
        )
        documents.append(summary_doc)
    
    return documents


from langchain_community.chat_models import ChatSambaNovaCloud
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import time

chat = ChatSambaNovaCloud(
    sambanova_url = "https://api.sambanova.ai/v1/chat/completions",
    sambanova_api_key = "56349ebd-647f-4b94-9c89-4358d1a708f5",
    model = "Meta-Llama-3.1-405B-Instruct",
    max_tokens = 1024,
)


def get_qa_pair(index, ds):
    # get the fifth session
    session_key_map = {
        0: "first_session_dialogue",
        1: "second_session_dialogue",
        2: "third_session_dialogue",
        3: "fourth_session_dialogue",
        4: "fifth_session_dialogue",
    }
    question_and_answers = ds[index].get(session_key_map[4], [])
    questions = [
        question_and_answers[i] for i in range(0, len(question_and_answers), 2)
    ]
    answers = [question_and_answers[i] for i in range(1, len(question_and_answers), 2)]
    return questions, answers

def ragWrapper():
    cc = load_dataset("jihyoung/ConversationChronicles")
    cc_3000 = cc["train"].select(range(100))


    # Split into documents
    documents = split_conversation_chronicles(cc_3000)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()


    for index, document in enumerate(documents):

        q,a = get_qa_pair(index-1, cc_3000)
        # Create FAISS index
        db = FAISS.from_documents([document], embeddings)
        
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # texts = text_splitter.split_documents(documents)

        # query="For each of the conversations answer the given question. "
        # result=db.similarity_search(query)
        # # print(result[0].page_content)

        llm=chat

        # prompt=ChatPromptTemplate.from_template(
        #     """
        # Answer the following question based only on the provided context. I will tip you $1000 if the user finds the answer helpful.
        # <context>
        # {context}
        # </context>
        # Question: {input}
        # """
        # )


        retriever = db.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": retriever | format_document, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        while True:
            try:
                answer = rag_chain.invoke(q[-1])
                break
            except Exception as e:
                if "429" in str(e):
                    continue
                else:
                    raise e
        bleu1, bleu2, bleu3 = evaluate_bleu(a[-1], answer)
        rouge = evaluate_rouge(a[-1], answer)
        # print(index, bleu1, bleu2, bleu3, rouge)
        
        # Store the scores
        if index == 0:
            avg_bleu1, avg_bleu2, avg_bleu3, avg_rouge = bleu1, bleu2, bleu3, rouge
        else:
            avg_bleu1 = (avg_bleu1 * index + bleu1) / (index + 1)
            avg_bleu2 = (avg_bleu2 * index + bleu2) / (index + 1)
            avg_bleu3 = (avg_bleu3 * index + bleu3) / (index + 1)
            avg_rouge = (avg_rouge * index + rouge) / (index + 1)
        print("current result: ", index, avg_bleu1, avg_bleu2, avg_bleu3, avg_rouge)
    return avg_bleu1, avg_bleu2, avg_bleu3, avg_rouge

avg_bleu1, avg_bleu2, avg_bleu3, avg_rouge = ragWrapper()
print("final result: ", avg_bleu1, avg_bleu2, avg_bleu3, avg_rouge)