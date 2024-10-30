"""
This script loads a pre-trained SentenceTransformer model to identify conversations related to a target hobby ("This person enjoys swimming") in a large chat dataset. 
It computes and saves conversation embeddings, then ranks conversations by their similarity to the target hobby. 
Conversations with a high similarity score are saved to a CSV file (`sorted_conversations_with_scores.csv`) for easy access and analysis. 
The script is designed to streamline retrieval of hobby-related conversations, with caching to optimize future searches.
"""

from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import csv
import torch
import os
import pickle

# Load pre-trained model
model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)

# Define the hobby sentence and encode it
hobby = ["This person enjoys swimming."]
hobby_embedding = model.encode(hobby)
hobby_embedding = torch.tensor(hobby_embedding, device="cuda")

# File to store embeddings and conversations
embedding_file = "llm_sys_conversation_embeddings.pkl"

# Load dataset and extract multi-turn conversations
ds = load_dataset("lmsys/lmsys-chat-1m")
conversations = []
dataset_size = 0

# Load existing embeddings and conversations or compute and save them
if os.path.exists(embedding_file):
    # Load embeddings and conversations from the file if it exists
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
        conversations = data["conversations"]
        conversation_embeddings = data["conversation_embeddings"]
else:
    # Otherwise, process dataset to compute and store embeddings
    conversation_embeddings = []
    for i in tqdm(range(len(ds["train"]))):
        # Process only multi-turn conversations
        if int(ds["train"][i]["turn"]) > 1:
            # Collect text content of each turn
            conversation = [x["content"] for x in ds["train"][i]["conversation"]]
            conversations.append(conversation)

            # Encode each conversation and move embedding to GPU
            conv_embedding = model.encode(conversation)
            conv_embedding = torch.tensor(conv_embedding, device="cuda")
            conversation_embeddings.append(conv_embedding)
            dataset_size += 1
            if dataset_size == 1000:
                break
        if dataset_size % 100 == 0:
            print(dataset_size)

    # Save embeddings and conversations to a file
    with open(embedding_file, "wb") as f:
        pickle.dump(
            {
                "conversations": conversations,
                "conversation_embeddings": conversation_embeddings,
            },
            f,
        )

# Initialize list to store average similarity scores
avg_similarity_scores = []

# Compute similarities for each conversation
for conversation_emb, conversation in tqdm(zip(conversation_embeddings, conversations)):
    # Compute cosine similarity for each turn in the conversation with the hobby embedding
    similarities = (
        torch.nn.functional.cosine_similarity(hobby_embedding, conversation_emb)
        .max()
        .item()
    )

    # Append the max similarity and corresponding conversation
    avg_similarity_scores.append((similarities, conversation))

# Sort conversations by average similarity score in descending order
sorted_conversations = sorted(avg_similarity_scores, key=lambda x: x[0], reverse=True)

# Display the sorted conversations and their scores
for score, conversation in sorted_conversations:
    print(f"Average Similarity: {score:.4f}")
    print("Conversation:", conversation)
    print()

# Prepare data for saving
csv_data = []
for score, conversation in sorted_conversations:
    if score > 0.3:
        csv_data.append(
            {
                "Average Similarity Score": score,
                "Hobby": hobby[0],
                "Conversation": " ".join(conversation),
            }
        )

# Save to CSV
csv_file_path = "sorted_conversations_with_scores.csv"
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=["Average Similarity Score", "Hobby", "Conversation"]
    )
    writer.writeheader()
    writer.writerows(csv_data)
