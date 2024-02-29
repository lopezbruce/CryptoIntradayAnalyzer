import json # for JSON data handling
import os # for file path operations
import numpy as np # for array operations
import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
from dotenv import load_dotenv  # for loading environment variables

load_dotenv()  # Load environment variables from .env file
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Initialize the OpenAI client

# Define constants
GPT_MODEL = os.getenv("MODEL_NAME","gpt-3.5-turbo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","text-embedding-3-small") 
BATCH_SIZE = int(os.getenv("BATCH_SIZE",1000))  # you can submit up to 2048 embedding inputs per request
JSON_FILE_PATH = os.getenv("JSON_FILE_PATH","data/raw/BTCUSD_Intraday_2023-02-28_2023-02-19.json")
SAVE_PATH = os.getenv("EMBEDDINGS_SAVE_PATH","data/processed/BTCUSD_Intraday_2023-02-28_2023-02-19.csv")

# Define the embedding function
def embed_json(data):
    # Example embedding function, you can replace it with your own embedding logic
    embedded_data = []
    for entry in data:
        embedded_entry = [
            entry["open"],
            entry["low"],
            entry["high"],
            entry["close"],
            entry["volume"]
        ]
        embedded_data.append(embedded_entry)
    return np.array(embedded_data)

# Step 1: Collect JSON
file_path = JSON_FILE_PATH

# Step 2: Chunk JSON
chunk_size = 1000  # Define the chunk size
output_directory = "json_chunks"
os.makedirs(output_directory, exist_ok=True)

with open(file_path, "r") as json_file:
    data = json.load(json_file)
    total_lines = len(data)
    num_chunks = total_lines // chunk_size + 1

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_lines)
        chunk_data = data[start_idx:end_idx]

        # Step 3: Embed JSON Chunks
        embedded_chunk_data = embed_json(chunk_data)
        
        # Step 4: Store JSON Chunks and Embeddings
        output_file_path = os.path.join(output_directory, f"chunk_{i+1}.npz")
        np.savez_compressed(output_file_path, embedded_chunk_data=embedded_chunk_data)

print("JSON data chunking and embedding completed.")

# Step 5: Generate embeddings for the chunks
json_strings = [json.dumps(chunk) for chunk in data]  # Convert JSON chunks to strings
embeddings = []


for batch_start in range(0, len(json_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = json_strings[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input
    batch_embeddings = [e.embedding for e in response.data]
    embeddings.extend(batch_embeddings)

# Step 6: Create DataFrame and save to CSV
df = pd.DataFrame({"text": json_strings, "embedding": embeddings})
df.to_csv(SAVE_PATH, index=False)

print("Embeddings generation and saving completed.")
