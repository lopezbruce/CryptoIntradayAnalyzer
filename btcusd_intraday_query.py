# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for accessing the OpenAI API
import numpy as np # for array operations
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from dotenv import load_dotenv  # for loading environment variables
from scipy.spatial.distance import cdist  # for calculating vector similarities for search

load_dotenv()  # Load environment variables from .env file
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Initialize the OpenAI client

# models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
GPT_MODEL = os.getenv("GPT_MODEL","gpt-3.5-turbo")
EMBEDDINGS_SAVE_PATH = os.getenv("EMBEDDINGS_SAVE_PATH","data/processed/BTCUSD_Intraday_2023-02-28_2023-02-19.csv")
MAX_TOKENS = int(os.getenv("MAX_TOKENS",4096))


# an example question about intraday close for BTCUSD on 2023-02-19
# query = 'What is the intraday close for BTCUSD on 2023-02-19 00:00:00?'

# response = client.chat.completions.create(
#     messages=[
#         {'role': 'system', 'content': 'You answer questions about the BTCUSD Intraday.'},
#         {'role': 'user', 'content': query},
#     ],
#     model=GPT_MODEL,
#     temperature=0,
# )

# print(response.choices[0].message.content)

# Load the embeddings and text data
embeddings_path = EMBEDDINGS_SAVE_PATH

# Load the embeddings and text data
df = pd.read_csv(embeddings_path)

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# token counting function
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - cdist([x], y, 'cosine')[0],
    top_n: int = 1000
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = np.array(query_embedding_response.data[0].embedding)
    embeddings = np.vstack(df['embedding'].values)
    relatednesses = relatedness_fn(query_embedding, embeddings)
    sorted_indices = np.argsort(-relatednesses)
    top_indices = sorted_indices[:top_n]
    top_strings = df['text'].iloc[top_indices].tolist()
    return top_strings

# query message function
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings = strings_ranked_by_relatedness(query, df)  # get only strings as relatednesses is not used
    introduction = 'Use the below JSON on the Intraday BTCUSD From 2023-02-28 to 2023-02-19 to answer the subsequent question. If the answer cannot be found in the JSON, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_obejct = f'\n{string}\n"""'
        if (
            num_tokens(message + next_obejct + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_obejct
    return message + question

# ask function
def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = MAX_TOKENS - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Intraday BTCUSD From 2023-02-28 to 2023-02-19."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

# test the ask function
print(ask("What is the intraday final close price for BTCUSD on 2023-02-19?", print_message=False))
print(ask("What date was the highest open.", print_message=False))
print(ask("Analyse the data and provide me a summary.", print_message=False))