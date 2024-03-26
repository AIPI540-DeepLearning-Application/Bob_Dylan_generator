import pandas as pd
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
import openai
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity

import json


def get_data():
    """
    Function to get the data from the csv file

    Returns:
        data (str): The raw text from the PDF file
    """

    # Read the PDF file
    df = pd.read_csv("../data/Bob_Dylan.csv")
    
    formatted_series = df.apply(lambda x: f"{x['title']}: {x['lyrics']}", axis=1)

    data = '\n'.join(formatted_series)
    
    return data

def get_chunks(text):
    """
    Function to get the chunks of text from the raw text

    Args:
        text (str): The raw text from the PDF file

    Returns:
        chunks (list): The list of chunks of text
    """

    # Initialize the text splitter
    splitter = CharacterTextSplitter(
        separator="\n", # Split the text by new line
        chunk_size=1250, # Split the text into chunks of 1600 characters
        chunk_overlap=200, # Overlap the chunks by 200 characters
        length_function=len # Use the length function to get the length of the text
    )

    # Get the chunks of text
    chunks = splitter.split_text(text)

    return chunks

def get_embeddings(chunk_data, client):
    """
    Get the embedding vectors for the chunk data

    Arg:
    - chunk_data: a list of chunk data

    Return:
    - Embedded vectors

    """
    
    response = client.embeddings.create(
        input=chunk_data,
        model="text-embedding-3-small"
        )

    vectors_list = [item.embedding for item in response.data]
    return vectors_list


# Store vectors in vector database
def vector_store(vectors_list):
    # Iterate over the vectors_list
    for i in range(len(vectors_list)):
        index.upsert(
            vectors=[
                {
                    'id': f'vec_{i}',
                    'values': vectors_list[i],
                    'metadata': {"text":chunk_data[i]}
                }
            ],
        )

def retrieve_embedding(index, num_embed):
    """
    Convert the information of vectors in the database into a panda dataframe
    
    Args:
    - index: Name of vector database(already set up)
    - num_embed: total number of vectors in the vector databse

    Return:
    - a dataframe which contains the embedded vectors and corresponding text
    """
    # Initialize a dictionary to store embedding data
    embedding_data = {"id":[], "values":[], "text":[]}
    
    # Fetch the embeddings 
    embedding = index.fetch([f'vec_{i}' for i in range(num_embed)])
    
    for i in range(num_embed):
        embedding_data["id"].append(i)
        idx = f"vec_{i}"
        embedding_data["text"].append(embedding['vectors'][idx]['metadata']['text'])
        embedding_data["values"].append(embedding['vectors'][idx]['values'])
        
    return pd.DataFrame(embedding_data)

def semantic_search(query_vector, db_embeddings):
    """
    Find the top three vectors which have the highest comsine similarity with the query vector

    Args:
    - query_vector: embedded vector of user query
    - db_embeddings: embedded vectors from vector database

    Return:
    - The indices of top three most similar vectors with the query vector
    """
    
    similarities = cosine_similarity(query_vector, db_embeddings)[0]
    # Get the indices of the top three similarity scores
    top_10_indices = np.argsort(similarities)[-10:][::-1]  # This sorts and then reverses to get top 3
    # Retrieve the top three most similar chunks and their similarity scores
    
    return top_10_indices

def get_text(embedding_data, top_10_indices):
    """
    Extracts text corresponding to the given top vectors from embedding data.

    Args:
    - embedding_data (DataFrame): DataFrame containing columns 'id', 'values', and 'text'.
    - top_vectors (list): List of indices for which corresponding text needs to be extracted.

    Returns:
    - combined_text (str): Combined text corresponding to the top vectors.
    """
   # Extract text from selected rows
    selected_texts = embedding_data.loc[top_10_indices, 'text'].tolist()

    # Combine the selected texts into a single string
    combined_text = ' '.join(selected_texts)

    return combined_text

def get_query_vectors(poems_title):
    
    queries = []
    
    for title in poems_title:
        query = f"create a poem for me in the style of Bob Dylan on the topic: {title}"
        queries.append(query)
    
    query_vectors = []

    # Loop through each query in the list
    for query in queries:
        # Generate an embedding for the current query
        embedding = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        # Append the generated embedding to the query_embeddings list
        query_vector = [item.embedding for item in embedding.data]
        query_vectors.append(query_vector)
    
    return query_vectors

def get_top_10_indices_list(query_vectors):
    
    top_10_indices_list = []
    
    for query_vector in query_vectors:
        top_10_indices = semantic_search(query_vector, vectors_list)
        top_10_indices_list.append(top_10_indices)
    return top_10_indices_list

def get_context(top_10_indices_list):
    
    contexts = []
    
    for indices in top_10_indices_list:
        # Retrieve text for the current set of top indices
        context = get_text(embedding_data, indices)
        contexts.append(context)
    
    return contexts

def get_response(poems_title, contexts):
    
    rag_response_list = []

    for title,context in zip(poems_title, contexts):
        
        system_prompt = """
            you are a Bob Dylan poetry generator bot. 
            Please generate a poem in Bob Dylan's style. The topic is: 
            """
            
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": title},
            {"role": "assistant", "content": context},

        ]
        )
        
        rag_response_list.append({title + ": ": completion.choices[0].message.content})
    
    return rag_response_list


if __name__ == "__main__":
        
    client = openai.OpenAI(api_key = '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')

    data = get_data()
    
    chunk_data = get_chunks(data)
    
    vectors_list = get_embeddings(chunk_data, client)
    
    pc = Pinecone(api_key='7658c7bf-1ee9-4e8d-9a5e-bc100843b51f')
    
    index = pc.Index("openai")
    
    vector_store(vectors_list)
    
    embedding_data = retrieve_embedding(index,len(vectors_list))
    
    poems_title = [
        "Shadows in the Moonlight",
        "Echoes of Forgotten Dreams",
        "Tides of Change",
        "Beneath the Velvet Sky",
        "Dancing with the Stars",
        "In the Heart of the Storm",
        "Serenade of the Sea",
        "Flames of Desire",
        "Through the Eyes of Time",
        "Whispers Among the Ruins",
        "Glimpses of Eternity",
        "Rhythms of the Rain",
        "Veils of Mist",
        "Embers of a Fading Sun",
        "Bridges Over Silent Waters",
        "Silent Whispers of the Night",
        "Melodies in the Twilight",
        "Footprints in the Sand",
        "Whispers of the Wind",
        "Mysteries of the Cosmos",
        "Songs of the Soul",
        "Harmony of the Universe",
        "Whispers of the Heart",
        "Stardust Serenade",
        "Journey to the Unknown",
        "Silent Echoes of Love",
        "Eternal Flames of Passion",
        "Chasing the Horizon",
        "Whispers of the Enchanted Forest",
        "Sailing through Stardust",
        "Moonlit Reflections",
        "Whispers of the Winter's Eve",
        "Whispers of the Autumn Leaves",
        "Whispers of the Spring Breeze",
        "Whispers of the Summer Rain",
        "Whispers of the Frozen Lake",
        "Whispers of the Mountain Peaks",
        "Whispers of the Flowing River",
        "Whispers of the Ocean Waves",
        "Whispers of the Desert Sands",
        "Whispers of the Starry Night",
        "Whispers of the Midnight Sky",
        "Whispers of the Distant Galaxies",
        "Whispers of the Nebulae",
        "Whispers of the Celestial Symphony",
        "Whispers of the Luminous Moon",
        "Whispers of the Shimmering Stars",
        "Whispers of the Twilight Sky",
        "Whispers of the Enchanted Garden",
        "Whispers of the Secret Garden"
        ]
    
    query_vectors = get_query_vectors(poems_title)
    
    top_10_indices_list = get_top_10_indices_list(query_vectors)  
    
    context = get_context(top_10_indices_list)
    
    rag_response_list = get_response(poems_title, context)

    with open('../output/poems_rag_gpt35.json', 'w') as f:
        json.dump(rag_response_list, f)
        
    
    

    
    
    
    

