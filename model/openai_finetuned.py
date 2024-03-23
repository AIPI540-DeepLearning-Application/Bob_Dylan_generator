from dotenv import load_dotenv

import PyPDF2
import sqlite3

import numpy as np
import pandas as pd
import pickle

from openai import OpenAI
import os

import streamlit as st

import json


def data_converter(row):
    '''
    This funciton converts the data into the format that OpenAI API requires
    
    '''
    system_prompt = """
    you are a Bob Dylan poetry generator bot. 
    Please generate a poem in Bob Dylan's style. The topic is: 
    """
    user_prompt = row['title']
    lyrics = row['lyrics']

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": lyrics}
        ]
    }

def generate_response(client):
    '''
    This function generates responses from the fine-tuned bob dylan model
    
    '''
    poems_title = [
    "Whispers of the Wind",
    "Echoes of the Forgotten",
    "Shadows at Noon",
    "Rivers of Time",
    "Beneath the Harvest Moon",
    "Silent Symphony",
    "Dancing in the Rain",
    "Embers of Yesterday",
    "Paths Untrodden",
    "Midnight's Serenade",
    ]

    response_list = []

    for title in poems_title:
        
        system_prompt = """
            you are a Bob Dylan poetry generator bot. 
            Please generate a poem in Bob Dylan's style. The topic is: 
            """
            
        completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:duke-university::95j7JIbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": title},
        ]
        )
        
        response_list.append({title + ": ": completion.choices[0].message.content})
    

    with open('../output/poems_finetuned_gpt35.json', 'w') as f:
        json.dump(response_list, f)
        
    

if __name__ == '__main__':
    
    load_dotenv(override=True)
    
    client = OpenAI()

    data_path = '../data/Bob_Dylan.csv'
    
    data = pd.read_csv(data_path)
    data = data.dropna(subset=['lyrics'])

    data_json = data.apply(data_converter, axis=1)
    
    # save data_json to jsonl file
    output_dir = '../data/Bob_Dylan.jsonl'
    data_json.to_json(output_dir, orient='records', lines=True)
    
    # create dataset file on OpenAI, uncomment to run for the first time
    # client.files.create(
    #   file=open(output_dir, "rb"),
    #   purpose="fine-tune"
    # )
    
    # create fine-tuning job, substitute the training_file with the file id from the previous step
    client.fine_tuning.jobs.create(
        training_file='file-AxHOy7XTtcVuHZDkQZ28s12H', 
        model="gpt-3.5-turbo"
        )
    
    # generate response based on ten titles
    # the response will be saved in a json file at the directory: ../output/poems_finetuned_gpt35.json
    generate_response(client)
    
    
    