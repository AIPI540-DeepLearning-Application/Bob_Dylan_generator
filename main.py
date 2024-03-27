import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.generation.utils import GenerationConfig
from peft import PeftModel, PeftConfig
from peft import AutoPeftModelForCausalLM
import os
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
import warnings
import transformers
import sys


@st.cache_resource
def init_model(model_hf, simple_title):
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_hf,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    system_prompt = """You are a Bob Dylan poetry generator. Please generate a poem in Bob Dylan's style. Please only give me the poem content and do not give other or incomplete information and blanks. Do not provide any '\n' in the end. The topic is: """
    sequences = pipeline(
        system_prompt + simple_title,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=512,
    )
    response_res = sequences[0]['generated_text'][len(system_prompt):].strip()
    print(f"Result:\n{response_res}\n")
    torch.cuda.empty_cache()
    return response_res


# clear history messages
def clear_chat_history():
    del st.session_state.messages

# initialize the chat history
def init_chat_history():
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("Hi, I am an assistant for generating the lyrics with the Bob Dylan's style")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🙋‍♂️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages

def generate_resp(user_input, model, tokenizer):
    # build pipeline
    system_prompt = """you are a Bob Dylan poetry generator bot. Please generate a poem in Bob Dylan's style. The topic is: """
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    sample_input =  [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]

    prompt = pipe.tokenizer.apply_chat_template(
        sample_input,
        tokenize=False,
        add_generation_prompt=True
        )

    outputs = pipe(prompt,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.1,
                top_k=50,
                top_p=0.1,
                eos_token_id=pipe.tokenizer.eos_token_id,
                pad_token_id=pipe.tokenizer.pad_token_id
                )
    return outputs[0]['generated_text'][len(prompt):].strip()


def main():
    st.set_page_config(page_title="Bob Dylan style lyrics")
    st.title("Bob Dylan style lyrics🤖️")
    messages = init_chat_history()

    if user_input := st.chat_input("Shift + Enter for switching a new line, Enter for sending"):
        with st.chat_message("user", avatar='🧑'):
            st.markdown(user_input)
        response = init_model(model_hf='meta-llama/Llama-2-7b-chat-hf', simple_title=user_input)
        print(response)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown(response)

        st.button("Clear Chat", on_click=clear_chat_history)
    
    
if __name__ == "__main__":
    main()