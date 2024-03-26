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
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# from langchain_core.prompts import PromptTemplate
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bob Dylan style lyrics")
st.title("Bob Dylan style lyrics🤖️")


@st.cache_resource
def init_model():
    path = "/home/ec2-user/Bob_Dylan_generator/web/llama-7b-bobdylan-local"

    if os.path.exists(path):
        print("llama-7b-bobdylan-local exist")
        model_id = "Pot-l/llama-7b-bobdylan"
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                        token = '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')

        model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)        
    else:
        print("llama-7b-bobdylan-local does not exist")
        
        # Load PEFT model on CPU
        model_id = "Pot-l/llama-7b-bobdylan"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token = "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")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token = "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"
        )
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained("llama-7b-bobdylan-local",safe_serialization=True, max_shard_size="2GB")
        model = AutoModelForCausalLM.from_pretrained(
            "./llama-7b-bobdylan-local",
            low_cpu_mem_usage=True
        )
    return model, tokenizer


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
    llm, tokenizer = init_model()
    messages = init_chat_history()

    if user_input := st.chat_input("Shift + Enter for switching a new line, Enter for sending"):
        with st.chat_message("user", avatar='🧑'):
            st.markdown(user_input)
        response = generate_resp(user_input, llm, tokenizer)  
        print(response)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown(response)

        st.button("Clear Chat", on_click=clear_chat_history)
    
    
if __name__ == "__main__":
    main()