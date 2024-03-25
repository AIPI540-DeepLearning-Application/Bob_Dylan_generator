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

        model = AutoModelForCausalLM.from_pretrained(path)        
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
    # build pipeline
    pipe = pipeline("text-generation",
            model=model,
            tokenizer= tokenizer,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            )

    llm = HuggingFacePipeline(pipeline = pipe)
    return llm


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



def main():
    llm = init_model()
    messages = init_chat_history()
    template = """
              You are a Bob Dylan poetry generator bot. Please generate a poem in Bob Dylan's style.
              Return your response in the format of a poetry.
              ```{text}```
              The poetry is:
           """

    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    if user_input := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(user_input)
        print(prompt)
        # print("type:" + str(type(user_input)) + " " + user_input)
        response = llm_chain.run(user_input)  
        print(len(response))
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            for res in response:
                placeholder.markdown(res)

        st.button("清空对话", on_click=clear_chat_history)
    
    
if __name__ == "__main__":
    main()
    
    # messages = init_chat_history()
    # if prompt := st.chat_input("Shift + Enter change line, Enter send"):
    #     with st.chat_message("user", avatar="🙋‍♂️"):
    #         st.markdown(prompt)
    #     messages.append({"role": "user", "content": prompt})
    #     print(f"[user] {prompt}", flush=True)
    #     with st.chat_message("assistant", avatar="🤖"):
    #         placeholder = st.empty()
    #         for response in model.chat(tokenizer, messages, stream=True):
    #             placeholder.markdown(response)
    #             if torch.backends.mps.is_available():
    #                 torch.mps.empty_cache()
    #     messages.append({"role": "assistant", "content": response})
    #     print(json.dumps(messages, ensure_ascii=False), flush=True)

        # st.button("clear chat", on_click=clear_chat_history)


# def main():
#     model, tokenizer = init_model()
#     messages = init_chat_history()
#     if prompt := st.chat_input("Shift + Enter change line, Enter send"):
#         with st.chat_message("user", avatar="🙋‍♂️"):
#             st.markdown(prompt)
#         messages.append({"role": "user", "content": prompt})
#         print(f"[user] {prompt}", flush=True)
        
#         with st.chat_message("assistant", avatar="🤖"):
#             placeholder = st.empty()
#             inputs = tokenizer(prompt, return_tensors="pt")
#             generate_ids = model.generate(inputs.input_ids, max_length=1000)
#             response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#             # for response in responses:
#             placeholder.markdown(response)
#             if torch.backends.mps.is_available():
#                 torch.mps.empty_cache()
#         messages.append({"role": "assistant", "content": response})
#         print(json.dumps(messages, ensure_ascii=False), flush=True)

#         st.button("clear chat", on_click=clear_chat_history)