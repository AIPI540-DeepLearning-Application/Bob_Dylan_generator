import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import PeftModel, PeftConfig
from peft import AutoPeftModelForCausalLM

st.set_page_config(page_title="Bob Dylan style lyrics")
st.title("Bob Dylan style lyrics🤖️")


@st.cache_resource
def init_model():
    model_path = "Pot-l/llama-7b-bobdylan"
    AutoModelForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, token = "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")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")

    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


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
    model, tokenizer = init_model()
    messages = init_chat_history()
    if prompt := st.chat_input("Shift + Enter change line, Enter send"):
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("clear chat", on_click=clear_chat_history)


if __name__ == "__main__":
    main()