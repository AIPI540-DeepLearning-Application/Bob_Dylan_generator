import torch

from huggingface_hub import login
import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, pipeline

import json


def model_loading():
    '''
    This function load our fine-tuned model and tokenizer
    This process may require over 30GB GPU memory, we run this on Colab A 100
    '''
    model_id = "Pot-l/llama-7b-bobdylan"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        )
    
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("llama-7b-bobdylan-local",safe_serialization=True, max_shard_size="2GB")
    
    model = AutoModelForCausalLM.from_pretrained(
        "/home/ec2-user/Bob_Dylan_generator/web/llama-7b-bobdylan-local",
        low_cpu_mem_usage=True
        )
    return model, tokenizer

def generate_response(model, tokenizer):
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
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
    
    system_prompt = """
        you are a Bob Dylan poetry generator bot.
        Please generate a poem in Bob Dylan's style. The topic is:
        """
        
    response_list = []

    for title in poems_title:
        sample_input =  [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": title}
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

        generated_lyrics = outputs[0]['generated_text'][len(prompt):].strip()
        print(f"Generated Lyrics:\n{generated_lyrics}")

        response_list.append({title + ": ": generated_lyrics})
    
    return response_list



if __name__ == "__main__":

    os.environ["HF_KEY"] = "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"

    login(token=os.environ.get('HF_KEY'), add_to_git_credential=True)
    
    model, tokenizer = model_loading()
    response = generate_response(model, tokenizer)
    
    with open('../output/poems_nofinetuned_llama2.json', 'w') as f:
        json.dump(response, f)
    
    