import torch;
# assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
from huggingface_hub import login
import os

from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForQuestionAnswering
from trl import setup_chat_format

from peft import LoraConfig

from transformers import TrainingArguments

from trl import SFTTrainer


def dataset_loading():
    train_dataset = load_dataset("json", data_files="../data/Bob_Dylan_train_dataset.json", split="train")
    test_dataset = load_dataset("json", data_files="../data/Bob_Dylan_test_dataset.json", split="train")
    
    return train_dataset, test_dataset

def load_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=os.environ.get('HF_KEY'),
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    model, tokenizer = setup_chat_format(model, tokenizer)
    
    return model, tokenizer


def build_trainer(model, tokenizer, train_dataset, test_dataset):
        
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    
    args = TrainingArguments(
        output_dir="llama-7b-bobdylan",         # saving dir name, on hugging face hub
        num_train_epochs=3,                     # num of epochs
        per_device_train_batch_size=2,          
        gradient_accumulation_steps=2,          
        gradient_checkpointing=True,           
        optim="adamw_torch_fused",              
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  
        learning_rate=2e-4,                     
        bf16=False,                              
        tf32=False,                              
        max_grad_norm=0.3,                      
        warmup_ratio=0.03,                      
        lr_scheduler_type="constant",           
        push_to_hub=True,                       # push the model to the hugging face hub
        report_to="tensorboard",                
    )
    
    max_seq_length = 3072 

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        }
    )
    return trainer

if __name__ == '__main__':
    
    # log into hugging face
    os.environ["HF_KEY"] = "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"

    login(token=os.environ.get('HF_KEY'), add_to_git_credential=True)
    
    print("torch.__version__: ", torch.__version__)
    print("torch.version.cuda: ", torch.version.cuda)
    
    train_dataset, test_dataset = dataset_loading()
    model, tokenizer = load_model()
    trainer = build_trainer(model, tokenizer, train_dataset, test_dataset)

    trainer.train()