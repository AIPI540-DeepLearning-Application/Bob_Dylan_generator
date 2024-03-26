from transformers import AutoTokenizer
import transformers
import torch

model_hf = "Pot-l/llama-7b-bobdylan"

tokenizer = AutoTokenizer.from_pretrained(model_hf)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_hf,
    torch_dtype=torch.float16,
    device_map="auto",
)

system_prompt = """You are a Bob Dylan poetry generator. Please generate a poem in Bob Dylan's style. Please only give me the poem content and do not give other or incomplete information and blanks. Do not provide any '\n' in the end. The topic is: """

simple_title = "A deep lake"

sequences = pipeline(
    system_prompt + simple_title,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=512,
)
for seq in sequences:
    print(f"Result:\n{seq['generated_text']}\n")

torch.cuda.empty_cache()