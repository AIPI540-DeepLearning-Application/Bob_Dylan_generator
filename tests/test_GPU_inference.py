from transformers import AutoTokenizer
import transformers
import torch
import json
import sys


def main(model_hf, simple_title):
    response_list = []
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
    response_list.append({simple_title + ": ": response_res})


    torch.cuda.empty_cache()
    tmp_path = '/home/ec2-user/Bob_Dylan_generator/output/' + model_hf.split('/')[1] + '.json'
    print(tmp_path)

    with open(tmp_path, 'a') as f:
        json.dump(response_list, f)
if __name__ == '__main__':
    # main(model_hf = "Pot-l/llama-7b-bobdylan")
    # main(model_hf = "meta-llama/Llama-2-7b-chat-hf")
    # model_hf="$1"
    # title="$2"
    model_hf = sys.argv[1]
    title = sys.argv[2]
    print(model_hf)
    print(title)
    main(model_hf, title)