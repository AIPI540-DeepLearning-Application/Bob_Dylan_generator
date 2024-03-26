from openai import OpenAI
import json

class Evaluation:
    def __init__(self, model_name = "gpt-4"):
        self.model_name = model_name
        self.client = OpenAI()

    def evaluate(self):
        results = []
        for i in range(10):
            response = self.client.chat.completions.create(
            model = self.model_name,
            # response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful poetry critic."},
                {"role": "user", "content": "Which of the following 5 poems is most close to Bob Dylan's style? Give me the index, like the first/second/third..."
                +list(self.poems_fine_tuned_gpt35[0])[0]+list(self.poems_fine_tuned_gpt35[0].values())[0]
                +list(self.poems_fine_tuned_llama2[0])[0]+list(self.poems_fine_tuned_llama2[0].values())[0]
                +list(self.poems_no_fine_tuned_gpt35[0])[0]+list(self.poems_no_fine_tuned_gpt35[0].values())[0]
                +list(self.poems_no_fine_tuned_llama2[0])[0]+list(self.poems_no_fine_tuned_llama2[0].values())[0]
                +list(self.poems_rag_gpt35[0])[0]+list(self.poems_rag_gpt35[0].values())[0]
                }
            ]
            )
            print(response.choices[0].message.content)
            results.append(response.choices[0].message.content)
        return results


    def load_test_data(self):
        
        with open('../output/poems_finetuned_gpt35.json', encoding='utf-8') as f:
            self.poems_fine_tuned_gpt35 = json.load(f)

        with open('../output/poems_finetuned_llama2.json', encoding='utf-8') as f:
            self.poems_fine_tuned_llama2 = json.load(f)

        with open('../output/poems_nofinetuned_gpt35.json', encoding='utf-8') as f:
            self.poems_no_fine_tuned_gpt35 = json.load(f)

        with open('../output/poems_nofinetuned_llama2.json', encoding='utf-8') as f:
            self.poems_no_fine_tuned_llama2 = json.load(f) 

        with open('../output/poems_rag_gpt35.json', encoding='utf-8') as f:
            self.poems_rag_gpt35 = json.load(f)
    