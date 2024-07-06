from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from fastchat.conversation import get_conv_template
import torch
import openai
import os
from retry import retry

class TargetBase():
    def __init__(self):
        pass

    def generate(self, prompt):
        raise NotImplementedError("generate method must be implemented in subclass.")
    
    def batch_generate(self, prompts):
        raise NotImplementedError("batch_generate method must be implemented in subclass.")



class TargetVicuna(TargetBase):
    def __init__(self, model_path, device_map):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.model_name = 'vicuna_v1.1' # v1.1 for convernience to use get_conv_template() in FastChat, actually it is v1.5

    def get_prompt(self, prompt):
        conv = get_conv_template(self.model_name)
        if type(prompt) != str:
            print("input prompt is not string, input type: ", type(prompt))
            prompt = str(prompt)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def generate(self, prompt, task, do_sample=False, max_new_tokens=100, temperature=0.0, top_p=1.0, num_return_sequences=1):
        with torch.no_grad():
            prompt = self.get_prompt(prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs['input_ids'].shape[1]
            if task == 'alpaca_eval':
                outputs = self.model.generate(**inputs, max_length=3072, do_sample=True, temperature=0.7, top_p=1.0)
                print(task)
            else:
                if task == "ReNeLLM":
                    max_new_tokens = 200
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                              temperature=temperature, top_p=top_p, 
                                              num_return_sequences=num_return_sequences)
                # print(task)
        if num_return_sequences > 1:
            return self.tokenizer.batch_decode(outputs[:,input_length:], skip_special_tokens=True)
        else:
            return self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    
    def batch_generate(self, prompts, task, do_sample=False, max_new_tokens=100, temperature=0.0, top_p=1.0):
        if task == 'alpaca_eval':
            with torch.no_grad():
                prompts = [self.get_prompt(prompt) for prompt in prompts]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
                input_length = inputs['input_ids'].shape[1]
                outputs = self.model.generate(**inputs, max_length=3072, do_sample=True, temperature=0.7, top_p=1.0)
                print(task)
        else:
            with torch.no_grad():
                prompts = [self.get_prompt(prompt) for prompt in prompts]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
                input_length = inputs['input_ids'].shape[1]
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)
                # print(task)

        return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    

class TargetLlama2(TargetBase):
    def __init__(self, model_path, device_map):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.model_name = 'Llama-2-7b-chat'
        self.sys = "You are a helpful, respectful and honest assistant."

    def get_prompt(self, prompt, task):
        if type(prompt) != str:
            print("input prompt is not a string, input type: ", type(prompt))
            prompt = str(prompt)
        if task == 'alpaca_eval':
            use_sys = self.sys
        else:
            use_sys = ""
        chat = [
            {
                "role": "system", "content": use_sys
            },
            {
                "role": "user", "content": prompt
            }
        ]
        if task == "GCG" or task == "AutoDAN":
            return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).rstrip(" [/INST]") + "[/INST]"
        else:
            return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    def generate(self, prompt, task, do_sample=False, max_new_tokens=100, temperature=0.0, top_p=1.0, num_return_sequences=1):
        with torch.no_grad():
            prompt = self.get_prompt(prompt, task)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs['input_ids'].shape[1]
            if task == 'alpaca_eval':
                outputs = self.model.generate(**inputs, max_length=3072, do_sample=True, temperature=0.7, top_p=1.0)
                print(task)
            else:
                if task == "ReNeLLM":
                    max_new_tokens = 200
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                              temperature=temperature, top_p=top_p, 
                                              num_return_sequences=num_return_sequences)
        if num_return_sequences > 1:
            return self.tokenizer.batch_decode(outputs[:,input_length:], skip_special_tokens=True)
        else:
            return self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    
    def batch_generate(self, prompts, task, do_sample=False, max_new_tokens=100, temperature=0.0, top_p=1.0):
        if task == 'alpaca_eval':
            with torch.no_grad():
                prompts = [self.get_prompt(prompt, task) for prompt in prompts]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
                input_length = inputs['input_ids'].shape[1]
                outputs = self.model.generate(**inputs, max_length=3072, do_sample=True, temperature=0.7, top_p=1.0)
                print(task)
        else:
            with torch.no_grad():
                prompts = [self.get_prompt(prompt, task) for prompt in prompts]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
                input_length = inputs['input_ids'].shape[1]
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)
                # print(task)

        return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    

class TargetChatGPT(TargetBase):
    def __init__(self):
        self.model_name = "gpt-3.5-turbo-0125"
        assert os.getenv("OPENAI_API_KEY") is not None, "Please set the OPENAI_API_KEY environment variable."
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.openai_api_key)

    def create_conv(self, prompt):
        """Create a FastChat conversation template for GPT."""
        if type(prompt) != str:
            print("input prompt is not string, input type: ", type(prompt))
            prompt = str(prompt)
        conv_template_name = "chatgpt"

        conv = get_conv_template(conv_template_name)
        conv.offset = 0
        conv.messages = []

        conv.append_message(conv.roles[0], prompt)

        return conv.to_openai_api_messages()
    
    @retry(tries=3, delay=2, backoff=2)
    def generate(self, prompt, task, max_new_tokens=100, do_sample=False, temperature=0.0, top_p=1.0):
        conv = self.create_conv(prompt)
        if task == "alpaca_eval":
            max_new_tokens = 3072
            temperature = 0.7
        if task == "SemanticSmoothParaphrase": 
            other_arg =  {"response_format" : { "type": "json_object" }}
        else:
            other_arg =  {}
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=conv,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=42,
                **other_arg
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print("Error in generating response: ", e)
        return response, completion.usage.total_tokens
    
    def batch_generate(self, prompts, task):
        responses = []
        total_token_usage = 0
        for prompt in prompts:
            response, token_usage = self.generate(prompt, task)
            responses.append(response)
            total_token_usage += token_usage
        return responses, total_token_usage