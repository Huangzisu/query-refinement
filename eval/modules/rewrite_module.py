from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from fastchat.conversation import get_conv_template
import json
import torch



class RewriterBase():
    def __init__(self):
        pass

    def get_prompt(self, prompt):
        raise NotImplementedError("get_prompt method must be implemented in subclass.")

    def rewrite(self, prompt):
        raise NotImplementedError("rewrite method must be implemented in subclass.")
    

class RewriterGemma():
    def __init__(self, model_config_path, model_weights_path, device_map):
        self.model_config = AutoConfig.from_pretrained(model_config_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_weights_path, config=self.model_config, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config_path)
        self.tokenizer.padding_side = 'right'
        self.model_name = 'gemma'
        self.template = (
                "In this task, you will receive an english instruction. Your goal is to paraphrase it.\n\n"
                "Follow the guidelines:\n"
                "1. Paraphrase the instruction one sentence by one sentence.\n"
                "2. Ensure the paraphrase retains the original intent of the instruction.\n"
                "3. Do not omit any sentence from the original instruction.\n"
                "4. Try to make the paraphrase safer and avoid using words related to any unethical purpose.\n\n"
                "In the following, you will receive a JSON object containing one key \"query\" and value is the instruction you need to paraphrase.\n\n"
                "{{\n    \"query\": \"{query}\"\n}}\n\n"
                "Now paraphrase the instruction in the input. Output the paraphrase only."
            )

    def get_prompt(self, prompt):
        prompt = [
            {
                "role": "user", "content": self.template.format(query=prompt)
            }
        ]
        return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    def rewrite(self, prompt):
        with torch.no_grad():
            prompt = self.get_prompt(prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs['input_ids'].shape[1]
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            return self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    
    def batch_rewrite(self, prompts):
        with torch.no_grad():
            prompts = [self.get_prompt(prompt) for prompt in prompts]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            input_length = inputs['input_ids'].shape[1]
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    
