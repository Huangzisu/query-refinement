from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import time
import pickle
import modules.rewrite_module as rewrite_module
import modules.target_module as target_module
import modules.refine_module as refine_module
import modules.judge_module as judge_module
import argparse
from tqdm import tqdm
import os
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch


seed = 20
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class Generator():
    def __init__(self, refine):
        self.refine = refine

    def generate(self, prompt):
        if self.refine.name == "Refiner":
            rewritten_prompt, response = self.refine.defend(prompt)
            return {
                "prompt": prompt,
                "rewritten_prompt": rewritten_prompt,
                "response": response
            }
        elif self.refine.name == "BaselineParaphraseDefend":
            rewritten_prompt, response, token_usage = self.refine.defend(prompt)
            return {
                "prompt": prompt,
                "rewritten_prompt": rewritten_prompt,
                "response": response,
                "token_usage": token_usage
            }
        elif self.refine.name == "SemanticSmoothParaphraseDefend":
            output, purtabations, all_outputs, are_copies_jailbroken, token_usage = self.refine.defend(prompt)
            return {
                "prompt": prompt,
                "output": output,
                "purtabations": purtabations,
                "all_outputs": all_outputs,
                "are_copies_jailbroken": are_copies_jailbroken,
                "token_usage": token_usage
            }
        else:
            response = self.refine.defend(prompt)
            return {
                "prompt": prompt,
                "response": response
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_path", type=str, help="Path to the target model")
    parser.add_argument("--method", type=str, default=None, help="Method")
    parser.add_argument("--rfm_config_path", type=str, default="", help="Path to the refinement model config")
    parser.add_argument("--rfm_weights_path", type=str, default="", help="Path to the refinement model weights")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--task", type=str, help="task")
    parser.add_argument("--save_path", type=str, help="Path to save the result")

    args = parser.parse_args()

    device_map = 'auto'
    method = args.method
    target_model_path = args.target_model_path
    rfm_config_path = args.rfm_config_path
    rfm_weights_path = args.rfm_weights_path
    data_path = args.data_path
    save_path = args.save_path
    eval_task = args.task

    if 'pair' in eval_task.lower():
        with open(data_path, "rb") as f:
            data = pickle.load(f)['jailbreak_prompt'].tolist()
    elif 'autodan' in eval_task.lower():
        with open(data_path, "r") as f:
            data = json.load(f)
            data = [data['jailbreak_prompt'][i]+ " " +data['goal'][i] for i in range(len(data['goal']))]
    elif 'gcg' in eval_task.lower():
        with open(data_path, "r") as f:
            data = json.load(f)
            data = [data['goal'][i]+ " " + data['controls'][i] for i in range(len(data['goal']))]
    elif 'renellm' in eval_task.lower():
        with open(data_path, "r") as f:
            data = json.load(f)
            data = [x['nested_prompt'] for x in data]
    elif 'alpaca_eval' in eval_task.lower():
        with open(data_path, "r") as f:
            data = json.load(f)
    data = data[:1]

    
    if "vicuna" in target_model_path:
        target = target_module.TargetVicuna(target_model_path, device_map)
        print("target model(vicuna) loaded......")
    elif "Llama-2-7b-chat-hf" in target_model_path:
        target = target_module.TargetLlama2(target_model_path, device_map)
        print("target model(llama2) loaded......")
    print("target model device", target.model.device)


    if "Refiner" in method:
        if "gemma" in rfm_config_path:
            rewriter = rewrite_module.RewriterGemma(rfm_config_path, rfm_weights_path, device_map)
            print(f"rwm model {rfm_weights_path} loaded......")
        else:
            print("Refinement model not supported.")
            exit(0)
        refine = refine_module.Refiner(target, eval_task, rewriter)
        print("Using RewriterDefend......")
    elif "BaselineParaphrase" in method:
        refine = refine_module.BaselineParaphraseDefend(target, eval_task)
        print("Using BaselineParaphrase......")
    elif "SemanticSmoothParaphrase" in method:
        judge = judge_module.GPTJudge(openai_api_key=os.environ["OPENAI_API_KEY"])
        refine = refine_module.SemanticSmoothParaphraseDefend(target, eval_task, judge)
        print("Using SemanticSmoothParaphrase......")
    else:
        refine = refine_module.BaseRefinement(target, eval_task)
        print("Using BaseDefend......")
    
    
    generator = Generator(refine)

    generated = []
    for i, prompt in tqdm(enumerate(data), total=len(data), desc="Generating " + eval_task):
        if eval_task == 'alpaca_eval':
            prompt = prompt['instruction']
        result = generator.generate(prompt)
        generated.append(result)

    save = {
        "setting": {
            "target_model_path": target_model_path,
            "method": method,
            "rfm_config_path": rfm_config_path,
            "rfm_weights_path": rfm_weights_path,
            "data_path": data_path,
            "task": eval_task
        },
        "results": generated
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Saving......")
    with open(save_path, "w") as f:
        json.dump(save, f, indent=4)