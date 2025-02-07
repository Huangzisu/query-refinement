import torch
import torch.nn.functional as F


# base class for reward module
class RewardModuleBase:
    def __init__(self):
        pass
    
    def get_prompt(self):
        pass

    def get_rewards(self, prompts: list, responses: list):
        raise NotImplementedError
    

class RMGemmaRewardModule(RewardModuleBase):
    def __init__(self, gemma_rm_pipeline, gemma_tokenizer):
        self.gemma_rm_pipeline = gemma_rm_pipeline
        self.gemma_tokenizer = gemma_tokenizer
        self.pipe_kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": 1
            }
    
    def get_prompt(self, prompts, responses):
        result = []
        for prompt, response in zip(prompts, responses):
            chat = [
                {
                    "role": "user", "content": prompt
                },
                {
                    "role": "assistant", "content": response
                }
            ]
            result.append(
                self.gemma_tokenizer.apply_chat_template(chat, tokenize=False,
                                                                  add_generation_prompt=False).replace(self.gemma_tokenizer.bos_token, "")
            )
        return result

    def get_rewards(self, prompts: list, responses: list, step:int):
        reward_input = self.get_prompt(prompts, responses)
        if step == 0:
            print("quality reward inputs: ", reward_input)
        reward_outputs = self.gemma_rm_pipeline(reward_input, **self.pipe_kwargs)
        rewards = torch.tensor([output[0]["score"] for output in reward_outputs])
        return rewards
    

class LlamaGuardRewardModule(RewardModuleBase):
    def __init__(self, llama_guard_model, llama_guard_tokenizer, z3_enabled=True):
        self.llama_guard_model = llama_guard_model
        self.llama_guard_tokenizer = llama_guard_tokenizer
        self.safe_token_id = self.llama_guard_tokenizer.encode("safe")[-1]
        self.unsafe_token_id = self.llama_guard_tokenizer.encode("unsafe")[-1]
        self.z3_enabled = z3_enabled

    def get_prompt(self, prompts, responses):
        chats = []
        for prompt, response in zip(prompts, responses):
            chat = [
                {
                    "role": "user", "content": prompt
                },
                {
                    "role": "assistant", "content": response
                }
            ]
            chats.append(
                self.llama_guard_tokenizer.apply_chat_template(chat, tokenize=False)
            )
        return chats

    def get_rewards(self, prompts: list, responses: list, step:int, device):
        chats = self.get_prompt(prompts, responses)
        if step == 0:
            print("llama guard inputs: ", chats)
        input = self.llama_guard_tokenizer(chats, return_tensors='pt', padding=True).to(device=device)
        prompt_length = input['input_ids'].shape[1]
        seq = self.llama_guard_model.module.generate(**input, 
                                                       max_new_tokens=100, 
                                                       pad_token_id=self.llama_guard_tokenizer.pad_token_id,
                                                       synced_gpus=self.z3_enabled, 
                                                       do_sample=False)
        attention_mask = seq.not_equal(self.llama_guard_tokenizer.pad_token_id).long()
        out = self.llama_guard_model(seq, attention_mask=attention_mask)
        target_logits = [torch.tensor([out.logits[i, prompt_length-1, self.safe_token_id].item(), out.logits[i, prompt_length-1, self.unsafe_token_id].item()]) 
                            for i in range(len(chats))]
        probs = [F.softmax(logits, dim=0)[0].item() for logits in target_logits]

        print(f"--- LlamaGuard response --> rank={torch.distributed.get_rank()}, {self.llama_guard_tokenizer.batch_decode(seq[:, prompt_length:], skip_special_tokens=True)}")
        del seq
        del out

        return torch.tensor(probs)
    


class QualitySafetyRewardModule(RewardModuleBase):
    def __init__(self, safety_reward_module, quality_reward_module, weight_safety, weight_quality):
        self.safety_reward_module = safety_reward_module
        self.quality_reward_model = quality_reward_module
        self.weight_safety = weight_safety
        self.weight_quality = weight_quality

    def get_rewards(self, prompts: list, responses: list, step:int, device):
        quality_rewards = self.quality_reward_model.get_rewards(prompts, responses, step)
        safety_rewards = self.safety_reward_module.get_rewards(prompts, responses, step, device)

        rewards = self.weight_quality * quality_rewards + self.weight_safety * safety_rewards
    
        return rewards, quality_rewards, safety_rewards