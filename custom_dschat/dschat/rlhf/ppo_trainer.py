# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import nltk
nltk.data.path.append('/cpfs01/projects-HDD/cfff-54173e75d604_HDD/wxh_22212010038/RWM-rl/nltk_data')
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
from fastchat.conversation import get_conv_template

from dschat.utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.protected_model = self.rlhf_engine.protected
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.tokenizer = self.rlhf_engine.tokenizer
        self.reward_safety_tokenizer = self.rlhf_engine.reward_safety_tokenizer
        self.protected_tokenizer = self.rlhf_engine.protected_tokenizer
        self.reward_quality_tokenizer = self.rlhf_engine.reward_quality_tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # reward module
        self.reward_module = self.rlhf_engine.reward_module

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        self.last_generated_experience = None

        # Those value can be changed
        self.kl_ctl = 0.01
        self.clip_reward_value = 10
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1
        self.lam = 0.95
        self.generate_time = 0.0

        # number of experiences to generate for each data point
        self.num_generate_samples = 2
        # max length for rewriting
        self.max_rewrite_length = 128

    

    def _generate_sequence(self, prompts, mask, original_instruction, step):
        max_min_length = self.max_rewrite_length + prompts.shape[1]
        if step == 0:
            print("generate_sequence max_length: ", max_min_length)
            print("prompt sample: ",self.tokenizer.decode(prompts[0], skip_special_tokens=True))
        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9

        with torch.no_grad():
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=self.num_generate_samples)

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        out_seq = []
        out_ans = []
        valid_index = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}\n'
                    f'rewritten-prompts: {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}'
                )
                continue
            else:
                out_seq.append(seq[i:i+1])
                out_ans.append(self.tokenizer.decode(ans[i], skip_special_tokens=True))
                valid_index.append(i)

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}\n'
                f'-> rewritten-prompts: {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}'
            )
            return None
        
        print(
            f"original_prompt: rank={torch.distributed.get_rank()}, {original_instruction}\n"
            f"rewritten_prompt: rank={torch.distributed.get_rank()}, {[item for item in out_ans]}"
        )

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq, out_ans

    def generate_experience(self, prompts, mask, original_instruction, step):
        self.eval()
        generate_start = time.time()

        # Get the whole generated sequence and rewritten prompts
        seq, rewritten_prompts_decoded = self._generate_sequence(prompts, mask, original_instruction, step)

        rewritten_input_decoded = self.fit_vicuna_template(rewritten_prompts_decoded)
        if step == 0:
            print("rewritten_input_decoded: ", rewritten_input_decoded)
        
        rewritten_input = self.protected_tokenizer(rewritten_input_decoded, return_tensors="pt", padding=True)

        if self.args.local_rank == -1:
            device = torch.device(get_accelerator().device_name())
        else:
            device = torch.device(get_accelerator().device_name(), self.args.local_rank)
        rewritten_input = rewritten_input.to(device=device)

        self.prompt_length = rewritten_input['input_ids'].shape[1]
        max_min_length = self.max_answer_seq_len
        if self.prompt_length > max_min_length:
            print("Input length is too long, truncating to max_min_length!")
            rewritten_input['input_ids'] = rewritten_input['input_ids'][:, :max_min_length]
            rewritten_input['attention_mask'] = rewritten_input['attention_mask'][:, :max_min_length]
            
        with torch.no_grad():
            response_seq = self.protected_model.module.generate(
                rewritten_input['input_ids'],
                attention_mask=rewritten_input['attention_mask'],
                max_length=max_min_length,
                pad_token_id=self.protected_tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                do_sample=False
                )
   
        responses_decoded = self.protected_tokenizer.batch_decode(response_seq[:, self.prompt_length:], skip_special_tokens=True)

        if self.args.print_answers and (step % self.num_generate_samples
                                        == 0):
            print(
                f"--- response --> step={step}, rank={torch.distributed.get_rank()}, {responses_decoded}"
            )
        
        generate_end = time.time()
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}

        self.train()

        attention_mask = seq.not_equal(self.tokenizer.pad_token_id).long()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)

            original_prompts = []
            for item in original_instruction:
                original_prompts.extend([item] * self.num_generate_samples)

            rewards, quality_rewards, safety_rewards = self.reward_module.get_rewards(original_prompts, responses_decoded, step, device)
            rewards = rewards.to(device)
            quality_rewards = quality_rewards.to(device)
            safety_rewards = safety_rewards.to(device)

            if step % 5 == 0:
                print(f"--- safety rewards --> step={step}, rank={torch.distributed.get_rank()}, {safety_rewards}"
                      f"--- quality rewards --> step={step}, rank={torch.distributed.get_rank()}, {quality_rewards}"
                      f"--- rewards --> step={step}, rank={torch.distributed.get_rank()}, {rewards}"
                )

            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start


        
        logprobs = gather_log_probs(logits[:, :-1, :], seq[:, 1:])
        ref_logprobs = gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:])
        # Estimate kl divergence
        log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
        kl = log_ratio.exp() - 1 - log_ratio

        return {
            'prompts': prompts,
            'rewritten_inputs': rewritten_input['input_ids'],
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs,
            'value': values,
            'rewards': rewards,
            'quality_rewards': quality_rewards,
            'safety_rewards': safety_rewards,
            'input_ids': seq,
            "attention_mask": attention_mask,
            'kl': kl
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        # print(rewards)
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs, step):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        rewritten_prompts = inputs['rewritten_inputs']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            if step == 0:
                print("old_rewards[:3]: ", old_rewards[:3].sum(dim=1))
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()

        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.ref_model.eval()
        self.protected_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        # reward_quality_model_norm = get_model_norm(self.reward_quality_model)
        # reward_safety_model_norm = get_model_norm(self.reward_safety_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        # print_all_ranks(f'{tag} global_reward_quality_model_norm', reward_quality_model_norm,
        #                 self.args.local_rank)
        # print_all_ranks(f'{tag} global_reward_safety_model_norm', reward_safety_model_norm,
        #                 self.args.local_rank)

    def fit_vicuna_template(self, rewritten_prompts_decoded):
        sys = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        template = "{sys} USER: {instruction} ASSISTANT:"
        result = []
        for rewritten_prompt_decoded in rewritten_prompts_decoded:
            result.append(template.format(sys=sys, instruction=rewritten_prompt_decoded))
        return result
        

class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
