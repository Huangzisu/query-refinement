from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import numpy as np
import random


data_path = "./data/bpo_train.json"
base_model_name = "/data/model/gemma-2b-zephyr-sft"
output_dir = "/data/model/sft_gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = (
    0
)
tokenizer.padding_side = "right"


def process_bpo(examples):
    template = (
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
    prompt = [
        {
            "role": "user", "content": template.format(query=examples['prompt'])
        }
    ]

    examples["prompt"] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    examples["output"] = examples['optimized_prompt'] + tokenizer.eos_token
    examples['text'] = examples["prompt"] + examples["output"]
    return examples


if __name__ == "__main__":
    train_data = load_dataset('json',data_files=data_path)['train']
    print(train_data)
    train_data = train_data.map(process_bpo)
    print(train_data[0]['text'])
    print(tokenizer(train_data[0]['text']))

    device_map = "auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    print(base_model)
    base_model.config.pretraining_tp = 1 

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/training_logs",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=100,
        num_train_epochs=3,
        #max_steps=3000,
        #logging_steps = 1
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        warmup_ratio=0.1,
        report_to='tensorboard',
        seed=42
    )

    from transformers import set_seed
    def set_random_seed(seed):
        if seed is not None:
            set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    print("seed: ", training_args.seed)
    set_random_seed(training_args.seed)

    max_seq_length = 512

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    import os
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir) # type: ignore