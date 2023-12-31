import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PromptTuningConfig,
    TaskType,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

MODEL_NAME = "25-percent-removed"
def train(
    use_lora: bool = True,
    use_prompt_tuning: bool = False,
    # model/data params
    base_model: str = "FreedomIntelligence/phoenix-inst-chat-7b",  # the only required argument
    data_path: str = f"llama2-data/{MODEL_NAME}/train.json",
    dev_data_path: str = f"llama2-data/{MODEL_NAME}/dev.json",
    output_dir: str = f"models/{MODEL_NAME}",
    val_set_size: int = 300,
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    cutoff_len: int = 3225,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # lora_target_modules: List[str] = [
    #     "q_proj",
    #     "v_proj",
    # ],
    lora_target_modules: List[str] = 
       ["query_key_value"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    # resume_from_checkpoint: str = '',  # either training checkpoint or final adapter
    # resume_from_checkpoint: str = './saved_models/no_added_errors/other/500/checkpoint-X',
    resume_from_checkpoint: str = f'models/{MODEL_NAME}/checkpoint-7500',
    prompt_template_name: str = "phoenix",  # The prompt template to use, will default to alpaca.
):
    print("LOCAL RANK", int(os.environ.get("LOCAL_RANK", 0)))
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"use_lora: {use_lora}\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
             f"dev_data_path: {dev_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='FreedomIntelligence/phoenix-inst-chat-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    print("TOTAL GRAD ACC STEPS", gradient_accumulation_steps)

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    # torch.cuda.set_device(4)
    # print("DEVICE", torch.cuda.current_device())
    # device_map = {"lm_head": 1, "transformer": 1}
    print("DDD", torch.cuda.current_device())
    device_map={'':torch.cuda.current_device()}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("WORLD SIZE", world_size, "NEW GRAD ACC STEPS", gradient_accumulation_steps)


    # Check if parameter passed or if set within environ
    use_wandb = False
    # use_wandb = len(wandb_project) > 0 or (
    #     "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if use_lora or use_prompt_tuning:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            cache_dir="cache/transformers",
        )
        print("DEVICE MAP", model.hf_device_map)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            cache_dir="cache/transformers",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        device_map=device_map,
        cache_dir="cache/transformers"
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        # print(tokenized_full_prompt)
        return tokenized_full_prompt
    if use_lora:
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    elif use_prompt_tuning:
        model = prepare_model_for_kbit_training(model)

        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=8,
            tokenizer_name_or_path=tokenizer,
        )
        model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, cache_dir="cache/datasets")
    else:
        data = load_dataset(data_path)
    
    if dev_data_path.endswith(".json") or dev_data_path.endswith(".jsonl"):
        dev_data = load_dataset("json", data_files=dev_data_path, cache_dir="cache/datasets")
    else:
        dev_data = load_dataset(dev_data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            # resume_from_checkpoint = (
            #     False  # So the trainer won't try loading its state
            # )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if use_lora:
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


    if val_set_size > 0:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, load_from_cache_file=True)
        # val_data = None
        val_data = dev_data['train'].shuffle().map(generate_and_tokenize_prompt, load_from_cache_file=True)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, load_from_cache_file=True)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        print("DDP 2!!!")
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=len(train_data) // 10, # Rougly 10% of the training data
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            optim="adamw_torch",
            logging_steps=500,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=500 if val_set_size > 0 else None,
            save_steps=500,
            logging_first_step = True,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    # early_stopping_callback = transformers.EarlyStoppingCallback(
    #     early_stopping_patience=4,
    #     early_stopping_threshold=1e-3,
    # )
    # trainer.add_callback(early_stopping_callback)

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    trainer.train()
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
