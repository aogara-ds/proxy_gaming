import os
import pathlib
from typing import List
import argparse
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import trlx
from trlx.data.configs import TRLConfig
from utils import get_scores

def load_model(model: str, load_path: str):
    config = AutoConfig.from_pretrained(model, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model, config=config, )
    model.load_state_dict(torch.load(load_path, map_location='cuda:0'), strict=False)  
    return model

def get_prompt_dataset(prompts: List[str], max_length: int) -> List[str]:
    formatted_prompts = []
    for i in tqdm(range(len(prompts))):
        tmp = tokenizer.decode(
            tokenizer(
                prompts[i],
                truncation=True,
                max_length=max_length,)["input_ids"],
            skip_special_tokens=True,
        ).strip()
        formatted_prompts.append(tmp)
    return formatted_prompts

def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> torch.tensor:
    # This function depends on global variables because TRLX doesn't include inputs for all necessary vars
    # Is there any way to fix that?
    original_samples = [prompt + " " + post_continuation_dict.get(prompt) for prompt in prompts]
    original_scores = get_scores(original_samples, rw_device, tokenizer, rw_model)
    scores = get_scores(samples, rw_device, tokenizer, rw_model)
    norms_scores = scores - original_scores
    return norms_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="roberta-large", help="base reward model")
    parser.add_argument("--dataset_path", type=str, default="data/utility_dataset.hf", help="path to load HuggingFace dataset of continuation pairs")
    parser.add_argument("--reward_checkpoint_path", type=str, default="models/reward/util_roberta-large.pt", help="path to load reward model weights")
    args = parser.parse_args()
    
    rw_tokenizer = AutoTokenizer.from_pretrained(args.model)
    # rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = load_model(args.model, args.reward_checkpoint_path)
    rw_device = torch.device("cuda:0") 
    rw_model.to(rw_device)
    print("loaded reward model")

    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_gpt2_utility.yml")
    config = TRLConfig.load_yaml(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = (config.train.seq_length - config.method.gen_kwargs["max_new_tokens"])

    dataset = load_from_disk(args.dataset_path)
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["val"]]
    train_posts, train_continuations = zip(*train_set)
    val_posts, val_continuations = zip(*val_set)
    post_continuation_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_continuation_dict[train_prompts[i]] = train_continuations[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_continuation_dict[val_prompts[i]] = val_continuations[i]
    print("loaded dataset")

    trainer = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],
        config=config,
    )
    print("done training")
