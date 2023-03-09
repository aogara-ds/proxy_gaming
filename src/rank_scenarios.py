import torch
import pandas as pd
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                          AutoConfig, TrainingArguments, Trainer, get_linear_schedule_with_warmup
from datasets import Dataset, DatasetDict
import pdb
import wandb
import math
import argparse

def load_model(model_name: str, device: torch.device, checkpoint: str = None) -> AutoModelForSequenceClassification:
    # Load model
    config = AutoConfig.from_pretrained(model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    # Load pretrained checkpoint if available
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)

    return model

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_scenarios(load_data_path: str):
    df = pd.read_csv(load_data_path, header=None)
    scenarios = df.values.tolist()
    return scenarios


def get_score(text, model, tokenizer, device):
    encodings_dict = tokenizer(
        text,
        truncation=True,
        max_length=tokenizer.model_max_length, 
        padding=True,
        return_tensors="pt",
    )
    input_ids = encodings_dict["input_ids"].to(device)
    attn_masks = encodings_dict["attention_mask"].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attn_masks)
        logits = output.logits.reshape((-1))
    return logits

if __name__=="__main__":
    # Parse CLI inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_model_name", type=str, default="roberta-large", help="gold model for assigning labels")
    parser.add_argument("--checkpoint", type=str, default="../models/reward/util_deberta-v3-large_1e-05_16_2.pkl", help="checkpoint for gold model")
    parser.add_argument("--gold_device", type=str, default="cuda:0", help="device for gold model")
    parser.add_argument("--load_data_path", type=str, default="../data/unlabeled_scenarios/1.csv", help="path to unlabeled data")
    parser.add_argument("--save_data_dir", type=str, default="../data/ranked_scenarios/", help="folder to save scenarios ranked by gold model")
    args = parser.parse_args()

    # Load gold model 
    gold_model = load_model(args.gold_model_name, args.gold_device, args.checkpoint)
    gold_tokenizer = load_tokenizer(args.gold_model_name)

    # Load unlabeled data
    scenarios = load_scenarios(args.load_data_path)

    # Rank unlabeled data
    ranked_scenarios = []
    agreements = []
    for pair in scenarios:
        scenario_A, scenario_B = pair
        score_A = get_score(scenario_A, gold_model, gold_tokenizer, args.gold_device)
        score_B = get_score(scenario_B, gold_model, gold_tokenizer, args.gold_device)

        if score_A > score_B:
            ranked_scenarios.append(pair)
            agreements.append(True)
        else:
            ranked_scenarios.append(pair[::-1])
            agreements.append(False)
    
    print(f"Agreement with zero shot generator: {round(sum(agreements)/len(agreements)*100, 2)}%")

    # Save ranked data
    df = pd.DataFrame(ranked_scenarios)
    df.to_csv(args.save_data_dir + args.load_data_path.split("/")[-1], header=False, index=False)
    
    # Save ranked data for training
    if True:
        # Split into train, test, and test_hard sets
        test_hard = df.loc[~pd.Series(agreements)].iloc[:100]
        test = df.drop(test_hard.index).iloc[:100]
        train = df.drop(test_hard.index).drop(test.index)

        # Save train, test, and test_hard sets
        train.to_csv("../data/generated_util/util_train.csv", header=False, index=False)
        test.to_csv("../data/generated_util/util_test.csv", header=False, index=False)
        test_hard.to_csv("../data/generated_util/util_test_hard.csv", header=False, index=False)