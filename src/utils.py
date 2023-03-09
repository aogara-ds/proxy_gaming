import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

def get_scores(
    samples: List[str], 
    device: torch.device, 
    tokenizer: AutoTokenizer, 
    model: AutoModelForSequenceClassification, 
    max_length: int = 512, 
    batch_size: int = 2
) -> torch.tensor:
    """
    Score a list of strings using the provided reward model. Returns a tensor of scores. 
    """
    scores_list = []
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]
        encodings_dict = tokenizer(
            sub_samples,
            truncation=True,
            max_length=max_length, 
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(device)
        attn_masks = encodings_dict["attention_mask"].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attn_masks)
            logits = output.logits.reshape((-1))
        scores_list.append(logits)
    scores = torch.cat(scores_list, dim=0)
    return scores

def get_rankings(
    scenario_pairs: List[List[str]], 
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    device: torch.device, 
    max_length: int = 512, 
    batch_size: int = 2
) -> torch.tensor:
    score_pairs = []
    for scenario_list in scenario_pairs:
        # Get score for each scenario
        scores = get_scores(scenario_list, device, tokenizer, model, max_length, batch_size)
        score_pairs.append(scores)
    # Return the index of the higher score for each pair
    rankings = torch.argmax(torch.stack(score_pairs, dim=0), dim=0)
    return rankings
    