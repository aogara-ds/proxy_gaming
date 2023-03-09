import torch
import pandas as pd
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                          AutoConfig, TrainingArguments, Trainer, get_linear_schedule_with_warmup
from datasets import Dataset, DatasetDict
from utils import get_scores, get_rankings
import pdb
import wandb
import math
import argparse

"""
1. Load gold model and tokenizer
2. Load pretrained base for proxy model and tokenizer
3. Load unlabeled data
4. Score unlabeled data
5. Create training dataset 
6. Train proxy model 
7. Evaluate proxy model [Not implemented yet]
"""

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


def load_scenarios(load_data_path: str, loss_fn: str):
    df = pd.read_csv(load_data_path, header=None)
    scenarios = df.values.tolist()
    return scenarios


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, device: torch.device) -> DatasetDict:
    tokenized_data = tokenizer(dataset["text"], padding="max_length", truncation=True, return_tensors="pt").to(device)
    return {"text": dataset['text'],
            "input_ids": tokenized_data['input_ids'].detach().cpu().numpy(),
            "attention_mask": tokenized_data['attention_mask'].detach().cpu().numpy(),
            "label": dataset['label']}


def create_dataset(scenarios: List[str], scores: torch.Tensor, tokenizer: AutoTokenizer, 
                   device: torch.device, train_split: float = 0.8) -> Dataset:
    assert train_split >= 0 and train_split <= 1, "train_split must be between 0 and 1"
    df = pd.DataFrame({"text": scenarios, "label": scores.detach().cpu().numpy()})
    dataset = DatasetDict({"train": Dataset.from_pandas(df.head(int(df.shape[0] * train_split)), preserve_index=False), 
                           "eval": Dataset.from_pandas(df.tail(df.shape[0] - int(df.shape[0] * train_split)), preserve_index=False)})
    tokenized_dataset = dataset.map(tokenize_dataset, batched=True, fn_kwargs={"tokenizer": tokenizer, "device": device})
    # TODO: Save another split of this data for evaluation. 
    # TODO: Should we save this to disk or just train the proxy model? 
    return tokenized_dataset


def compute_mse(eval_pred: List[torch.Tensor]) -> Dict[str, float]:
    preds, labels = eval_pred
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    loss_fn = torch.nn.MSELoss()
    return {"mse_loss": loss_fn(preds, labels).item()}


def get_optimizer_and_scheduler(args: argparse.Namespace, proxy_model: AutoModelForSequenceClassification, 
                                num_train_examples: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    # Initialize hyperparameters
    steps_per_epoch = math.ceil(num_train_examples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    
    # Initialize optimizer and LR scheduler
    optimizer = torch.optim.Adam(proxy_model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    return optimizer, lr_scheduler


def train_mse(args: argparse.Namespace, proxy_model: AutoModelForSequenceClassification, proxy_tokenizer: AutoTokenizer, 
              scenarios: List[List[str]]) -> AutoModelForSequenceClassification:
    # Create Hugging Face dataset
    dataset = create_dataset(scenarios, scores, proxy_tokenizer, args.proxy_device)

    # Get optimizer and LR scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(args, proxy_model, len(dataset['train']))

    # Specify problem type to designate MSE loss
    # See https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L1563-L1583
    proxy_model.config.problem_type = "regression" 
    
    # Train proxy model
    training_args = TrainingArguments(
        output_dir="proxy_trainer", 
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        report_to="wandb",
        logging_strategy="epoch",
        learning_rate=4e-3,
        do_eval=True,
        evaluation_strategy="epoch", 
        per_device_eval_batch_size=args.batch_size,
    )

    trainer = Trainer(
        model=proxy_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        compute_metrics=compute_mse,
        optimizers = (optimizer, lr_scheduler)
    )

    trainer.train()

    # Evaluate proxy model
    wandb.finish()

    return proxy_model


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.cat([tensor[:, 0], tensor[:, 1]])
    return tensor


def unflatten(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.stack([tensor[:tensor.shape[0] // 2], tensor[tensor.shape[0] // 2:]], axis=1)
    return tensor


def train_pairwise(args: argparse.Namespace, proxy_model: AutoModelForSequenceClassification, proxy_tokenizer: AutoTokenizer, 
                   scenarios: List[List[str]]) -> AutoModelForSequenceClassification:
    # Get optimizer and LR scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(args, proxy_model, len(dataset['train']))

    # Get pairwise ranking labels
    labels = get_rankings(scenarios, proxy_model, proxy_tokenizer, args.proxy_device)

    # Create PyTorch dataset and dataloader
    flat_scenarios = [text for column in scenarios for text in column]
    tokenized_data = tokenizer(flat_scenarios, padding="max_length", truncation=True, return_tensors="pt").to(device)
    dataset = TensorDataset(flat_scenarios, input_ids, masks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Train proxy model
    # Set model to training mode
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    ntrain_steps = len(train_dataloader)




    


def get_save_path(save_dir: str):
    # TODO: Link with wandb to save metadata about each run
    file_name = str(len([name for name in os.listdir(save_dir)
                    if os.path.isfile(os.path.join(save_dir, name))]))
    full_path = save_dir + '/' + file_name + '.pt'
    return full_path


if __name__=="__main__":
    # Parse CLI inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_model_name", type=str, default="roberta-large", help="gold model for assigning labels")
    parser.add_argument("--checkpoint", type=str, default="models/reward/util_roberta-large.pt", help="checkpoint for gold model")
    parser.add_argument("--gold_device", type=str, default="cuda:0", help="device for gold model")
    parser.add_argument("--proxy_model_name", type=str, default="roberta-large", help="pretrained base for proxy model")
    parser.add_argument("--proxy_device", type=str, default="cuda:0", help="device for proxy model")
    parser.add_argument("--load_data_path", type=str, default="data/unlabeled_scenarios/1.csv", help="path to unlabeled data")
    parser.add_argument("--loss_fn", type=str, default="mse", help="loss function to use for training")
    parser.add_argument("--save_dir", type=str, default="models/proxy/", help="folder to save proxy model")
    parser.add_argument("--lr", type=float, default=4e-3, help="learning rate for training")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs to train for")
    parser.add_argument("--warmup_frac", type=float, default=0.1, help="fraction of total steps to warmup for")
    args = parser.parse_args()

    print(type(args))

    # Load gold model 
    gold_model = load_model(args.gold_model_name, args.gold_device, args.checkpoint)
    gold_tokenizer = load_tokenizer(args.gold_model_name)

    # Load pretrained base for proxy model 
    proxy_model = load_model(args.proxy_model_name, args.proxy_device)
    proxy_tokenizer = load_tokenizer(args.proxy_model_name)

    # Load unlabeled data
    scenarios = load_scenarios(args.load_data_path, args.loss_fn)
    # Initialize wandb
    wandb.init(project='Proxy Gaming', entity='aogara')

    # Train proxy model
    if args.loss_fn == "mse":
        flat_scenarios = [text for column in scenarios for text in column]
        labels = get_scores(flat_scenarios, args.gold_device, gold_tokenizer, gold_model)
        proxy_model = train_mse(args, proxy_model, proxy_tokenizer, scenarios)
    
    elif args.loss_fn == "pairwise":
        labels = get_rankings(scenarios, gold_model, gold_tokenizer, args.gold_device)
        proxy_model = train_pairwise(args, proxy_model, proxy_tokenizer, scenarios, labels)
        
    # Save proxy model
    save_path = get_save_path(args.save_dir)
    torch.save(proxy_model.state_dict(), save_path)