import argparse
import os

import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def predict(texts, tokenizer, model, threshold=0.5):
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(DEVICE)

    logits = model(**encodings).logits
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).int()
    return probs.cpu(), preds.cpu()


def main(args):
    #define model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=6,
        problem_type="multi_label_classification"
    ).to(DEVICE)

    #load model
    state_dict = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    #set to evaluation mode for predictions
    model.eval()

    #define tokenzier
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")
