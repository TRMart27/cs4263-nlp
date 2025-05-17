import argparse
import os
import tqdm
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from cs4263.models.bilstm import BiLSTMClassifier
from cs4263.models.cnn import CNNClassifier

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
STOP_WORDS = set(stopwords.words('english'))

@torch.no_grad()
def bert_logits(model, ids, masks):
    return model(input_ids=ids, attention_mask=masks).logits

@torch.no_grad()
def get_model_logits(model, ids):
    return model(ids)

def main(args):
    #define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    #load berty
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=6,
        problem_type="multi_label_classification",
    ).to(DEVICE)

    #load fine-tuned berty + set mode
    print(f"[INFO] Loading BERT model from {args.bert_path}...")
    bert_model.load_state_dict(torch.load(args.bert_path, map_location=DEVICE))
    bert_model.eval()
    print(f"[INFO] Success!\n")


    #load bilstm
    lstm = BiLSTMClassifier(
        len(tokenizer),
        pad_idx=tokenizer.pad_token_id,
    ).to(DEVICE)

    #load weights + set mode
    print(f"[INFO] Loading BiLSTM model from {args.lstm_path}...")
    lstm.load_state_dict(torch.load(args.lstm_path, map_location=DEVICE))
    lstm.eval()
    print(f"[INFO] Success!\n")

    #cnn
    cnn = CNNClassifier(
        len(tokenizer),
        pad_idx=tokenizer.pad_token_id,
    ).to(DEVICE)

    #load weights
    print(f"[INFO] Loading CNN model from {args.cnn_path}...")
    cnn.load_state_dict(torch.load(args.cnn_path, map_location=DEVICE))
    cnn.eval()
    print(f"[INFO] Success!\n")

    #load data
    validate_data = torch.load(os.path.join(args.data_dir, "validate.pt"))
    val_ids    = validate_data["input_ids"].to(DEVICE)
    val_masks  = validate_data["attention_mask"].to(DEVICE)
    val_labels = validate_data["labels"].to(DEVICE)

    all_probs, all_preds = [], []
    thresholds = torch.tensor([0.65, 0.45, 0.65, 0.55, 0.55, 0.55]).to(DEVICE)
    for start in tqdm.tqdm(range(0, val_ids.size(0), args.batch_size)):
        end = start + args.batch_size

        chunk_ids = val_ids[start:end]
        chunk_masks = val_masks[start:end]
        chunk_labels = val_labels[start:end]

        logits = (
            0.50 * bert_logits(bert_model, chunk_ids, chunk_masks) +
            0.25 * get_model_logits(lstm, chunk_ids) +
            0.25 * get_model_logits(cnn, chunk_ids)
        )

        #TODO manual gridsearch to find
        #   1) optimal ensemble weights
        #   2) optimal threshold weights

        #compute prob + predictions
        probs = torch.sigmoid(logits)
        preds = (probs > thresholds).int()

        #send back to cpu
        probs = probs.cpu()
        preds = preds.cpu()

        #store em
        all_probs.append(probs)
        all_preds.append(preds)

        #get true labels
        expected = val_labels.cpu()
        expected = expected.numpy()

    print("\n###################################################################\n")
    #concat the sutfff an d  thingssszaaa
    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)

    #compute f1
    macro_f1 = f1_score(expected, preds.numpy(), average="macro")
    per_label_f1 = f1_score(expected, preds.numpy(), average=None)
    print(f"[METRIC] F1")
    print(f"\t[METRIC] Macro F1: {macro_f1:.4f}\n")
    for label, f1 in zip(LABELS, per_label_f1):
        print(f"\t[METRIC] {label} F1: {f1:.4f}")

    #compute auc
    print(f"[METRIC] ROC_AUC")
    per_label_auc = []
    for i in range(6):
        try:
            auc_score = roc_auc_score(expected[:, i], probs[:, i])
        except ValueError:
            auc_score = np.nan
        per_label_auc.append(auc_score)
    macro_auc = np.nanmean(per_label_auc)

    print(f"\t[METRIC] Macro AUC_ROC: {macro_auc:.4f}\n")
    for label, auc_score in zip(LABELS, per_label_auc):
        print(f"\t[METRIC] {label} AUC_ROC: {auc_score:.4f}")

    print("\n###################################################################\n")
    #save predictions
    save_path = os.path.join(args.save_path, "ensemble_results.pt")
    torch.save(probs, save_path)
    print(f"[INFO] Saved ensemble results to {save_path}/ensemble_results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--base_model",   default="bert-base-uncased")
    parser.add_argument("--bert_path",  default="models/best.pt")
    parser.add_argument("--lstm_path",  default="models/bilstm_best.pt")
    parser.add_argument("--cnn_path",   default="models/cnn_best.pt")
    parser.add_argument("--save_path",   default="models/runs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=50)

    main(parser.parse_args())
