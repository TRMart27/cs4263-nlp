import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pt(path):
    data = torch.load(path)
    return data["input_ids"], data["attention_mask"], data["target_labels"]

def compute_f1(logits, labels):
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return f1_score(labels.cpu().numpy(), preds, average="macro")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    #get the data
    train_ids, train_masks, train_labels = load_pt(os.path.join(args.data_dir, "train.pt"))
    val_ids, val_mask, val_labels = load_pt(os.path.join(args.data_dir, "validate.pt"))

    #define the TensorDatasets for later use
    train_dataset    = TensorDataset(train_ids, train_masks, train_labels)
    validate_dataset = TensorDataset(val_ids, val_mask, val_labels)

    # Compute pos_weight for class imbalance
    counts = train_labels.sum(0)
    neg = train_labels.size(0) - counts
    pos_weight = (neg / counts).to(DEVICE)

    #Sampler for class imbalance
    class_weight = (neg / counts).cpu().numpy()
    sample_weight = (train_labels.cpu().numpy() * class_weight).sum(1) + 1e-6
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_w), replacement=True)

    # get the data loaders for train and validate
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size
    )

    val_loader   = DataLoader(validate_dataset, batch_size=args.batch_size*2,
                              shuffle=False
    )

    #define tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    #define model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=6,
        problem_type="multi_label_classification",
    ).to(DEVICE)

    #define optimizer
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=0.01,
    )

    #define scheduler
    num_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(0.1 * num_steps),
        num_steps,
    )

    #define loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):

        # set model to training mode
        model.train()
        train_loss = 0.0

        for ids, mask, labels in train_loader:

            #sent the stuff to the GPU
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

            #zero out the gradients
            optimizer.zero_grad()

            #send through model
            outputs = model(input_ids=ids, attention_mask=mask)

            #get the logits
            logits = outputs.logits

            #compute the loss
            loss = criterion(logits, labels)

            #backprop
            loss.backward()

            #norm the gradients for explosion
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            #move on
            optimizer.step()
            scheduler.step()

            #accumulate loss
            train_loss += loss.item() * labels.size(0)
        #average
        train_loss /= len(train_loader.dataset)

        # set model to evaluation mode - doesnt update gradients
        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad(): #dont update gradients
            for ids, mask, labels in val_loader:

                #send the stuff to the gpu
                ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

                #send through the model
                outputs = model(input_ids=ids, attention_mask=mask)

                #get the logits
                logits = outputs.logits

                #compute the loss
                loss = criterion(logits, labels)

                #accumulate loss
                val_loss += loss.item() * labels.size(0)

                #cache that shiz
                all_logits.append(logits)
                all_labels.append(labels)
        #average
        val_loss /= len(val_loader.dataset)

        #concat to torch tensors
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        #compure the f1
        val_f1 = compute_f1(all_logits, all_labels)

        print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")

        #see if this is better
        if val_f1 > best_f1:
            #update best
            best_f1 = val_f1

            #build save path + save
            save_path = os.path.join(args.output_dir, "best.pt")
            torch.save(model.state_dict(), save_path)
            print(f" Saved best model (F1={best_f1:.4f}) to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--output_dir",  default="models")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    main(parser.parse_args())

