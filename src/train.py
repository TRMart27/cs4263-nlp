import argparse
import os
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score

from cs4263.models.bilstm import BiLSTMClassifier
from cs4263.models.cnn import CNNClassifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(model, loader, criterion, optimizer, scheduler):
    print("\n\n\n############################################################")
    print("                    Training...")

    #set to training mode
    model.train()
    train_loss = 0.0
    running_loss_average = 0.0

    for i, (input_id, _, label) in tqdm.tqdm(enumerate(loader)):
        #send to gpu
        input_id,  label = input_id.to(DEVICE), label.to(DEVICE)

        #clear gradients
        optimizer.zero_grad()


        logits = model(input_id)

        #compute loss
        loss   = criterion(logits, label)

        #backprop
        loss.backward()

        #clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #move on
        optimizer.step()
        scheduler.step()

        #accumulate
        batch_sum = loss.item() * label.size(0)

        if i % 50 == 0 and i != 0:
            running_loss_average /= 50
            tqdm.tqdm.write(f"\t[INFO]: Training Loss = {running_loss_average:3f}")
            running_loss_average = 0.0
        else:
            running_loss_average += loss.item()

        train_loss += batch_sum

    print("\n############################################################")
    return train_loss / len(loader.dataset)

@torch.no_grad()
def valid_epoch(model, loader, criterion):
    print("\n\n\n############################################################")
    print("                    Validating...")

    model.eval()
    val_loss = 0
    all_logits, all_labels = [], []

    for input_id, _, label in tqdm.tqdm(loader):
        input_id, label = input_id.to(DEVICE), label.to(DEVICE)

        #get output
        logits = model(input_id)

        #compute loss
        loss =   criterion(logits, label)
        val_loss += loss.item() * label.size(0)

        #store dat stuff
        all_logits.append(logits)
        all_labels.append(label)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    print("\n############################################################")
    return all_logits, all_labels, val_loss / len(loader.dataset)

def load_data(filepath):
    data = torch.load(filepath)
    return data["input_ids"], data["attention_mask"], data["labels"]


def compute_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits).cpu()
    preds = (probs > threshold).int().numpy()
    labels = labels.cpu().numpy()
    f1 = f1_score(labels, preds, average="macro")
    return {"f1": f1}

def main(args):

    train_ids, train_masks, train_labels = load_data(os.path.join(args.data_dir, "train.pt"))
    valid_ids, valid_masks, valid_labels = load_data(os.path.join(args.data_dir, "validate.pt"))

    #Dataset + Dataloader
    train_dataset = TensorDataset(train_ids, train_masks, train_labels)
    validate_dataset = TensorDataset(valid_ids, valid_masks, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size * 2, shuffle=False)

    #define tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    #get model
    if args.model == 'bilstm':
        model = BiLSTMClassifier(vocab_size=len(tokenizer),
                                 pad_idx=tokenizer.pad_token_id
        ).to(DEVICE)
    elif args.model == 'cnn':
        model = CNNClassifier(vocab_size=len(tokenizer),
                              pad_idx=tokenizer.pad_token_id
        ).to(DEVICE)
    else:
        raise ValueError('Invalid input\n'
                         'bilstm | cnn')

    #compute pos_weight
    counts = train_labels.sum(0)
    neg = train_labels.size(0) - counts
    pos_weight = (neg / counts).to(torch.float)
    pos_weight = torch.clamp(pos_weight, max=10)

    #define optimizer + loss funcs
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    #define scheduler
    num_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(0.1 * num_steps),
        num_steps,
    )

    patience = 3
    no_improvement = 0
    best = float("-inf")
    for epoch in range(1, args.epochs + 1):
        #train epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        #validate epoch
        all_logits, all_labels, val_loss = valid_epoch(model, validate_loader, criterion)

        val_metrics = compute_metrics(all_logits, all_labels)

        if val_metrics['f1'] > best:
            #improvement found, reset
            no_improvement = 0

            #update
            best = val_metrics['f1']

            #save
            save_path = os.path.join(args.output_dir, f'{args.model}_best.pt')
            torch.save(model.state_dict(), save_path)

            #tell me wtf is going on
            print(f"[INFO] Saving checkpoint at {args.model}_best.pt")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"[INFO] Early Stopping Triggered after epoch {epoch} - no validation F1 improvement")
                break

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Validation F1: {val_metrics['f1']:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bilstm', 'cnn'], required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)

    main(parser.parse_args())
