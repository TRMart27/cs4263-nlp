import argparse
import os
import torch
import string

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

#ensure we got NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

#define global variables
STOP_WORDS = set(stopwords.words('english'))
TARGET_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


#cleaning procedures
def normalize_encoding(text):
    return text.encode('utf-8', errors='ignore').decode()

def clean_text(text):
    text = normalize_encoding(text=text)
        #remove functionalities - aligns with base transformer training procedure
        #text = remove_stopwords(text=text)
        #text = remove_punctuation(text=text)
    return text.lower()

#get the synonym swap attacks
def get_synonym(word):
    #gather data for <word>
    synsets = wordnet.synsets(word)

    #collect ALL lemmas
    #   .replace => wordnet left me with "_" separations instead of " " separations
    lemmas = {l.name().replace("_", " ") for s in sysnets for l in s.lemmas()}

    #dont swap with original word, toss it
    lemmas.discard(word)

    #confirm nothing went wrong
    if lemmas:
        choice = random.choice(list(lemmas))
        return choice
    else:
        return None

def synonym_swap(text, total_swaps=1):
    tokens = word_tokenize(text)

    #get the indexes of the valid swap candidates
    ids = []
    for idx, word in enumerate(tokens):
        if word.isalpha() and word not in STOP_WORDS:
            idx.append(idx)

    #shuffle so there is "order"
    random.shuffle(ids)

    num_swaps = 0
    for i in ids:
        #get the random swap
        synonym = get_synonym(tokens[i])

        #ensure we got something back
        if synonym:
            #update it + count it
            tokens[i] = synonym
            num_swaps += 1
        #check if we desired number of swaps, exit loop
        if num_swaps == total_swaps:
            break
    #re-stringify the tokens and return - adversarial training complete
    res = " ".join(tokens)
    return res

#tokenization wrappers
def wp_encode(texts, tokenizer, max_len):
    encoded_texts = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return encoded_texts["input_ids"], encoded_texts["attention_mask"]

def char_tokenize(text, max_len=300):
    #convert characters (up to :max_len) into id
    #   - 0 = padding
    #   - 1-256 = byte+1

    #only consider up to max_len
    view = text[:max_len]

    #convert to byte value (256 max)
    ids = [min(ord(c), 255) + 1 for c in view]

    #1 = character
    mask = [1] * len(ids)

    #pad for consistent length
    ids += [0] * (max_len - len(ids))
    mask += [0] * (max_len - len(mask))

    return ids, mask


def main(args):
    #load data
    df = pd.read_csv(args.csv).dropna(subset=['comment_text'])
    df['comment_text'] = df['comment_text'].map(clean_text)

    #define inputs as "X"
    X = df['comment_text'].tolist()
    #define outputs/targets as "y"
    y = df[TARGET_LABELS].values.astype("float32")

    #split training data into test/validation sets
    X_train, X_validate, y_train, y_validate = train_test_split(
        X, y,
        test_size=args.val_split,
        random_state=42,
        shuffle=True
    )

    #generate the synonym swap examples
    adv_texts = []
    adv_labels = []
    for text, label in zip(X_train, y_train):

    #tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ids, train_mask       = encode(X_train, tokenizer, args.max_len)
    validate_ids, validate_mask = encode(X_validate, tokenizer, args.max_len)

    #store the tensors for later use
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {"input_ids"     : train_ids,
        "attention_mask" : train_mask,
        "labels"  : torch.tensor(y_train)},
        os.path.join(args.output_dir, "train.pt")
    )
    torch.save(
        {"input_ids"     : validate_ids,
        "attention_mask" : validate_mask,
        "labels"  : torch.tensor(y_validate)},
        os.path.join(args.output_dir, "validate.pt")
    )
    print(f"Saved files: \n{args.output_dir}/train.pt\n{args.output_dir}/validate.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True, help="Path to the train.csv")
    parser.add_argument("--output_dir", type=str, default="data_bin", help="Directory to store data")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.21)
    args = parser.parse_args()

    main(args)

