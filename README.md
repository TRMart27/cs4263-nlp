# Sentiment Analysis in C

## Overview

his project implements a multi-model ensemble for multi-label toxicity detection on social media comments. While it began as an experimental C-based sentiment analysis exploration, the final deliverable focuses on a Python pipeline combining:

BERT: Fine‑tuned transformer for deep contextual embeddings

BiLSTM with Self-Attention: Captures sequential patterns and salient tokens

CNN: Extracts n‑gram features via convolution and pooling

  *TODO The ensemble (“teacher”) guides a future distilled model for real‑time, resource‑efficient inference without sacrificing performance.*

## Project Structure -- AS IT STANDS (work in progress!)

**c_headers/**
- **ht.h**  
  Implements a header-only hash table for storing string keys and counting token frequencies. Uses separate chaining to handle collisions.  
  

- **util.h**  
  Contains utility functions for logging and file handling. These functions provide basic support for debugging and error management.  


- **da.h**  
  Implements a dynamic array to store pairs of tokens, which can be used in the Byte Pair Encoding process. It includes a macro (`da_append`) to handle array growth.  \

- **tokenizer.h**  
  Provides a basic tokenizer that splits input strings into tokens using commas as delimiters. Currently, it prints tokens to stdout but can be extended to store tokens for further processing.

**src/**
- **bert.py**
    Defines a pretained "bert-base-uncased" model and a predict function for evaluating. Honestly I'm not entirely sure if I even use this or why I wrote this.
  
- **bilstm.py**
    Implements a BiLSTMClassifer - two layer bidirectional LSTM that includes a SelfAttention mechanism. Includes embedding layer and dropouts, and forward pass functionality.
    
- **cnn.py**
    Contains the CNNClassifer implementation. Uses multiple kernel sizes for n-gram extraction, an embedding layer, followed by max-pooling.
  
- **dataset.py**
    Handles the data preprocessing step. Reads a raw CSV file, cleans the text, generates adversarial swaps, and subwork tokenization from Hugging Face. Saves data to .pt files for later use.
  
- **fine_tune_bert.py**
    Script for BERT tuning. Loads data from .pt files, defines the optimizer, scheduler, and a loss function (accounts for class weighting to accomodate class imbalance present). Iterates over epochs, computes metrics, and stores the best model. Logs metrics for later plotting. 

- **train.py**
    Training pipeline for non-transformer models (bilstm, cnn). Loads data from .pt files, builds selected model, defines optimizer, scheduler, and runs training + validation epochs. Logs metrics for later plotting
  
- **ensemble.py**
    Combines the predictions from fine-tuned BERT, Bi-LSTM with attention, and CNN, applies the thresholds (hard-coded, I manually adjusted) to generate multi-label predictions, reports macro AND per-class F1 + ROC-AUC scores.


**review/**
- **tokenization.ipynb**
    My exploration of the BPE tokenization pipeline. This was implemented after I had realized I was biting time but was still too stubborn to give up doing things from scratch because that's where the fun programming is at. Had some functional version in python.
  
- **cs4263.ipynb**
    Documents my exploration at early NLP techniques. Tokenization approaches, static-lexicons, text cleaning, vocabularies, etc. 

## Prereqs
- **C**
    - GCC / clang supporting C99 (-std=c99) flag
    - make
- **Python**
  - 3.12
  - pip
  - PyTorch
  - HuggingFace Transformers
  - scikit-learn
  - NLTK
  ** NOTE CUDA was used for training purposes. If CUDA is available, scripts will use the GPU, otherwise it will default to cpu. CUDA 12.8 was used for the development process (I believe).

## Build Instructions 
- **Install Libraries**
    - pip install torch transformers scikit-learn nltk tqdm pandas numpy
       - *CUDA is option but highly recommended. Please ensure your CUDA version is compatabile with PyTorch version*
     
       - 
- **Ensure checkpoints and data**
    - models/bert_bert.pt (fine-tuned BERT)
    - models/bilstm_best.pt (BiLSTM)
    - models/cnn_best.pt    (CNN)
    - data_bin/validate.pt  (processed validation tensors)
       - *if data is not loaded run  - python dataset.py --csv <your_local_path/to/csv>*

- **Run Ensemble Script**
    - python ensemble.py
      - *if model weights are not available run*
        -*- python fine_tune_bert.py         !!ensure paths to --data_dir and --output_dir are aligned with expectations*
        -*- python train.py --model bilstm   !!ensure paths to --data_dir and --output_dir are aligned with expectations*
        -*- python train.py --model cnn      !!ensure paths to --data_dir and --output_dir are aligned with expectations*
        -*model weights will be saved in the provided --output_dir, defaults to models/*


- **Inspect Output**
    - Macro F1 + Per-Class F1 + ROC-AUC prints to **STDOUT**
    - Tensor of probabilites saved to
        - <models/run/ensemble_results.pt>
      

    
