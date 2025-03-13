# Sentiment Analysis in C

## Overview

This project is an experimental implementation of a Sentiment Analysis model written entirely in C. The goal is to build a naive Recurrent Neural Network (RNN) for sentiment analysis, with a focus on understanding low-level NLP techniques. As part of this project, a Byte Pair Encoding (BPE) routine is being implemented along with various supporting data structures.

The project serves as a learning tool to deepen understanding of:
- Natural Language Processing (NLP) techniques
- Data structures in C (e.g., hash tables, dynamic arrays)
- Basic RNN architectures for sentiment analysis

Once the C implementation is refined, the plan is to transition to Python for more sophisticated RNN networks.

## Project Structure -- AS IT STANDS (work in progress!)

- **ht.h**  
  Implements a header-only hash table for storing string keys and counting token frequencies. Uses separate chaining to handle collisions.  
  [See implementation details](&#)

- **util.h**  
  Contains utility functions for logging and file handling. These functions provide basic support for debugging and error management.  
  [See implementation details](&#)

- **da.h**  
  Implements a dynamic array to store pairs of tokens, which can be used in the Byte Pair Encoding process. It includes a macro (`da_append`) to handle array growth.  
  [See implementation details](&#)

- **tokenizer.h**  
  Provides a basic tokenizer that splits input strings into tokens using commas as delimiters. Currently, it prints tokens to stdout but can be extended to store tokens for further processing.  
  [See implementation details](&#)
