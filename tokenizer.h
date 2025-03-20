#ifndef TOKENIZER_H_
#define TOKENIZER_H_

// -- dependencies
#include "util.h"
#include "da.h"
#include "ht.h"

// -- macro definitions
#define TOKEN(token) (token).t

// structure defintions
typedef struct Tokens 
{
   size_t capacity;
   size_t size;
   Token* items;
} Tokens;

typedef struct Pairs
{
    size_t capacity;
    size_t size;
    Pair*  items;
} Pairs;


//function prototypes
char* bpe_generate_key(Token t);
void bpe_clean(char* stream);
void bpe_tokenize(Tokens *tokens_arr, char* stream);
void bpe_get_freqs(HashTable *freq_table, Tokens* tokens_arr);
HashEntry *bpe_get_max(HashTable* freq_table);
void bpe_encode(Tokens* tokens_arr, HashTable* freq_table, HashTable* vocab);
void bpe_render_tokens(Tokens* tokens_arr);

#ifdef TOKENIZER_IMPL_


char* bpe_generate_key(Token t)
{
    char *key = malloc(sizeof(2));
    key[0] = TOKEN(t); 
    key[1] = '\0';
    return key;

}
void bpe_clean(char* stream)
{
    assert(stream);
    size_t len = strlen(stream);
    for(size_t i = 0; i < len; ++i)
    { 
        if(isupper(stream[i]))
                stream[i] = tolower(stream[i]);
        else if(stream[i] == ' ') 
            stream[i] = '_';
        else if(stream[i] == '\n')
            stream[i] = 'N';
    }
    printf("%s\n", stream);
    return;
}


void bpe_tokenize(Tokens *tokens_arr, char* stream)
{
    assert(stream);    
    assert(tokens_arr);    
    assert(tokens_arr->items);


    size_t stream_len = strlen(stream);
    for(size_t i = 0; i < stream_len; i+=2)
    {
        if(stream_len % 2 != 0 && i == stream_len - 1)
        {
            Token odd_one_out = { .t = stream[i] };
            da_append(tokens_arr, odd_one_out);
            break;
        }
        Token t1 = { .t = stream[i] };
        Token t2 = { .t = stream[i + 1] };
 
        da_append(tokens_arr, t1); 
        da_append(tokens_arr, t2);
    }
}


void bpe_get_freqs(HashTable *freq_table, Tokens* tokens_arr)
{
    assert(freq_table);
    assert(tokens_arr);

    size_t len = tokens_arr->size;
    for(size_t i = 0; i < len - 1; i+=2)
    { 
        char pair_key[3];
        pair_key[0] = TOKEN(tokens_arr->items[i]);
        pair_key[1] = TOKEN(tokens_arr->items[i + 1]);
        pair_key[2] = '\0';   
        ht_enter(freq_table, pair_key, 1); 
    }
}

HashEntry *bpe_get_max(HashTable* freq_table)
{
    assert(freq_table);

    size_t len = freq_table->num_buckets;
    HashEntry* max = NULL;

    for(size_t i = 0; i < len; ++i)
    {
        HashEntry *curr = freq_table->buckets[i];
        //if the bucket is empty 
        if(!curr) {
            continue; 
        }

        //otherwise iterate through the chain
        while(curr)
        {
            //if max entry hasnt been set or the current count is higher than the currently stored max count
            if(!max || curr->value > max->value)                 
                max = curr;
            curr = curr->next;
        }
    }
    return max;
}


void bpe_encode(Tokens *tokens_arr, HashTable* freq_table, HashTable* vocab)
{
    assert(tokens_arr);
    assert(freq_table);
    assert(vocab);
   for(size_t i = 0; i < 2; ++i)
    {
        HashEntry* most_freq = bpe_get_max(freq_table);
        ht_delete(freq_table, (const char*)most_freq->key);
        ht_enter(vocab, most_freq->key, 0);
    }
    return;
}

void bpe_render_tokens(Tokens* tokens_arr)
{
    assert(tokens_arr);


    size_t len = tokens_arr->size - 1;
    for(size_t i = 0; i < len; ++i)
        printf("%c", TOKEN(tokens_arr->items[i]));

    return;
}
#endif //TOKENIZER_IMPL_
#endif //TOKENIZER_H_
