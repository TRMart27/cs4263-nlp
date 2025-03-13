#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include "util.h"
#include "da.h"

// structure defintions
typedef struct Tokens 
{
   size_t capacity;
   size_t size;
   Token* items;
} Tokens;


//function prototypes
void tokenize(Tokens *tokens_arr, char* stream);


#ifdef TOKENIZER_IMPL_


void tokenize(Tokens *tokens_arr, char* stream)
{
    assert(stream);    
    assert(tokens_arr);    
    
    char *token = strtok(stream, ",");
    while(token)
    {
        printf("%s\n", token);
        token = strtok(NULL, ",");
    }
}



#endif //TOKENIZER_IMPL_
#endif //TOKENIZER_H_
