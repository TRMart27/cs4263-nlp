#ifndef DA_H_
#define DA_H_

//include statements
#include "util.h"

//macro definitions
#define INIT_DA_CAP 10 
#define GROWTH_FACTOR 2

#define da_append(arr, item) do\
    {\
        assert( (arr) );\
        if( (arr)->size == (arr)->capacity )\
        {\
            (arr)->capacity *= GROWTH_FACTOR;\
            (arr)->items     = realloc( (arr)->items, sizeof((arr)->items[0]) * (arr)->capacity);\
            if(!(arr)->items)\
            {\
                perror("[ERROR]\tFailed to reallocate items array in <da_append>\n");\
                exit(EXIT_FAILURE);\
            }\
        }\
        \
        (arr)->items[ (arr)->size ] = item;\
        (arr)->size+=1;\
    } while(0);

//structure defintions
typedef struct Token
{
   char t;
} Token;


typedef struct Pair 
{
    Token left, right;
} Pair;


typedef struct DA
{
    size_t  size;
    size_t  capacity;
    Pair   *items;
} DA;


//function prototypes
DA da_init();
void da_print(DA *arr);
void da_free(DA* arr);


#ifdef DA_IMPL_

DA da_init()
{ 
    DA arr = {
        .size     = 0,
        .capacity = INIT_DA_CAP,
        .items    = malloc(sizeof(*arr.items) * INIT_DA_CAP)
    };

    printf("sizeof(*arr.items) => %zu\n", sizeof(*arr.items));
    //malloc can -> FAIL <- 
    if(!arr.items)
    {
        perror("[ERROR]\tItems failed to be allocated in <da_init>\n");
        exit(EXIT_FAILURE);
    }

return arr;
}


void da_print(DA *arr)
{
    assert(arr);

    for(size_t i = 0; i < arr->size; ++i)
        printf("(%c , %c)\n", arr->items[i].left.t, arr->items[i].right.t);

    return;
}
void da_free(DA* arr)
{
    assert(arr);
    free(arr);
    return;
}



#endif //DA_IMPL_
#endif //DA_H_
