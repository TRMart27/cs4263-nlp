#include "da.h"
#include "ht.h"
#include "tokenizer.h"

int main(void)
{
    FILE* data = open_file("data.csv", "r+");
#if 1
    HashTable ht = ht_alloc(17);
   
    HashEntry *w1 = ht_create_entry("I ", 1);
    HashEntry *w2 = ht_create_entry("love", 1);
    HashEntry *w3 = ht_create_entry(" my", 1);
    HashEntry *w4 = ht_create_entry("girlfriend", 1);
    HashEntry *w5 = ht_create_entry(" moriah", 1);
    HashEntry *w6 = ht_create_entry("mcgill ", 1);
    HashEntry *w7 = ht_create_entry(". ", 1);
    HashEntry *w8 = ht_create_entry(". ", 1);

    ht_enter(&ht, w1->key);
    ht_enter(&ht, w2->key);
    ht_enter(&ht, w3->key); 
    ht_enter(&ht, w4->key); 
    ht_enter(&ht, w5->key); 
    ht_enter(&ht, w6->key); 
    ht_enter(&ht, w7->key); 
    ht_enter(&ht, w8->key); 

    HashEntry* query = ht_get(&ht, w1->key);
    if(!query)
        return -1;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);
    query = ht_get(&ht, w2->key);
    if(!query)
        return -2;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);
    query = ht_get(&ht, w3->key);
    if(!query)
        return -3;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);
    query = ht_get(&ht, w4->key);
    if(!query)
        return -4;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);
    query = ht_get(&ht, w5->key);
    if(!query)
        return -5;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);
    query = ht_get(&ht, w6->key);
    if(!query)
        return -6;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);
    query = ht_get(&ht, w7->key);
    if(!query)
        return -7;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);
    query = ht_get(&ht, w8->key);
    if(!query)
        return -7;
    printf("{ Key = <%s>, Value = <%zu> }\n", (char*)query->key, query->value);


    ht_delete(&ht, w1->key);
    ht_delete(&ht, w2->key);
    ht_delete(&ht, w3->key);
    ht_delete(&ht, w4->key);
    ht_delete(&ht, w5->key);
    ht_delete(&ht, w6->key);
    ht_delete(&ht, w7->key);
    ht_delete(&ht, w8->key);

    if(! ht_get(&ht, w1->key) )
        printf("THE WORKED BRO WHAT THE FUCK\n");
    if(! ht_get(&ht, w2->key) )
        printf("THE WORKED TWICE BRO WHAT THE FUCK\n");
    if(! ht_get(&ht, w3->key) )
        printf("THE THRICE TIMES I THINK ITS WORKING BRO WOWWW\n");



    ht_free_entry(w1);
    ht_free_entry(w2);
    ht_free_entry(w3);
    ht_free_entry(w4);
    ht_free_entry(w5);
    ht_free_entry(w6);
    ht_free_entry(w7);
#endif

#if 0
    DA  new_arr = da_init();
    
    Token a = {.t = 'h'};
    Token b = {.t = 'i'};

    Pair *pairs = malloc(sizeof(Pair) * 10);
    pairs[0].left  = a;
    pairs[0].right = b;
 
    da_append(&new_arr, pairs[0]); 

    da_print(&new_arr);
#endif
    close_file(data);
    return 0;
}

