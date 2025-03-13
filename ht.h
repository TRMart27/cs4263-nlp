#ifndef HT_H_
#define HT_H_

// -- include statements / dependencies 
#include "util.h"

// -- macro definitions
#define INIT_HASH_CAP 16
#define LOAD_FACTOR   .69 //hehe
#define HT_GROWTH_FACTOR 1.69 //heh

typedef struct HashEntry 
{
    void*  key;
    size_t value; //TODO make void* pointer, let user handle casting for a dynamic implementation
    struct HashEntry* next;
} HashEntry;

typedef struct HashTable
{
    HashEntry* *buckets; // array of pointers to hash entries    
    size_t num_buckets;
    size_t size;
} HashTable;


// -- memory allocation and deallocation **HashTable
HashTable ht_alloc(size_t num_buckets);
void ht_free(HashTable* ht);


// -- memory allocation and deallocation **Entry
HashEntry *ht_create_entry(const char* key, size_t value);
void ht_free_entry(HashEntry* entry);


//CRUD for HashTable
void ht_enter(HashTable* ht, const char* request_key);
void ht_delete(HashTable* ht, const char* key);
HashEntry *ht_get(HashTable *ht, const char* key);


// -- hashing functions
unsigned long ht_hash_string(const char* str);

#ifdef HT_IMPL_

// -- memory allocation and deallocation **HashTable
HashTable ht_alloc(size_t num_buckets)
{
    HashTable ht = {
        .num_buckets = num_buckets,
        .size        = 0,
        .buckets     = calloc(num_buckets, sizeof(HashEntry*))
    };
    
    if(!ht.buckets)
    {
        perror("[ERROR]\tFailed to allocate buckets!\n");
        exit(EXIT_FAILURE);
    }
    
    return ht;
}
void ht_free(HashTable* ht)
{
    assert(ht);

    for(size_t i = 0; i < ht->num_buckets; ++i)
    {
        HashEntry *curr_entry = ht->buckets[i];
        while(curr_entry) 
        {
            HashEntry *next = curr_entry->next;
            free(curr_entry->key);
            free(curr_entry);
            curr_entry = next;
        }
    }
    ht->num_buckets = 0;
    ht->size        = 0;
    free(ht->buckets);
    free(ht);

    return;
}

// -- memory allocation and deallocation **Entry
HashEntry *ht_create_entry(const char* key, size_t value)
{
    HashEntry *e = malloc(sizeof(HashEntry));
    if(!e)
    {
        perror("[ERROR] Failed to allocate HashEntry in <ht_create_entry>!\n");
        exit(EXIT_FAILURE);
    }
    
    e->value = value;
    e->next  = NULL;
    e->key   = strdup(key);
    
    if(!e->key)
    {
        perror("[ERROR]\tFailed to copy key in <ht_create_entry>!\n");
        free(e);
        exit(EXIT_FAILURE);
    }

    return e;
}


void ht_free_entry(HashEntry* entry)
{
    assert(entry);
    
    free(entry->key);
    entry->value = 0;
    entry->next = NULL;

    return;
}

//CRUD for HashTable
void ht_enter(HashTable* ht, const char *request_key)
{
    assert(ht);
    assert(request_key);

    size_t hashed_index  = ht_hash_string(request_key) % ht->num_buckets;

    //check if the entry exists
    HashEntry *curr_entry = ht->buckets[hashed_index];
    while(curr_entry)
    {
        if(strcmp(curr_entry->key, request_key) == 0)
        {
            curr_entry->value++; 
            return;
        }
        curr_entry = curr_entry->next;
    }

    //if we did not find it
    HashEntry *new_entry = ht_create_entry(request_key, 1);
    new_entry->next = ht->buckets[hashed_index];

    ht->buckets[hashed_index] = new_entry;
    ht->size++;

    if( (float)(ht->size / ht->num_buckets) > LOAD_FACTOR)
    {
        HashTable resized_table = ht_alloc(ht->size * HT_GROWTH_FACTOR);
             

    }
    return;
}
void ht_delete(HashTable* ht, const char* key)
{
    assert(ht);
    assert(key);

    size_t hashed_index  = ht_hash_string(key) % ht->num_buckets;

    HashEntry *curr = ht->buckets[hashed_index];
    HashEntry *prev = NULL;
    
    while(curr)
    {
        if(strcmp(curr->key, key) == 0)
        {
            //first try?????
            if(prev)
                prev->next = curr->next;
            else
                ht->buckets[hashed_index] = curr->next;
            free(curr->key);
            free(curr);
            ht->size--;

            return;
        }
        //not it, continue forward! 
        prev = curr;
        curr = curr->next;
    }
}
HashEntry *ht_get(HashTable *ht, const char* key)
{
    assert(ht);
    assert(key);

    unsigned long hashed = ht_hash_string(key);
    size_t hashed_index  = hashed % ht->num_buckets;

    HashEntry *curr = ht->buckets[hashed_index];
    while(curr)
    {
        if(strcmp(curr->key, key) == 0)
            return curr;
        curr = curr->next; 
    }
    return NULL;
}



// -- hashing functions
//      --djb2 hash function for strings 
unsigned long ht_hash_string(const char* str)
{
    unsigned long hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; //hash * 33 + c
    return hash;
}

#endif //HT_IMPL_
#endif //HT_H_
