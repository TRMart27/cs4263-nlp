#ifndef HT_H_
#define HT_H_

// -- include statements / dependencies 
#include "util.h"

// -- macro definitions
#define INIT_HASH_CAP 16
#define LOAD_FACTOR   .69 //hehe
#define HT_GROWTH_FACTOR 1.69 //he

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
void ht_enter(HashTable* ht, const char* request_entry, int COUNT_FLAG); //TODO make request a void* pointer for a more dynamic implementation, but im too lazy and want to work on my actual project idk why i hate myself tbh
void ht_delete(HashTable* ht, const char* key);
HashEntry *ht_get(HashTable *ht, const char* key);
void ht_keys(HashTable* ht);


// -- hashing functions
unsigned long ht_hash_string(const char* str);

#ifdef HT_IMPL_

// -- memory allocation and deallocation **HashTable
HashTable ht_alloc(size_t num_buckets)
{
    HashTable ht = {
        .num_buckets = num_buckets,
        .size        = 0,
        .buckets     = calloc(num_buckets, sizeof(HashEntry))
    };
    
    CHECK_MEMORY_FATAL(ht.buckets, "Failed to allocate buckets in <ht_alloc>!");    

    return ht;
}


void ht_free(HashTable* ht)
{
    assert(ht);

    for(size_t i = 0; i < ht->num_buckets; ++i)
        free(ht->buckets[i]);

    ht->num_buckets = 0;
    ht->size        = 0;
    free(ht->buckets);
//    free(ht);

    return;
}

// -- memory allocation and deallocation **Entry
HashEntry *ht_create_entry(const char* key, size_t value)
{
    HashEntry* e = malloc(sizeof(HashEntry));
    CHECK_MEMORY_FATAL(e, "Failed to allocate HashEntry e in <ht_create_entry>!");

    e->value = value;
    e->next  = NULL;
    e->key   = strdup(key); //malloc(sizeof(*e->key));

    CHECK_MEMORY_FATAL(e->key, "Failed to duplicate entry key in <ht_create_entry>!");

    return e;
}


void ht_free_entry(HashEntry* entry)
{
    assert(entry);
    
    free(entry->key);
    free(entry);
    entry->value = 1; 
    entry->next = NULL;

    return;
}

//CRUD for HashTable
void ht_enter(HashTable* ht, const char* request_entry, int COUNT_FLAG)
{
    assert(ht);
    assert(request_entry);
    size_t hashed_index  = ht_hash_string(request_entry) % ht->num_buckets;

    HashEntry *curr_entry = ht->buckets[hashed_index];
    while(curr_entry)
    {
        if(strcmp( (const char*)curr_entry->key, request_entry) == 0) //TODO get rid of hard coded casting, update to memcmp (i think)
        {
            if(COUNT_FLAG > 0)
                curr_entry->value++; 
            else 
                return;
            return;
        }
        curr_entry = curr_entry->next;
    }

    //if we made it this far, the key doesn't exist so lets make one and add it
    HashEntry* new_entry = NULL;
    if(COUNT_FLAG)
        new_entry = ht_create_entry(request_entry, 1);
    else
       new_entry = ht_create_entry(request_entry, ht->size);
    if(ht->buckets[hashed_index]) 
        new_entry->next = ht->buckets[hashed_index]; //set next to old head

    ht->buckets[hashed_index] = new_entry; //update the head to the new entry
    ht->size++; //increase size of the table
#if 1
    if( (float)(ht->size / ht->num_buckets) > LOAD_FACTOR)
    {
        size_t new_size = ht->num_buckets * HT_GROWTH_FACTOR;
        HashTable resized_table = ht_alloc(new_size);
       
        for(size_t i = 0; i < ht->num_buckets - 1; ++i)
        {
            if(!ht->buckets[i]) //if the bucket has nothing in it
                continue; //skip 

            HashEntry *curr = ht->buckets[i]; //otherwise iterate through the list
            while(curr)
            {
                size_t new_index = ht_hash_string(curr->key) % new_size; //re hash each element
                resized_table.buckets[new_index] = curr;  //add the new element in the revised table 
                curr = curr->next; //move onto next 
            }
        } 
        *ht = resized_table;    
    }
#endif
    return;
}


void ht_delete(HashTable* ht, const char* key)
{
    assert(ht);
    assert(key);
    printf("key => %s\n", key);
    size_t hashed_index  = ht_hash_string(key) % ht->num_buckets;
    printf("Hashed_Index => %zu\n", hashed_index);
    HashEntry *curr = ht->buckets[hashed_index];
    HashEntry *prev = NULL;
    
    while(curr)
    {
        if(strcmp( (const char*)curr->key, key) == 0)
        {
            printf("made it ehre\n");
            if(prev)
                prev->next = curr->next;
            else
                ht->buckets[hashed_index] = curr->next;
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



void ht_keys(HashTable* ht)
{
    for(size_t i = 0; i < ht->num_buckets; ++i)
    {
        HashEntry *curr_entry = ht->buckets[i];
        while(curr_entry)
        {
            printf("{ Key => <%s> | Value => <%zu> }\n", (const char*)curr_entry->key, curr_entry->value);
            curr_entry = curr_entry->next;
        }
//        printf("}\n");
    }
    return;


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
