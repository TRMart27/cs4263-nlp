#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define TRUE 1
#define FALSE 0
#define LOG(x) my_log( (x) );

void my_log(char* log_message);
FILE* open_file(const char* filepath, const char* mode);
void  close_file(FILE* file_ref);

#ifdef UTIL_IMPL_


void my_log(char* log_message)
{
    printf("[LOG]\t%s\n", log_message);

    return;
}

FILE* open_file(const char *filepath, const char* mode)
{
    assert(filepath);

    FILE* file = fopen(filepath, mode);
    if(!file)
    { 
        perror("[ERROR]\tfailed to open file!\n"); 
        return NULL;
    }

    return file;
}

void close_file(FILE* file_ref)
{
    assert(file_ref);
    if(fclose(file_ref) != 0)
        perror("[ERROR]\tFailed to close file!\n");
    return;
}
#endif //UTIL_IMPL_
#endif //UTIL_H_
