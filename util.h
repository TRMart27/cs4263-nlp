#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#define TRUE 1
#define FALSE 0

#define LOG(msg) do {\
    printf("[LOG]:\t%s\n", log_message);\
} while(0)


#define CHECK_MEMORY_FATAL(var, err_msg) do{\
    if(! (var) ) {\
        fprintf(stderr, "[ERROR]:\t%s\t", (err_msg) );\
        exit(EXIT_FAILURE);\
    }\
} while(0)


#define ERR_FATAL(err_msg) do {\
    fprintf(stderr, "[ERROR]:\t%s\t", (err_msg) );\
    exit(EXIT_FAILURE);\
} while(0)


#define ERR_RET_NULL(err_msg) do {\
    fprintf(stderr, "[ERROR]:\t%s\t", (err_msg) );\
    return NULL;\
} while(0)


#define ERR_RET_NONPOS(err_msg) do {\
    fprintf(stderr, "[ERROR]:\t%s\t", (err_msg) );\
    return -1;\
} while(0) 


#define ERR_RET_FALSE(err_msg) do {\
    fprintf(stderr, "[ERROR]:\t%s\t", (err_msg) );\
    return FALSE;\
} while(0) 


FILE* open_file(const char* filepath, const char* mode);
void  close_file(FILE* file_ref);

#ifdef UTIL_IMPL_


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
