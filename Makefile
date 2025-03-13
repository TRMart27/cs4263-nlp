CC = clang
CFLAGS = -I include/ -Wall -Wextra 
SRC = main.c src/util.c src/tokenizer.c src/ht.c src/da.c
OBJ = $(SRC:.c=.o)
TARGET = target

all: $(TARGET)                          

#the TARGET needs all the object files, which are in OBJ
        # $(TARGET) DEPENDS ON (:) $(OBJ)
        # @ means "target name"
        # ^ means prereqs (the .o files)
$(TARGET): $(OBJ)
		$(CC) -o $@ $^


#for compiling (.c) files to object (.o) files 
        # (object DEPENDS ON (:) needs .c)
        
%.o: %.c
		$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
