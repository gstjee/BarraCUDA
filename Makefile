CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -pedantic -Werror -O2 \
          -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
          -Wformat=2 -Wundef -Wcast-align -Wnull-dereference \
          -Wstack-usage=4096 -Wno-error=stack-usage= \
          -Wconversion -Wold-style-definition \
          -Wdouble-promotion -Wswitch-enum -Wredundant-decls -Wwrite-strings
SRCDIR  = src
SOURCES = $(SRCDIR)/main.c $(SRCDIR)/preproc.c $(SRCDIR)/lexer.c $(SRCDIR)/parser.c $(SRCDIR)/sema.c $(SRCDIR)/bir.c $(SRCDIR)/bir_print.c $(SRCDIR)/bir_lower.c $(SRCDIR)/bir_mem2reg.c $(SRCDIR)/amdgpu_isel.c $(SRCDIR)/amdgpu_emit.c
OBJECTS = $(SOURCES:.c=.o)
TARGET  = barracuda

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRCDIR)/*.o $(TARGET) $(TARGET).exe

.PHONY: all clean
