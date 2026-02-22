CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -pedantic -Werror -O2 \
          -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
          -Wformat=2 -Wundef -Wcast-align -Wnull-dereference \
          -Wstack-usage=4096 -Wno-error=stack-usage= \
          -Wconversion -Wold-style-definition \
          -Wdouble-promotion -Wswitch-enum -Wredundant-decls -Wwrite-strings \
          -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE -fcf-protection
LDFLAGS = -pie
# Linux/ELF only: -Wl,-z,relro,-z,now -Wl,-z,noexecstack
SRCDIR  = src
SOURCES = $(SRCDIR)/main.c $(SRCDIR)/preproc.c $(SRCDIR)/lexer.c $(SRCDIR)/parser.c $(SRCDIR)/sema.c $(SRCDIR)/bir.c $(SRCDIR)/bir_print.c $(SRCDIR)/bir_lower.c $(SRCDIR)/bir_mem2reg.c $(SRCDIR)/amdgpu_isel.c $(SRCDIR)/amdgpu_emit.c $(SRCDIR)/tensix_isel.c $(SRCDIR)/tensix_emit.c $(SRCDIR)/tensix_coarsen.c $(SRCDIR)/tensix_datamov.c
OBJECTS = $(SOURCES:.c=.o)
TARGET  = barracuda

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRCDIR)/*.o $(TARGET) $(TARGET).exe

.PHONY: all clean
