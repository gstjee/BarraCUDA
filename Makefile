CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -pedantic -Werror -O2 \
          -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
          -Wformat=2 -Wundef -Wcast-align -Wnull-dereference \
          -Wstack-usage=4096 -Wno-error=stack-usage= \
          -Wconversion -Wold-style-definition \
          -Wdouble-promotion -Wswitch-enum -Wredundant-decls -Wwrite-strings \
          -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE -fcf-protection \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/amdgpu
LDFLAGS = -pie
# Linux/ELF only: -Wl,-z,relro,-z,now -Wl,-z,noexecstack

SOURCES = src/main.c \
          src/fe/preproc.c src/fe/lexer.c src/fe/parser.c src/fe/sema.c \
          src/ir/bir.c src/ir/bir_print.c src/ir/bir_lower.c src/ir/bir_mem2reg.c \
          src/amdgpu/isel.c src/amdgpu/emit.c src/amdgpu/encode.c src/amdgpu/enc_tab.c
OBJECTS = $(SOURCES:.c=.o)
TARGET  = barracuda

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Test Suite ----
TCFLAGS = -std=c99 -Wall -Wextra -O0 -g -Isrc -Isrc/fe -Isrc/ir -Isrc/amdgpu
TSRC    = tests/tmain.c tests/tsmoke.c tests/tcomp.c tests/tenc.c \
          tests/ttabs.c tests/ttypes.c tests/terrs.c tests/tphase.c
TOBJS   = $(TSRC:.c=.o)
COBJS   = src/ir/bir.o src/ir/bir_print.o src/ir/bir_lower.o src/ir/bir_mem2reg.o \
          src/amdgpu/encode.o src/amdgpu/enc_tab.o src/amdgpu/isel.o src/amdgpu/emit.o \
          src/fe/lexer.o src/fe/parser.o src/fe/preproc.o src/fe/sema.o

test: $(TARGET) trunner
	./trunner --all

trunner: $(TOBJS) $(COBJS)
	$(CC) $(TCFLAGS) -o $@ $^

tests/%.o: tests/%.c
	$(CC) $(TCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(TARGET).exe trunner trunner.exe $(TOBJS)

.PHONY: all clean test
