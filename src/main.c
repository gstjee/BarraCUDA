#include "preproc.h"
#include "lexer.h"
#include "parser.h"
#include "sema.h"
#include "bir_lower.h"
#include "bir_mem2reg.h"
#include "amdgpu.h"
#include <stdlib.h>

static char       source_buf[BC_MAX_SOURCE];
static char       pp_out_buf[BC_MAX_SOURCE];  /* preprocessor output */
static token_t    token_buf[BC_MAX_TOKENS];
static ast_node_t node_buf[BC_MAX_NODES];
static bir_module_t *bir_module; /* heap-allocated (~11 MB) */

static int read_file(const char *path, char *buf, uint32_t max, uint32_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s'\n", path);
        return BC_ERR_IO;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0 || (uint32_t)sz >= max) {
        fprintf(stderr, "error: file too large (%ld bytes, max %u)\n",
                sz, max);
        fclose(fp);
        return BC_ERR_IO;
    }
    *out_len = (uint32_t)fread(buf, 1, (size_t)sz, fp);
    buf[*out_len] = '\0';
    fclose(fp);
    return BC_OK;
}

static void dump_tokens(const lexer_t *L)
{
    char text[256];
    for (uint32_t i = 0; i < L->num_tokens; i++) {
        const token_t *t = &L->tokens[i];
        lexer_token_text(L, t, text, sizeof(text));

        printf("%4u:%-3u  %-20s  %s\n",
               t->line, t->col,
               token_type_name(t->type),
               text);
    }
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "BarraCUDA - CUDA Compiler\n"
        "Usage: %s [options] <file.cu>\n"
        "\n"
        "Options:\n"
        "  --lex         Tokenize and dump token stream\n"
        "  --parse       Parse and dump AST\n"
        "  --ir          Lower to BIR and print IR\n"
        "  --no-mem2reg  Skip mem2reg optimization pass\n"
        "  --sema        Run semantic analysis and dump types\n"
        "  --pp          Preprocess only and print result\n"
        "  --no-pp       Skip preprocessor\n"
        "  -I <dir>      Add include search path\n"
        "  -D <name[=val]> Define a preprocessor macro\n"
        "  --amdgpu      Compile to AMDGCN assembly (gfx1100)\n"
        "  --amdgpu-bin  Compile to AMDGPU ELF code object (.hsaco)\n"
        "  -o <file>     Output file (for --amdgpu-bin)\n"
        "  --help        Show this message\n"
        "\n", prog);
}

int main(int argc, char *argv[])
{
    const char *file = NULL;
    const char *output_file = NULL;
    int mode_pp = 0;
    int mode_lex = 0;
    int mode_parse = 0;
    int mode_sema = 0;
    int mode_ir = 0;
    int mode_amdgpu = 0;
    int mode_amdgpu_bin = 0;
    int no_mem2reg = 0;
    int no_pp = 0;

    /* Collect -I and -D options for preprocessor */
    const char *include_paths[PP_MAX_INCLUDE_PATHS];
    int num_include_paths = 0;
    const char *defines[128];
    int num_defines = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lex") == 0)
            mode_lex = 1;
        else if (strcmp(argv[i], "--parse") == 0)
            mode_parse = 1;
        else if (strcmp(argv[i], "--sema") == 0)
            mode_sema = 1;
        else if (strcmp(argv[i], "--ir") == 0)
            mode_ir = 1;
        else if (strcmp(argv[i], "--pp") == 0)
            mode_pp = 1;
        else if (strcmp(argv[i], "--no-pp") == 0)
            no_pp = 1;
        else if (strcmp(argv[i], "--amdgpu") == 0)
            mode_amdgpu = 1;
        else if (strcmp(argv[i], "--amdgpu-bin") == 0)
            mode_amdgpu_bin = 1;
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            if (num_include_paths < PP_MAX_INCLUDE_PATHS)
                include_paths[num_include_paths++] = argv[++i];
        } else if (strncmp(argv[i], "-I", 2) == 0 && argv[i][2]) {
            if (num_include_paths < PP_MAX_INCLUDE_PATHS)
                include_paths[num_include_paths++] = argv[i] + 2;
        } else if (strcmp(argv[i], "-D") == 0 && i + 1 < argc) {
            if (num_defines < 128)
                defines[num_defines++] = argv[++i];
        } else if (strncmp(argv[i], "-D", 2) == 0 && argv[i][2]) {
            if (num_defines < 128)
                defines[num_defines++] = argv[i] + 2;
        } else if (strcmp(argv[i], "--no-mem2reg") == 0)
            no_mem2reg = 1;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-')
            file = argv[i];
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!file) {
        usage(argv[0]);
        return 1;
    }

    if (!mode_pp && !mode_lex && !mode_parse && !mode_sema && !mode_ir &&
        !mode_amdgpu && !mode_amdgpu_bin)
        mode_parse = 1;

    uint32_t src_len = 0;
    if (read_file(file, source_buf, BC_MAX_SOURCE, &src_len) != BC_OK)
        return 1;

    /* Preprocessing */
    const char *lex_src = source_buf;
    uint32_t    lex_len = src_len;

    if (!no_pp) {
        preproc_t *pp = (preproc_t *)malloc(sizeof(preproc_t));
        if (!pp) {
            fprintf(stderr, "error: failed to allocate preprocessor\n");
            return 1;
        }
        pp_init(pp, source_buf, src_len, pp_out_buf, BC_MAX_SOURCE, file);

        for (int i = 0; i < num_include_paths; i++)
            pp_add_include_path(pp, include_paths[i]);
        for (int i = 0; i < num_defines; i++) {
            char dname[BC_MAX_IDENT];
            const char *eq = strchr(defines[i], '=');
            if (eq) {
                uint32_t nlen = (uint32_t)(eq - defines[i]);
                if (nlen >= BC_MAX_IDENT) nlen = BC_MAX_IDENT - 1;
                memcpy(dname, defines[i], nlen);
                dname[nlen] = '\0';
                pp_define(pp, dname, eq + 1);
            } else {
                pp_define(pp, defines[i], "1");
            }
        }

        int prc = pp_process(pp);

        if (pp->num_errors > 0) {
            for (int i = 0; i < pp->num_errors; i++) {
                fprintf(stderr, "%s:%u: preproc error: %s\n",
                        file, pp->errors[i].loc.line, pp->errors[i].msg);
            }
        }

        if (mode_pp) {
            fwrite(pp_out_buf, 1, pp->out_len, stdout);
            free(pp);
            return prc != BC_OK ? 1 : 0;
        }

        lex_src = pp_out_buf;
        lex_len = pp->out_len;
        free(pp);
    }

    lexer_t L;
    lexer_init(&L, lex_src, lex_len, token_buf, BC_MAX_TOKENS);
    int rc = lexer_tokenize(&L);

    if (L.num_errors > 0) {
        for (int i = 0; i < L.num_errors; i++) {
            fprintf(stderr, "%s:%u:%u: error: %s\n",
                    file, L.errors[i].loc.line, L.errors[i].loc.col,
                    L.errors[i].msg);
        }
    }

    if (mode_lex) {
        dump_tokens(&L);
        printf("\n%u tokens, %d error(s)\n", L.num_tokens, L.num_errors);
    }

    if (mode_parse || mode_sema || mode_ir || mode_amdgpu || mode_amdgpu_bin) {
        parser_t P;
        parser_init(&P, token_buf, L.num_tokens, lex_src,
                    node_buf, BC_MAX_NODES);
        uint32_t root = parser_parse(&P);

        if (P.num_errors > 0) {
            for (int i = 0; i < P.num_errors; i++) {
                fprintf(stderr, "%s:%u:%u: parse error: %s\n",
                        file, P.errors[i].loc.line, P.errors[i].loc.col,
                        P.errors[i].msg);
            }
        }

        if (mode_parse) {
            ast_dump(&P, root, 0);
            printf("\n%u nodes, %d parse error(s)\n",
                   P.num_nodes, P.num_errors);
        }

        /* Semantic analysis */
        sema_ctx_t *sema_ctx = NULL;
        if ((mode_sema || mode_ir || mode_amdgpu || mode_amdgpu_bin)
            && P.num_errors == 0)
        {
            sema_ctx = (sema_ctx_t *)malloc(sizeof(sema_ctx_t));
            if (!sema_ctx) {
                fprintf(stderr, "error: failed to allocate sema context\n");
                return 1;
            }
            sema_init(sema_ctx, &P, root);
            sema_check(sema_ctx, root);

            if (sema_ctx->num_errors > 0) {
                for (int i = 0; i < sema_ctx->num_errors; i++) {
                    fprintf(stderr, "%s:%u:%u: sema: %s\n",
                            file, sema_ctx->errors[i].line,
                            sema_ctx->errors[i].col,
                            sema_ctx->errors[i].msg);
                }
            }

            if (mode_sema) {
                sema_dump(sema_ctx, root);
                int sema_rc = sema_ctx->num_errors > 0 ? 1 : 0;
                free(sema_ctx);
                return sema_rc;
            }
        }

        if ((mode_ir || mode_amdgpu || mode_amdgpu_bin) && P.num_errors == 0) {
            bir_module = (bir_module_t *)malloc(sizeof(bir_module_t));
            if (!bir_module) {
                fprintf(stderr, "error: failed to allocate BIR module\n");
                return 1;
            }
            int lrc = bir_lower(&P, root, bir_module, sema_ctx);
            if (lrc == BC_OK) {
                if (!no_mem2reg)
                    bir_mem2reg(bir_module);

                if (mode_ir) {
                    bir_print_module(bir_module, stdout);
                    printf("\n; %u functions, %u globals, %u instructions\n",
                           bir_module->num_funcs, bir_module->num_globals,
                           bir_module->num_insts);
                }

                if (mode_amdgpu || mode_amdgpu_bin) {
                    amd_module_t *amd = (amd_module_t *)malloc(sizeof(amd_module_t));
                    if (!amd) {
                        fprintf(stderr, "error: failed to allocate AMD module\n");
                        free(bir_module);
                        return 1;
                    }
                    int arc = amdgpu_compile(bir_module, amd);
                    if (arc == BC_OK) {
                        amdgpu_regalloc(amd);
                        if (mode_amdgpu_bin)
                            amdgpu_emit_elf(amd,
                                output_file ? output_file : "a.hsaco");
                        else
                            amdgpu_emit_asm(amd, stdout);
                    } else {
                        fprintf(stderr, "error: AMDGPU compilation failed\n");
                        rc = arc;
                    }
                    free(amd);
                }
            }
            free(bir_module);
            if (lrc != BC_OK) rc = lrc;
        }

        if (sema_ctx) free(sema_ctx);
        if (P.num_errors > 0) rc = BC_ERR_PARSE;
    }

    return rc != BC_OK ? 1 : 0;
}
