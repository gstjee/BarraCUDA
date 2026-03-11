#ifndef BARRACUDA_PARSER_H
#define BARRACUDA_PARSER_H

#include "ast.h"
#include "token.h"

typedef struct {
    const token_t  *tokens;
    uint32_t        num_tokens;
    uint32_t        pos;
    const char     *src;

    ast_node_t     *nodes;
    uint32_t        num_nodes;
    uint32_t        max_nodes;

    /* Launch bounds stash — held between parse_type_spec and func node creation */
    uint32_t    lb_max_pending;
    uint32_t    lb_min_pending;

    /* Type name registry — struct/typedef names for cast disambiguation.
     * Without this, (var) * expr parses as cast+deref instead of mul.
     * The classic C ambiguity that has ruined more weekends than ISO 8601. */
    struct { uint32_t off; uint16_t len; } tnames[128];
    int             num_tnames;

    bc_error_t      errors[BC_MAX_ERRORS];
    int             num_errors;
} parser_t;

void parser_init(parser_t *P, const token_t *tokens, uint32_t num_tokens,
                 const char *src, ast_node_t *nodes, uint32_t max_nodes);

uint32_t parser_parse(parser_t *P);

void ast_dump(const parser_t *P, uint32_t node, int depth);

#endif /* BARRACUDA_PARSER_H */
