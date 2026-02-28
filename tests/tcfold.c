/* tcfold.c -- Constant folding tests.
 * Verify that constant folding evaluates constant expressions. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- Helpers ---- */

static const char *strnstr_range(const char *start, const char *end,
                                 const char *needle)
{
    size_t nlen = strlen(needle);
    for (const char *p = start; p + nlen <= end; p++) {
        if (memcmp(p, needle, nlen) == 0) return p;
    }
    return NULL;
}

/* ---- cf: integer arithmetic folded ---- */

static void cf_int_arith(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_int_arith");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* The constant add (3+4) must be folded — add with param survives */
    CHECK(strnstr_range(body, fn_end, "= add") != NULL);
    /* The constant 7 (= 3+4) should appear as an operand */
    CHECK(strnstr_range(body, fn_end, " 7") != NULL);
    PASS();
}
TH_REG("cf", cf_int_arith)

/* ---- cf: chained constants fold ---- */

static void cf_chain(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_chain");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* mul (5*4=20) must be folded away */
    CHECK(strnstr_range(body, fn_end, "= mul") == NULL);
    /* The constant 20 should appear as an operand */
    CHECK(strnstr_range(body, fn_end, " 20") != NULL);
    PASS();
}
TH_REG("cf", cf_chain)

/* ---- cf: icmp + select with constant condition ---- */

static void cf_icmp_select(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_icmp_select");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* icmp and select must be folded away */
    CHECK(strnstr_range(body, fn_end, "= icmp") == NULL);
    CHECK(strnstr_range(body, fn_end, "= select") == NULL);
    PASS();
}
TH_REG("cf", cf_icmp_select)

/* ---- cf: integer division by zero not folded ---- */

static void cf_divzero(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_divzero");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* sdiv by zero must survive — folding it would be UB */
    CHECK(strnstr_range(body, fn_end, "= sdiv") != NULL);
    PASS();
}
TH_REG("cf", cf_divzero)

/* ---- cf: integer/float conversions folded ---- */

static void cf_conv(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_conv");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* sitofp and fptosi must be folded away */
    CHECK(strnstr_range(body, fn_end, "= sitofp") == NULL);
    CHECK(strnstr_range(body, fn_end, "= fptosi") == NULL);
    PASS();
}
TH_REG("cf", cf_conv)

/* ---- cf: float arithmetic folded ---- */

static void cf_float(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_float");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* fadd must be folded away — constant 4 (=1.5+2.5) stored directly */
    CHECK(strnstr_range(body, fn_end, "= fadd") == NULL);
    /* The folded value 4 should appear as an f32 operand */
    CHECK(strnstr_range(body, fn_end, "f32 4") != NULL);
    PASS();
}
TH_REG("cf", cf_float)
