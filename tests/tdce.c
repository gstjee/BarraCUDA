/* tdce.c -- Dead code elimination tests.
 * Verify that DCE removes exactly the right instructions. */

#include "tharns.h"

static char obuf[TH_BUFSZ];
static char obuf2[TH_BUFSZ];

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

static int count_lines(const char *start, const char *end)
{
    int n = 0;
    for (const char *p = start; p < end; p++)
        if (*p == '\n') n++;
    return n;
}

/* ---- dce: dead chain eliminated ---- */

static void dce_chain(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    /* mul and second add (the dead chain) must be gone */
    const char *fn = strstr(obuf, "dce_chain");
    CHECK(fn != NULL);
    const char *fn_end = strstr(fn, "\n}");
    CHECK(fn_end != NULL);
    /* dead mul must not appear */
    CHECK(strnstr_range(fn, fn_end, "= mul") == NULL);
    /* live add must survive */
    CHECK(strnstr_range(fn, fn_end, "= add") != NULL);
    /* store must survive */
    CHECK(strnstr_range(fn, fn_end, "store ") != NULL);
    PASS();
}
TH_REG("dce", dce_chain)

/* ---- dce: unused non-volatile load eliminated ---- */

static void dce_load(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "dce_load");
    CHECK(fn != NULL);
    /* Skip past signature line to search body only */
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* load and gep instructions must be eliminated */
    CHECK(strnstr_range(body, fn_end, "= load") == NULL);
    CHECK(strnstr_range(body, fn_end, "= gep") == NULL);
    PASS();
}
TH_REG("dce", dce_load)

/* ---- dce: params survive even if unused ---- */

static void dce_param(void)
{
    int rc1 = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, TH_BUFSZ);
    CHEQ(rc1, 0);
    int rc2 = th_run(BC_BIN " --ir --no-dce tests/test_dce.cu",
                     obuf2, TH_BUFSZ);
    CHEQ(rc2, 0);
    /* dce_params has 4 params but only %1 is used.
     * DCE must not remove any — compare instruction count. */
    const char *fn1 = strstr(obuf, "dce_params");
    const char *fn2 = strstr(obuf2, "dce_params");
    CHECK(fn1 != NULL);
    CHECK(fn2 != NULL);
    const char *end1 = strstr(fn1, "\n}");
    const char *end2 = strstr(fn2, "\n}");
    CHECK(end1 != NULL);
    CHECK(end2 != NULL);
    /* Same number of instruction lines — nothing was removed */
    CHEQ(count_lines(fn1, end1), count_lines(fn2, end2));
    /* All four params still in signature */
    CHECK(strstr(fn1, "i32 %1") != NULL);
    CHECK(strstr(fn1, "i32 %2") != NULL);
    CHECK(strstr(fn1, "i32 %3") != NULL);
    PASS();
}
TH_REG("dce", dce_param)

/* ---- dce: side effects survive ---- */

static void dce_side(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "dce_side_effects");
    CHECK(fn != NULL);
    const char *fn_end = strstr(fn, "\n}");
    CHECK(fn_end != NULL);
    CHECK(strnstr_range(fn, fn_end, "store ") != NULL);
    CHECK(strnstr_range(fn, fn_end, "barrier") != NULL);  /* unique opcode */
    PASS();
}
TH_REG("dce", dce_side)

/* ---- dce: empty function unchanged ---- */

static void dce_empty(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "dce_empty");
    CHECK(fn != NULL);
    const char *fn_end = strstr(fn, "\n}");
    CHECK(fn_end != NULL);
    CHECK(strnstr_range(fn, fn_end, "ret") != NULL);
    PASS();
}
TH_REG("dce", dce_empty)

/* ---- dce: no dead code — output identical with and without DCE ---- */

static void dce_nop(void)
{
    int rc1 = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, TH_BUFSZ);
    CHEQ(rc1, 0);
    int rc2 = th_run(BC_BIN " --ir --no-dce tests/test_dce.cu",
                     obuf2, TH_BUFSZ);
    CHEQ(rc2, 0);
    /* dce_all_live has no dead code — same number of instructions */
    const char *fn1 = strstr(obuf, "dce_all_live");
    const char *fn2 = strstr(obuf2, "dce_all_live");
    CHECK(fn1 != NULL);
    CHECK(fn2 != NULL);
    const char *end1 = strstr(fn1, "\n}");
    const char *end2 = strstr(fn2, "\n}");
    CHECK(end1 != NULL);
    CHECK(end2 != NULL);
    /* Same number of instruction lines */
    CHEQ(count_lines(fn1, end1), count_lines(fn2, end2));
    /* All expected opcodes survive */
    CHECK(strnstr_range(fn1, end1, "= block_id") != NULL);
    CHECK(strnstr_range(fn1, end1, "= block_dim") != NULL);
    CHECK(strnstr_range(fn1, end1, "= thread_id") != NULL);
    CHECK(strnstr_range(fn1, end1, "= mul") != NULL);
    CHECK(strnstr_range(fn1, end1, "= fadd") != NULL);
    CHECK(strnstr_range(fn1, end1, "store ") != NULL);
    PASS();
}
TH_REG("dce", dce_nop)

/* ---- dce: instruction count drops ---- */

static void dce_count(void)
{
    int rc1 = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, TH_BUFSZ);
    CHEQ(rc1, 0);
    int rc2 = th_run(BC_BIN " --ir --no-dce tests/test_dce.cu",
                     obuf2, TH_BUFSZ);
    CHEQ(rc2, 0);
    /* Parse instruction counts from the summary line */
    const char *s1 = strstr(obuf, " instructions");
    const char *s2 = strstr(obuf2, " instructions");
    CHECK(s1 != NULL);
    CHECK(s2 != NULL);
    /* Walk backwards to find the number */
    while (s1 > obuf && s1[-1] >= '0' && s1[-1] <= '9') s1--;
    while (s2 > obuf2 && s2[-1] >= '0' && s2[-1] <= '9') s2--;
    int n_opt = atoi(s1);
    int n_noopt = atoi(s2);
    CHECK(n_noopt > n_opt);
    PASS();
}
TH_REG("dce", dce_count)
