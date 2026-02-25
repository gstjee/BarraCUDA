/* tsmoke.c -- CLI smoke tests
 * Does the binary do anything? Anything at all? Let's find out. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- smoke: help ---- */

static void smk_help(void)
{
    int rc = th_run(BC_BIN " --help", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "Usage") != NULL);
    PASS();
}
TH_REG("smoke", smk_help)

/* ---- smoke: no args ---- */

static void smk_noarg(void)
{
    int rc = th_run(BC_BIN, obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smoke", smk_noarg)

/* ---- smoke: version ---- */

static void smk_vers(void)
{
    SKIP("not implemented");
}
TH_REG("smoke", smk_vers)

/* ---- smoke: bad flag ---- */

static void smk_badf(void)
{
    int rc = th_run(BC_BIN " --nonsense", obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smoke", smk_badf)
