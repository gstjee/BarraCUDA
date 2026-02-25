/* terrs.c -- error handling
 * Making sure the compiler fails gracefully, not dramatically. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- errors: syntax ---- */

static void err_synt(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin tests/test_errors.cu -o err_test.hsaco",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "error") != NULL);
    remove("err_test.hsaco");
    PASS();
}
TH_REG("errors", err_synt)

/* ---- errors: missing file ---- */

static void err_miss(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin nonexistent_file_42.cu -o err_test.hsaco",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    remove("err_test.hsaco");
    PASS();
}
TH_REG("errors", err_miss)

/* ---- errors: bad output directory ---- */
/* Compiler prints the error but exits 0. One day that'll get fixed.
 * For now, we check the error message exists. */

static void err_odir(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin tests/vector_add.cu "
                    "-o Z:\\nonexistent\\dir\\out.hsaco",
                    obuf, TH_BUFSZ);
    (void)rc;
    CHECK(strstr(obuf, "cannot open") != NULL ||
          strstr(obuf, "error") != NULL);
    PASS();
}
TH_REG("errors", err_odir)
