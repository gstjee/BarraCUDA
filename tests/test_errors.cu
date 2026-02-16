/* test_errors.cu — Deliberate syntax errors to verify error recovery.
 * The compiler should report multiple errors without hanging or crashing.
 * It should NOT produce valid IR — just survive the ordeal with dignity. */

/* Missing semicolons */
__device__ int missing_semi(int x)
{
    int a = x + 1
    int b = a * 2;
    return b;
}

/* Bad expression */
__device__ void bad_expr(float *out)
{
    out[0] = + * 3.0f;
    out[1] = 42.0f;
}

/* Missing closing paren */
__device__ int bad_paren(int a, int b
{
    return a + b;
}

/* Extra tokens at top level */
@ $ %
__device__ void after_garbage(int *p)
{
    p[0] = 1;
}
