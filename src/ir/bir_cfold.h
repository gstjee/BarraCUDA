#ifndef BARRACUDA_BIR_CFOLD_H
#define BARRACUDA_BIR_CFOLD_H

#include "bir.h"

/*
 * Constant folding.
 *
 * Runs after mem2reg, before DCE.  Evaluates constant expressions
 * at compile time (arithmetic, comparisons, conversions, select)
 * and replaces them with constant values.  DCE cleans up the
 * now-dead instructions afterward.
 *
 * Returns the total number of instructions folded (>= 0).
 */
int bir_cfold(bir_module_t *M);

#endif /* BARRACUDA_BIR_CFOLD_H */
