#ifndef BARRACUDA_BIR_DCE_H
#define BARRACUDA_BIR_DCE_H

#include "bir.h"

/*
 * Dead code elimination.
 *
 * Runs after mem2reg.  Removes instructions whose results
 * are never used and that have no side effects.
 *
 * Returns the total number of instructions removed (>= 0).
 */
int bir_dce(bir_module_t *M);

#endif /* BARRACUDA_BIR_DCE_H */
