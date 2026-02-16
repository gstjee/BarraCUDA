#ifndef BARRACUDA_BIR_MEM2REG_H
#define BARRACUDA_BIR_MEM2REG_H

#include "bir.h"

/*
 * Promote stack allocas to SSA registers.
 *
 * Scalar allocas whose only uses are plain loads and stores
 * (no GEP, no call argument, no volatile) get replaced with
 * direct values and phi nodes.  Array / struct / address-taken
 * allocas are left alone.
 *
 * Returns the number of allocas promoted (>= 0).
 */
int bir_mem2reg(bir_module_t *M);

#endif /* BARRACUDA_BIR_MEM2REG_H */
