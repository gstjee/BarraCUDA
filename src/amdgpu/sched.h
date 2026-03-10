#ifndef BARRACUDA_AMDGPU_SCHED_H
#define BARRACUDA_AMDGPU_SCHED_H

#include "amdgpu.h"

/*
 * Instruction scheduling for AMDGPU backend.
 *
 * Reorders instructions within basic blocks to hide memory
 * latency.  Runs between isel and regalloc on virtual registers.
 */

void amdgpu_sched(amd_module_t *A);

#endif /* BARRACUDA_AMDGPU_SCHED_H */
