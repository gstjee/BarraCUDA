#ifndef BARRACUDA_RA_SSA_H
#define BARRACUDA_RA_SSA_H

#include "amdgpu.h"

/*
 * Divergence-aware SSA register allocator.
 * Spills uniform VGPRs cheaply (readfirstlane, 4 bytes scratch)
 * and preserves divergent VGPRs in registers (256 bytes scratch each).
 * Operates on SSA form before phi elimination.
 *
 * References:
 *   Sampaio et al. (2013) "Divergence Analysis", ACM TOPLAS 35(4)
 *   Cooper et al. (2001) "A Simple, Fast Dominance Algorithm"
 *   Braun & Hack (2009) "Register Spilling for SSA-Form Programs"
 */

void ra_ssa(amd_module_t *A, uint32_t mf_idx);

#endif /* BARRACUDA_RA_SSA_H */
