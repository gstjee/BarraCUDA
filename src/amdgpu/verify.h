#ifndef BARRACUDA_AMDGPU_VERIFY_H
#define BARRACUDA_AMDGPU_VERIFY_H

#include "amdgpu.h"

/*
 * Post-isel verification pass.
 *
 * Checks machine instructions for encoding violations that would
 * silently miscompile into GPU faults.  Runs twice: once after isel
 * (virtual regs ok) and once after regalloc (physical only).
 *
 * Cheaper than debugging "Reason: Unknown" on a MI300X at 3am.
 */

typedef struct {
    uint32_t errs;
    uint32_t wrns;
} vfy_res_t;

#define VFY_ISEL  0   /* post-isel, virtual regs allowed */
#define VFY_RA    1   /* post-regalloc, physical only    */

vfy_res_t bc_vfy(const amd_module_t *A, int phase);

#endif /* BARRACUDA_AMDGPU_VERIFY_H */
