#include "amdgpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/*
 * AMDGPU emitter: phi elimination, register allocation, assembly printer,
 * binary encoder, and ELF code object writer.
 * Targets RDNA 3 (gfx1100, Wave32). Dependencies: libc, optimism, tea.
 */

/* ---- Encoding Table ---- */

/* Maps amd_op_t -> (format, hw_opcode, mnemonic) */
/* GFX11 (gfx1100, RDNA 3) hardware opcodes.
   Extracted by assembling with llvm-mc and reading the bones.
   */
const amd_enc_entry_t amd_enc_table[AMD_OP_COUNT] = {
    /* SOP2 */
    [AMD_S_ADD_U32]         = { AMD_FMT_SOP2, 0x00, "s_add_u32"         },
    [AMD_S_SUB_U32]         = { AMD_FMT_SOP2, 0x01, "s_sub_u32"         },
    [AMD_S_MUL_I32]         = { AMD_FMT_SOP2, 0x2C, "s_mul_i32"         },
    [AMD_S_AND_B32]         = { AMD_FMT_SOP2, 0x16, "s_and_b32"         },
    [AMD_S_OR_B32]          = { AMD_FMT_SOP2, 0x18, "s_or_b32"          },
    [AMD_S_XOR_B32]         = { AMD_FMT_SOP2, 0x1A, "s_xor_b32"         },
    [AMD_S_LSHL_B32]        = { AMD_FMT_SOP2, 0x08, "s_lshl_b32"        },
    [AMD_S_LSHR_B32]        = { AMD_FMT_SOP2, 0x0A, "s_lshr_b32"        },
    [AMD_S_ASHR_I32]        = { AMD_FMT_SOP2, 0x0C, "s_ashr_i32"        },
    [AMD_S_ANDN2_B32]       = { AMD_FMT_SOP2, 0x22, "s_andn2_b32"       },
    [AMD_S_ORN2_B32]        = { AMD_FMT_SOP2, 0x24, "s_orn2_b32"        },
    [AMD_S_BFE_I32]         = { AMD_FMT_SOP2, 0x27, "s_bfe_i32"         },
    [AMD_S_CSELECT_B32]     = { AMD_FMT_SOP2, 0x30, "s_cselect_b32"     },

    /* SOP1 */
    [AMD_S_MOV_B32]         = { AMD_FMT_SOP1, 0x00, "s_mov_b32"         },
    [AMD_S_NOT_B32]         = { AMD_FMT_SOP1, 0x1E, "s_not_b32"         },
    [AMD_S_AND_SAVEEXEC_B32]= { AMD_FMT_SOP1, 0x20, "s_and_saveexec_b32"},
    [AMD_S_SETPC_B64]       = { AMD_FMT_SOP1, 0x48, "s_setpc_b64"       },
    [AMD_S_SWAPPC_B64]      = { AMD_FMT_SOP1, 0x49, "s_swappc_b64"      },
    [AMD_S_GETPC_B64]       = { AMD_FMT_SOP1, 0x47, "s_getpc_b64"       },

    /* SOPC */
    [AMD_S_CMP_EQ_U32]      = { AMD_FMT_SOPC, 0x06, "s_cmp_eq_u32"      },
    [AMD_S_CMP_NE_U32]      = { AMD_FMT_SOPC, 0x06, "s_cmp_eq_u32"      }, /* GFX11: no NE, use EQ+invert */
    [AMD_S_CMP_LT_U32]      = { AMD_FMT_SOPC, 0x0A, "s_cmp_lt_u32"      },
    [AMD_S_CMP_LE_U32]      = { AMD_FMT_SOPC, 0x0B, "s_cmp_le_u32"      },
    [AMD_S_CMP_GT_U32]      = { AMD_FMT_SOPC, 0x08, "s_cmp_gt_u32"      },
    [AMD_S_CMP_GE_U32]      = { AMD_FMT_SOPC, 0x09, "s_cmp_ge_u32"      },
    [AMD_S_CMP_LT_I32]      = { AMD_FMT_SOPC, 0x04, "s_cmp_lt_i32"      },
    [AMD_S_CMP_LE_I32]      = { AMD_FMT_SOPC, 0x05, "s_cmp_le_i32"      },
    [AMD_S_CMP_GT_I32]      = { AMD_FMT_SOPC, 0x02, "s_cmp_gt_i32"      },
    [AMD_S_CMP_GE_I32]      = { AMD_FMT_SOPC, 0x03, "s_cmp_ge_i32"      },
    [AMD_S_CMP_EQ_I32]      = { AMD_FMT_SOPC, 0x00, "s_cmp_eq_i32"      },
    [AMD_S_CMP_NE_I32]      = { AMD_FMT_SOPC, 0x00, "s_cmp_eq_i32"      }, /* GFX11: no NE, use EQ+invert */

    /* SOPP */
    [AMD_S_BRANCH]           = { AMD_FMT_SOPP, 0x20, "s_branch"           },
    [AMD_S_CBRANCH_SCC0]     = { AMD_FMT_SOPP, 0x21, "s_cbranch_scc0"     },
    [AMD_S_CBRANCH_SCC1]     = { AMD_FMT_SOPP, 0x22, "s_cbranch_scc1"     },
    [AMD_S_CBRANCH_EXECZ]    = { AMD_FMT_SOPP, 0x25, "s_cbranch_execz"    },
    [AMD_S_CBRANCH_EXECNZ]   = { AMD_FMT_SOPP, 0x26, "s_cbranch_execnz"   },
    [AMD_S_ENDPGM]           = { AMD_FMT_SOPP, 0x30, "s_endpgm"           },
    [AMD_S_BARRIER]          = { AMD_FMT_SOPP, 0x3D, "s_barrier"          },
    [AMD_S_WAITCNT]          = { AMD_FMT_SOPP, 0x09, "s_waitcnt"          },
    [AMD_S_NOP]              = { AMD_FMT_SOPP, 0x00, "s_nop"              },
    [AMD_S_WAIT_LOADCNT]     = { AMD_FMT_SOPP, 0x40, "s_wait_loadcnt"    },
    [AMD_S_WAIT_STORECNT]    = { AMD_FMT_SOPP, 0x41, "s_wait_storecnt"   },
    [AMD_S_WAIT_DSCNT]       = { AMD_FMT_SOPP, 0x46, "s_wait_dscnt"     },
    [AMD_S_WAIT_KMCNT]       = { AMD_FMT_SOPP, 0x47, "s_wait_kmcnt"     },

    /* SMEM — GFX11: prefix=0x3D, OP at [25:18] */
    [AMD_S_LOAD_DWORD]       = { AMD_FMT_SMEM, 0x00, "s_load_dword"       },
    [AMD_S_LOAD_DWORDX2]     = { AMD_FMT_SMEM, 0x01, "s_load_dwordx2"     },
    [AMD_S_LOAD_DWORDX4]     = { AMD_FMT_SMEM, 0x02, "s_load_dwordx4"     },

    /* VOP2 */
    [AMD_V_ADD_U32]          = { AMD_FMT_VOP2, 0x25, "v_add_u32"          },
    [AMD_V_SUB_U32]          = { AMD_FMT_VOP2, 0x26, "v_sub_u32"          },
    [AMD_V_MUL_LO_U32]      = { AMD_FMT_VOP3, 0x32C,"v_mul_lo_u32"       },
    [AMD_V_AND_B32]          = { AMD_FMT_VOP2, 0x1B, "v_and_b32"          },
    [AMD_V_OR_B32]           = { AMD_FMT_VOP2, 0x1C, "v_or_b32"           },
    [AMD_V_XOR_B32]          = { AMD_FMT_VOP2, 0x1D, "v_xor_b32"          },
    [AMD_V_LSHLREV_B32]      = { AMD_FMT_VOP2, 0x18, "v_lshlrev_b32"      },
    [AMD_V_LSHRREV_B32]      = { AMD_FMT_VOP2, 0x19, "v_lshrrev_b32"      },
    [AMD_V_ASHRREV_I32]      = { AMD_FMT_VOP2, 0x1A, "v_ashrrev_i32"      },
    [AMD_V_ADD_F32]          = { AMD_FMT_VOP2, 0x03, "v_add_f32"          },
    [AMD_V_SUB_F32]          = { AMD_FMT_VOP2, 0x04, "v_sub_f32"          },
    [AMD_V_MUL_F32]          = { AMD_FMT_VOP2, 0x08, "v_mul_f32"          },
    [AMD_V_CNDMASK_B32]      = { AMD_FMT_VOP2, 0x01, "v_cndmask_b32"      },
    [AMD_V_MIN_F32]          = { AMD_FMT_VOP2, 0x0F, "v_min_f32"          },
    [AMD_V_MAX_F32]          = { AMD_FMT_VOP2, 0x10, "v_max_f32"          },

    /* VOP1 */
    [AMD_V_MOV_B32]          = { AMD_FMT_VOP1, 0x01, "v_mov_b32"          },
    [AMD_V_CVT_F32_I32]      = { AMD_FMT_VOP1, 0x05, "v_cvt_f32_i32"      },
    [AMD_V_CVT_F32_U32]      = { AMD_FMT_VOP1, 0x06, "v_cvt_f32_u32"      },
    [AMD_V_CVT_I32_F32]      = { AMD_FMT_VOP1, 0x08, "v_cvt_i32_f32"      },
    [AMD_V_CVT_U32_F32]      = { AMD_FMT_VOP1, 0x07, "v_cvt_u32_f32"      },
    [AMD_V_CVT_F32_F16]      = { AMD_FMT_VOP1, 0x0B, "v_cvt_f32_f16"      },
    [AMD_V_CVT_F16_F32]      = { AMD_FMT_VOP1, 0x0A, "v_cvt_f16_f32"      },
    [AMD_V_CVT_F64_F32]      = { AMD_FMT_VOP1, 0x10, "v_cvt_f64_f32"      },
    [AMD_V_CVT_F32_F64]      = { AMD_FMT_VOP1, 0x0F, "v_cvt_f32_f64"      },
    [AMD_V_RCP_F32]          = { AMD_FMT_VOP1, 0x2A, "v_rcp_f32"          },
    [AMD_V_SQRT_F32]         = { AMD_FMT_VOP1, 0x33, "v_sqrt_f32"         },
    [AMD_V_RSQ_F32]          = { AMD_FMT_VOP1, 0x2E, "v_rsq_f32"          },
    [AMD_V_EXP_F32]          = { AMD_FMT_VOP1, 0x25, "v_exp_f32"          },
    [AMD_V_LOG_F32]          = { AMD_FMT_VOP1, 0x27, "v_log_f32"          },
    [AMD_V_SIN_F32]          = { AMD_FMT_VOP1, 0x35, "v_sin_f32"          },
    [AMD_V_COS_F32]          = { AMD_FMT_VOP1, 0x36, "v_cos_f32"          },
    [AMD_V_FLOOR_F32]        = { AMD_FMT_VOP1, 0x24, "v_floor_f32"        },
    [AMD_V_CEIL_F32]         = { AMD_FMT_VOP1, 0x22, "v_ceil_f32"         },
    [AMD_V_TRUNC_F32]        = { AMD_FMT_VOP1, 0x21, "v_trunc_f32"        },
    [AMD_V_RNDNE_F32]        = { AMD_FMT_VOP1, 0x23, "v_rndne_f32"        },
    [AMD_V_FRACT_F32]        = { AMD_FMT_VOP1, 0x20, "v_fract_f32"        },
    [AMD_V_NOT_B32]          = { AMD_FMT_VOP1, 0x37, "v_not_b32"          },
    [AMD_V_READFIRSTLANE_B32]= { AMD_FMT_VOP1, 0x02, "v_readfirstlane_b32"},

    /* VOP3 — GFX11: prefix=0x35, 10-bit opcodes */
    [AMD_V_MAD_U32_U24]      = { AMD_FMT_VOP3, 0x20B,"v_mad_u32_u24"      },
    [AMD_V_BFE_I32]          = { AMD_FMT_VOP3, 0x211,"v_bfe_i32"          },
    [AMD_V_BFE_U32]          = { AMD_FMT_VOP3, 0x210,"v_bfe_u32"          },
    [AMD_V_LSHL_ADD_U32]     = { AMD_FMT_VOP3, 0x246,"v_lshl_add_u32"     },
    [AMD_V_ADD3_U32]         = { AMD_FMT_VOP3, 0x255,"v_add3_u32"         },

    /* VOPC — GFX11: OP at [24:17], 8-bit field */
    [AMD_V_CMP_EQ_U32]       = { AMD_FMT_VOPC, 0x4A, "v_cmp_eq_u32"       },
    [AMD_V_CMP_NE_U32]       = { AMD_FMT_VOPC, 0x4D, "v_cmp_ne_u32"       },
    [AMD_V_CMP_LT_U32]       = { AMD_FMT_VOPC, 0x49, "v_cmp_lt_u32"       },
    [AMD_V_CMP_LE_U32]       = { AMD_FMT_VOPC, 0x4B, "v_cmp_le_u32"       },
    [AMD_V_CMP_GT_U32]       = { AMD_FMT_VOPC, 0x4C, "v_cmp_gt_u32"       },
    [AMD_V_CMP_GE_U32]       = { AMD_FMT_VOPC, 0x4E, "v_cmp_ge_u32"       },
    [AMD_V_CMP_LT_I32]       = { AMD_FMT_VOPC, 0x41, "v_cmp_lt_i32"       },
    [AMD_V_CMP_LE_I32]       = { AMD_FMT_VOPC, 0x43, "v_cmp_le_i32"       },
    [AMD_V_CMP_GT_I32]       = { AMD_FMT_VOPC, 0x44, "v_cmp_gt_i32"       },
    [AMD_V_CMP_GE_I32]       = { AMD_FMT_VOPC, 0x46, "v_cmp_ge_i32"       },
    [AMD_V_CMP_EQ_I32]       = { AMD_FMT_VOPC, 0x42, "v_cmp_eq_i32"       },
    [AMD_V_CMP_NE_I32]       = { AMD_FMT_VOPC, 0x45, "v_cmp_ne_i32"       },
    [AMD_V_CMP_EQ_F32]       = { AMD_FMT_VOPC, 0x12, "v_cmp_eq_f32"       },
    [AMD_V_CMP_NE_F32]       = { AMD_FMT_VOPC, 0x1D, "v_cmp_neq_f32"      },
    [AMD_V_CMP_LT_F32]       = { AMD_FMT_VOPC, 0x11, "v_cmp_lt_f32"       },
    [AMD_V_CMP_LE_F32]       = { AMD_FMT_VOPC, 0x13, "v_cmp_le_f32"       },
    [AMD_V_CMP_GT_F32]       = { AMD_FMT_VOPC, 0x14, "v_cmp_gt_f32"       },
    [AMD_V_CMP_GE_F32]       = { AMD_FMT_VOPC, 0x16, "v_cmp_ge_f32"       },
    [AMD_V_CMP_O_F32]        = { AMD_FMT_VOPC, 0x17, "v_cmp_o_f32"        },
    [AMD_V_CMP_U_F32]        = { AMD_FMT_VOPC, 0x18, "v_cmp_u_f32"        },
    [AMD_V_CMP_NLT_F32]      = { AMD_FMT_VOPC, 0x1E, "v_cmp_nlt_f32"      },
    [AMD_V_CMP_NLE_F32]      = { AMD_FMT_VOPC, 0x1C, "v_cmp_nle_f32"      },
    [AMD_V_CMP_NGT_F32]      = { AMD_FMT_VOPC, 0x1B, "v_cmp_ngt_f32"      },
    [AMD_V_CMP_NGE_F32]      = { AMD_FMT_VOPC, 0x19, "v_cmp_nge_f32"      },
    [AMD_V_CMP_NEQ_F32]      = { AMD_FMT_VOPC, 0x1D, "v_cmp_neq_f32"      },

    /* DS */
    [AMD_DS_READ_B32]         = { AMD_FMT_DS, 0x36, "ds_load_b32"         },
    [AMD_DS_WRITE_B32]        = { AMD_FMT_DS, 0x0D, "ds_store_b32"        },
    [AMD_DS_ADD_RTN_U32]      = { AMD_FMT_DS, 0x20, "ds_add_rtn_u32"      },
    [AMD_DS_SUB_RTN_U32]      = { AMD_FMT_DS, 0x21, "ds_sub_rtn_u32"      },
    [AMD_DS_AND_RTN_B32]      = { AMD_FMT_DS, 0x29, "ds_and_rtn_b32"      },
    [AMD_DS_OR_RTN_B32]       = { AMD_FMT_DS, 0x2A, "ds_or_rtn_b32"       },
    [AMD_DS_XOR_RTN_B32]      = { AMD_FMT_DS, 0x2B, "ds_xor_rtn_b32"      },
    [AMD_DS_MIN_RTN_I32]      = { AMD_FMT_DS, 0x25, "ds_min_rtn_i32"      },
    [AMD_DS_MAX_RTN_I32]      = { AMD_FMT_DS, 0x26, "ds_max_rtn_i32"      },
    [AMD_DS_SWIZZLE_B32]      = { AMD_FMT_DS, 0x35, "ds_swizzle_b32"      },
    [AMD_DS_BPERMUTE_B32]     = { AMD_FMT_DS, 0xB3, "ds_bpermute_b32"     },

    /* FLAT_GBL — GFX11: OP at [25:18], seg at [17:16] (global=2) */
    [AMD_GLOBAL_LOAD_DWORD]      = { AMD_FMT_FLAT_GBL, 0x14, "global_load_dword"      },
    [AMD_GLOBAL_STORE_DWORD]     = { AMD_FMT_FLAT_GBL, 0x1A, "global_store_dword"     },
    [AMD_GLOBAL_LOAD_DWORDX2]    = { AMD_FMT_FLAT_GBL, 0x15, "global_load_dwordx2"    },
    [AMD_GLOBAL_STORE_DWORDX2]   = { AMD_FMT_FLAT_GBL, 0x1B, "global_store_dwordx2"   },
    [AMD_GLOBAL_ATOMIC_ADD]      = { AMD_FMT_FLAT_GBL, 0x35, "global_atomic_add"      },
    [AMD_GLOBAL_ATOMIC_SUB]      = { AMD_FMT_FLAT_GBL, 0x36, "global_atomic_sub"      },
    [AMD_GLOBAL_ATOMIC_AND]      = { AMD_FMT_FLAT_GBL, 0x3C, "global_atomic_and"      },
    [AMD_GLOBAL_ATOMIC_OR]       = { AMD_FMT_FLAT_GBL, 0x3D, "global_atomic_or"       },
    [AMD_GLOBAL_ATOMIC_XOR]      = { AMD_FMT_FLAT_GBL, 0x3E, "global_atomic_xor"      },
    [AMD_GLOBAL_ATOMIC_SMIN]     = { AMD_FMT_FLAT_GBL, 0x38, "global_atomic_smin"     },
    [AMD_GLOBAL_ATOMIC_SMAX]     = { AMD_FMT_FLAT_GBL, 0x3A, "global_atomic_smax"     },
    [AMD_GLOBAL_ATOMIC_SWAP]     = { AMD_FMT_FLAT_GBL, 0x33, "global_atomic_swap"     },
    [AMD_GLOBAL_ATOMIC_CMPSWAP]  = { AMD_FMT_FLAT_GBL, 0x34, "global_atomic_cmpswap"  },

    /* FLAT_SCR — GFX11: seg=1 (scratch) */
    [AMD_SCRATCH_LOAD_DWORD]     = { AMD_FMT_FLAT_SCR, 0x14, "scratch_load_dword"     },
    [AMD_SCRATCH_STORE_DWORD]    = { AMD_FMT_FLAT_SCR, 0x1A, "scratch_store_dword"    },

    /* Pseudo */
    [AMD_PSEUDO_PHI]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_PHI"  },
    [AMD_PSEUDO_COPY]            = { AMD_FMT_PSEUDO, 0, "PSEUDO_COPY" },
    [AMD_PSEUDO_DEF]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_DEF"  },
};

/* ---- Phi Elimination ---- */

/*
 * Proper phi elimination: insert copies at predecessor block ends.
 * For each PSEUDO_PHI with operands [(pred, src), ...], insert
 * PSEUDO_COPY dst, srcN at the end of each predecessor (before its
 * terminator). Then NOP the PHI. Cycles are mercifully rare from our
 * frontend, so we don't break them — yet.
 */

#define PHI_MAX_COPIES 4096

typedef struct {
    uint32_t   pred_mb;
    moperand_t dst;
    moperand_t src;
} phi_copy_t;

static phi_copy_t phi_copies[PHI_MAX_COPIES];

/* Is this instruction a block terminator? */
static int is_terminator(uint16_t op)
{
    return op == AMD_S_BRANCH || op == AMD_S_CBRANCH_SCC0 ||
           op == AMD_S_CBRANCH_SCC1 || op == AMD_S_CBRANCH_EXECZ ||
           op == AMD_S_CBRANCH_EXECNZ || op == AMD_S_ENDPGM ||
           op == AMD_S_SETPC_B64;
}

void amdgpu_phi_elim(amd_module_t *A)
{
    uint32_t nc = 0;

    /* Phase 1: collect copies from PHIs, NOP the PHIs */
    for (uint32_t i = 0; i < A->num_minsts; i++) {
        minst_t *mi = &A->minsts[i];
        if (mi->op != AMD_PSEUDO_PHI) continue;

        moperand_t dst = mi->operands[0];
        for (uint8_t p = 0; p + 1 < mi->num_uses && nc < PHI_MAX_COPIES; p += 2) {
            uint32_t off = mi->num_defs + p;
            if (off + 1 >= MINST_MAX_OPS) break;
            if (mi->operands[off].kind != MOP_LABEL) continue;

            phi_copies[nc].pred_mb = (uint32_t)mi->operands[off].imm;
            phi_copies[nc].dst = dst;
            phi_copies[nc].src = mi->operands[off + 1];
            nc++;
        }
        mi->op = AMD_S_NOP;
        mi->num_defs = 0;
        mi->num_uses = 0;
    }

    if (nc == 0) return;

    /* Phase 2: count copies per predecessor block */
    static uint32_t cpb[AMD_MAX_MBLOCKS]; /* copies per block */
    memset(cpb, 0, A->num_mblocks * sizeof(uint32_t));
    for (uint32_t i = 0; i < nc; i++) {
        if (phi_copies[i].pred_mb < A->num_mblocks)
            cpb[phi_copies[i].pred_mb]++;
    }

    /* Phase 3: insert copies before terminators, processing blocks in
       reverse order so shifts don't affect already-processed blocks. */
    for (uint32_t mb = A->num_mblocks; mb > 0; mb--) {
        uint32_t b = mb - 1;
        uint32_t copies_here = cpb[b];
        if (copies_here == 0) continue;
        if (A->num_minsts + copies_here > AMD_MAX_MINSTS) continue;

        mblock_t *B = &A->mblocks[b];

        /* Find insertion point: before trailing terminators */
        uint32_t insert_rel = B->num_insts;
        for (uint32_t ii = B->num_insts; ii > 0; ii--) {
            if (is_terminator(A->minsts[B->first_inst + ii - 1].op))
                insert_rel = ii - 1;
            else
                break;
        }
        uint32_t insert_abs = B->first_inst + insert_rel;

        /* Shift tail of instruction array to make room */
        uint32_t tail_len = A->num_minsts - insert_abs;
        memmove(&A->minsts[insert_abs + copies_here],
                &A->minsts[insert_abs],
                tail_len * sizeof(minst_t));

        /* Insert copies */
        uint32_t ci = 0;
        for (uint32_t i = 0; i < nc && ci < copies_here; i++) {
            if (phi_copies[i].pred_mb != b) continue;
            minst_t *copy = &A->minsts[insert_abs + ci];
            memset(copy, 0, sizeof(minst_t));
            copy->op = AMD_PSEUDO_COPY;
            copy->num_defs = 1;
            copy->num_uses = 1;
            copy->operands[0] = phi_copies[i].dst;
            copy->operands[1] = phi_copies[i].src;
            ci++;
        }

        A->num_minsts += copies_here;
        B->num_insts += copies_here;

        /* Update first_inst for all subsequent blocks */
        for (uint32_t later = b + 1; later < A->num_mblocks; later++)
            A->mblocks[later].first_inst += copies_here;
    }
}

/* ---- Register Allocation (Linear Scan) ---- */

/* Live interval for a virtual register */
typedef struct {
    uint32_t vreg;
    uint32_t start;    /* first def */
    uint32_t end;      /* last use */
    uint16_t phys;     /* allocated physical reg */
    uint8_t  file;     /* 0=SGPR, 1=VGPR */
    uint8_t  spilled;
} live_interval_t;

/* Static storage for regalloc (~4 MB) */
static struct {
    live_interval_t intervals[AMD_MAX_VREGS];
    uint32_t        num_intervals;

    /* Sort index */
    uint32_t        sorted[AMD_MAX_VREGS];

    /* Free register pools */
    uint8_t         sgpr_free[AMD_MAX_SGPRS];
    uint8_t         vgpr_free[AMD_MAX_VGPRS];
    uint32_t        num_sgpr_free;
    uint32_t        num_vgpr_free;

    /* Active intervals sorted by end point */
    uint32_t        active[AMD_MAX_VREGS];
    uint32_t        num_active;

    /* Track max used */
    uint16_t        max_sgpr;
    uint16_t        max_vgpr;
} RA;

static int interval_cmp_start(const void *a, const void *b)
{
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (RA.intervals[ia].start != RA.intervals[ib].start)
        return (RA.intervals[ia].start < RA.intervals[ib].start) ? -1 : 1;
    return 0;
}

/* Get the vreg referenced by an operand, or 0xFFFF if not a vreg */
static uint16_t operand_vreg(const moperand_t *op)
{
    if (op->kind == MOP_VREG_S || op->kind == MOP_VREG_V)
        return op->reg_num;
    return 0xFFFF;
}

static void compute_live_intervals(const amd_module_t *A, const mfunc_t *F)
{
    RA.num_intervals = 0;

    /* Initialize: one interval per vreg, with start=MAX, end=0 */
    for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
        RA.intervals[v].vreg = v;
        RA.intervals[v].start = 0xFFFFFFFF;
        RA.intervals[v].end = 0;
        RA.intervals[v].phys = 0xFFFF;
        RA.intervals[v].file = A->reg_file[v];
        RA.intervals[v].spilled = 0;
    }

    /* Walk all instructions in the function */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            const minst_t *mi = &A->minsts[mi_idx];

            /* Defs */
            for (uint8_t d = 0; d < mi->num_defs && d < MINST_MAX_OPS; d++) {
                uint16_t vr = operand_vreg(&mi->operands[d]);
                if (vr != 0xFFFF) {
                    if (mi_idx < RA.intervals[vr].start)
                        RA.intervals[vr].start = mi_idx;
                    if (mi_idx > RA.intervals[vr].end)
                        RA.intervals[vr].end = mi_idx;
                }
            }

            /* Uses */
            for (uint8_t u = mi->num_defs; u < mi->num_defs + mi->num_uses && u < MINST_MAX_OPS; u++) {
                uint16_t vr = operand_vreg(&mi->operands[u]);
                if (vr != 0xFFFF) {
                    if (mi_idx < RA.intervals[vr].start)
                        RA.intervals[vr].start = mi_idx;
                    if (mi_idx > RA.intervals[vr].end)
                        RA.intervals[vr].end = mi_idx;
                }
            }
        }
    }

    /* Collect valid intervals */
    RA.num_intervals = 0;
    for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
        if (RA.intervals[v].start != 0xFFFFFFFF) {
            RA.sorted[RA.num_intervals++] = v;
        }
    }

    /* Sort by start point */
    qsort(RA.sorted, RA.num_intervals, sizeof(uint32_t), interval_cmp_start);
}

static void expire_old(uint32_t point)
{
    /* Remove intervals that have ended before this point */
    uint32_t j = 0;
    for (uint32_t i = 0; i < RA.num_active; i++) {
        uint32_t v = RA.active[i];
        if (RA.intervals[v].end >= point) {
            RA.active[j++] = v;
        } else {
            /* Free the register */
            uint16_t phys = RA.intervals[v].phys;
            if (RA.intervals[v].file == 0 && phys < AMD_MAX_SGPRS) {
                RA.sgpr_free[RA.num_sgpr_free++] = (uint8_t)phys;
            } else if (phys < AMD_MAX_VGPRS) {
                RA.vgpr_free[RA.num_vgpr_free++] = (uint8_t)phys;
            }
        }
    }
    RA.num_active = j;
}

static void regalloc_function(amd_module_t *A, uint32_t mf_idx)  /* called from amdgpu_regalloc */
{
    mfunc_t *F = &A->mfuncs[mf_idx];

    /* Initialize free pools */
    RA.num_sgpr_free = 0;
    RA.num_vgpr_free = 0;
    RA.max_sgpr = 0;
    RA.max_vgpr = 0;
    RA.num_active = 0;

    /* Push high regs first so low regs are popped first (stack order) */
    uint16_t sgpr_start = F->is_kernel ? F->first_alloc_sgpr : 0;
    if (sgpr_start < AMD_KERN_RESERVED_SGPR && F->is_kernel)
        sgpr_start = AMD_KERN_RESERVED_SGPR;
    for (uint16_t r = AMD_MAX_SGPRS; r-- > sgpr_start; )
        RA.sgpr_free[RA.num_sgpr_free++] = (uint8_t)r;
    for (uint16_t r = AMD_MAX_VGPRS; r-- > 0; )
        RA.vgpr_free[RA.num_vgpr_free++] = (uint8_t)r;

    compute_live_intervals(A, F);

    /* Linear scan */
    for (uint32_t i = 0; i < RA.num_intervals; i++) {
        uint32_t v = RA.sorted[i];
        live_interval_t *iv = &RA.intervals[v];

        expire_old(iv->start);

        uint16_t phys = 0xFFFF;
        if (iv->file == 0) {
            /* SGPR */
            if (RA.num_sgpr_free > 0) {
                phys = RA.sgpr_free[--RA.num_sgpr_free];
                if (phys >= RA.max_sgpr) RA.max_sgpr = phys + 1;
            }
        } else {
            /* VGPR */
            if (RA.num_vgpr_free > 0) {
                phys = RA.vgpr_free[--RA.num_vgpr_free];
                if (phys >= RA.max_vgpr) RA.max_vgpr = phys + 1;
            }
        }

        if (phys == 0xFFFF) {
            /* Spill: find the active interval with the farthest end */
            uint32_t farthest = 0, farthest_idx = 0;
            for (uint32_t a = 0; a < RA.num_active; a++) {
                uint32_t av = RA.active[a];
                if (RA.intervals[av].file == iv->file &&
                    RA.intervals[av].end > farthest) {
                    farthest = RA.intervals[av].end;
                    farthest_idx = a;
                }
            }
            if (farthest > iv->end && RA.num_active > 0) {
                /* Spill the farthest, give its reg to us */
                uint32_t sv = RA.active[farthest_idx];
                phys = RA.intervals[sv].phys;
                RA.intervals[sv].spilled = 1;
                RA.intervals[sv].phys = 0xFFFF;
                /* Remove from active */
                RA.active[farthest_idx] = RA.active[--RA.num_active];
            } else {
                /* Spill ourselves */
                iv->spilled = 1;
                phys = 0; /* fallback */
            }
        }

        iv->phys = phys;
        A->reg_map[v] = phys;

        /* Add to active */
        if (RA.num_active < AMD_MAX_VREGS)
            RA.active[RA.num_active++] = v;
    }

    /* Record usage for kernel descriptor */
    F->num_sgprs = RA.max_sgpr;
    F->num_vgprs = RA.max_vgpr;

    /* Minimum 1 SGPR/VGPR for the descriptor */
    if (F->num_sgprs == 0) F->num_sgprs = 1;
    if (F->num_vgprs == 0) F->num_vgprs = 1;

    /* __launch_bounds__ VGPR cap. More threads = fewer registers.
       The maths of sharing: 256 VGPRs divided among the waves you
       promised the hardware you'd run. Break the promise at your peril. */
    if (F->launch_bounds_max > 0 && F->launch_bounds_max < 1024) {
        uint32_t desired_waves = (F->launch_bounds_max + 31) / 32;
        if (desired_waves > 0) {
            uint32_t vgpr_cap = (256 / desired_waves) & ~7u;
            if (vgpr_cap < 8) vgpr_cap = 8;
            if (F->num_vgprs > vgpr_cap)
                F->num_vgprs = (uint16_t)vgpr_cap;
        }
    }

    /* Rewrite virtual reg operands to physical */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            minst_t *mi = &A->minsts[mi_idx];

            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

            for (uint8_t k = 0; k < total; k++) {
                moperand_t *op = &mi->operands[k];
                if (op->kind == MOP_VREG_S) {
                    op->kind = MOP_SGPR;
                    op->reg_num = A->reg_map[op->reg_num];
                } else if (op->kind == MOP_VREG_V) {
                    op->kind = MOP_VGPR;
                    op->reg_num = A->reg_map[op->reg_num];
                }
            }

            /* Convert PSEUDO_COPY to actual MOV */
            if (mi->op == AMD_PSEUDO_COPY) {
                if (mi->operands[0].kind == MOP_VGPR)
                    mi->op = AMD_V_MOV_B32;
                else
                    mi->op = AMD_S_MOV_B32;
            }
        }
    }

    /* Dead copy elimination: kill MOVs where src == dst.
       These appear when regalloc assigns the same phys reg to both sides
       of a copy. Harmless but noisy — like a postman delivering a letter
       back to the sender. */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            minst_t *mi = &A->minsts[mi_idx];

            if ((mi->op == AMD_V_MOV_B32 || mi->op == AMD_S_MOV_B32) &&
                mi->num_defs == 1 && mi->num_uses == 1 &&
                mi->operands[0].kind == mi->operands[1].kind &&
                mi->operands[0].reg_num == mi->operands[1].reg_num) {
                /* Convert to NOP — the emitter already handles these */
                mi->op = AMD_PSEUDO_DEF;
                mi->num_defs = 0;
                mi->num_uses = 0;
            }
        }
    }
}

/* ---- Assembly Text Printer ---- */

static void asm_append(amd_module_t *A, const char *fmt, ...)
{
    if (A->asm_len >= AMD_ASM_SIZE - 256) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(A->asm_buf + A->asm_len, AMD_ASM_SIZE - A->asm_len, fmt, ap);
    va_end(ap);
    if (n > 0) A->asm_len += (uint32_t)n;
}

static void print_operand(amd_module_t *A, const moperand_t *op)
{
    switch (op->kind) {
    case MOP_SGPR:
        asm_append(A, "s%u", op->reg_num);
        break;
    case MOP_VGPR:
        asm_append(A, "v%u", op->reg_num);
        break;
    case MOP_VREG_S:
        asm_append(A, "%%vs%u", op->reg_num);
        break;
    case MOP_VREG_V:
        asm_append(A, "%%vv%u", op->reg_num);
        break;
    case MOP_IMM:
        /* Print hex for large values, decimal for small */
        if (op->imm >= -16 && op->imm <= 64)
            asm_append(A, "%d", op->imm);
        else
            asm_append(A, "0x%x", (uint32_t)op->imm);
        break;
    case MOP_LABEL:
        asm_append(A, ".LBB%u", (uint32_t)op->imm);
        break;
    case MOP_SPECIAL:
        switch (op->imm) {
        case AMD_SPEC_VCC:  asm_append(A, "vcc_lo"); break;
        case AMD_SPEC_EXEC: asm_append(A, "exec_lo"); break;
        case AMD_SPEC_SCC:  asm_append(A, "scc"); break;
        case AMD_SPEC_M0:   asm_append(A, "m0"); break;
        default:            asm_append(A, "???"); break;
        }
        break;
    default:
        break;
    }
}

static void print_sgpr_pair(amd_module_t *A, uint16_t base)
{
    asm_append(A, "s[%u:%u]", base, base + 1);
}

static void print_minst(amd_module_t *A, const minst_t *mi)
{
    if (mi->op >= AMD_OP_COUNT) return;
    const amd_enc_entry_t *enc = &amd_enc_table[mi->op];
    if (enc->mnemonic == NULL) return;

    /* Skip pseudo-instructions that survived */
    if (enc->fmt == AMD_FMT_PSEUDO) return;

    asm_append(A, "    %s", enc->mnemonic);

    /* Format-specific operand printing */
    uint8_t total = mi->num_defs + mi->num_uses;
    if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

    switch (enc->fmt) {
    case AMD_FMT_SMEM: {
        /* s_load_dword[x2/x4] sDst, sBase, offset */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            if (mi->op == AMD_S_LOAD_DWORDX2 || mi->op == AMD_S_LOAD_DWORDX4) {
                uint16_t base = mi->operands[0].reg_num;
                uint16_t cnt = (mi->op == AMD_S_LOAD_DWORDX2) ? 2 : 4;
                asm_append(A, "s[%u:%u]", base, base + cnt - 1);
            } else {
                print_operand(A, &mi->operands[0]);
            }
            asm_append(A, ", ");
            /* Base is a pair */
            if (mi->operands[1].kind == MOP_SGPR)
                print_sgpr_pair(A, mi->operands[1].reg_num);
            else
                print_operand(A, &mi->operands[1]);
            if (mi->num_uses > 1) {
                asm_append(A, ", ");
                print_operand(A, &mi->operands[2]);
            }
        }
        break;
    }
    case AMD_FMT_SOPP: {
        /* s_branch target / s_waitcnt encoding / s_endpgm / s_barrier */
        if (mi->op == AMD_S_WAITCNT) {
            uint16_t w = mi->flags;
            int vm = (w & AMD_WAIT_VMCNT0) != 0;
            int lgkm = (w & AMD_WAIT_LGKMCNT0) != 0;
            if (vm && lgkm)
                asm_append(A, " vmcnt(0) lgkmcnt(0)");
            else if (vm)
                asm_append(A, " vmcnt(0)");
            else if (lgkm)
                asm_append(A, " lgkmcnt(0)");
            else
                asm_append(A, " 0x%04x", w);
        } else if (mi->op == AMD_S_WAIT_LOADCNT ||
                   mi->op == AMD_S_WAIT_STORECNT ||
                   mi->op == AMD_S_WAIT_DSCNT ||
                   mi->op == AMD_S_WAIT_KMCNT) {
            asm_append(A, " 0x%x", mi->flags);
        } else if (mi->num_uses > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
        }
        break;
    }
    case AMD_FMT_FLAT_GBL: case AMD_FMT_FLAT_SCR: {
        /* Load:  global_load_dword  vDst, vOffset, sBase|off */
        /* Store: global_store_dword vOffset, vSrc, sBase|off */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
            asm_append(A, ", ");
            print_operand(A, &mi->operands[1]);
            asm_append(A, ", ");
            if (mi->num_uses > 1 && mi->operands[2].kind == MOP_SGPR)
                print_sgpr_pair(A, mi->operands[2].reg_num);
            else
                asm_append(A, "off");
        } else {
            if (mi->num_uses >= 2) {
                asm_append(A, " ");
                print_operand(A, &mi->operands[0]);
                asm_append(A, ", ");
                print_operand(A, &mi->operands[1]);
                asm_append(A, ", ");
                if (mi->num_uses > 2 && mi->operands[2].kind == MOP_SGPR)
                    print_sgpr_pair(A, mi->operands[2].reg_num);
                else
                    asm_append(A, "off");
            }
        }
        if (mi->flags & AMD_FLAG_GLC) asm_append(A, " glc");
        break;
    }
    case AMD_FMT_DS: {
        /* ds_read_b32 vDst, vAddr [, offset] */
        /* ds_write_b32 vAddr, vSrc [, offset] */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
            for (uint8_t k = mi->num_defs; k < total; k++) {
                asm_append(A, ", ");
                print_operand(A, &mi->operands[k]);
            }
        } else {
            for (uint8_t k = 0; k < mi->num_uses && k < MINST_MAX_OPS; k++) {
                if (k > 0) asm_append(A, ",");
                asm_append(A, " ");
                print_operand(A, &mi->operands[k]);
            }
        }
        break;
    }
    case AMD_FMT_SOP1: {
        /* s_mov_b32 sDst, sSrc */
        /* s_setpc_b64 sBase */
        if (mi->op == AMD_S_SETPC_B64 || mi->op == AMD_S_SWAPPC_B64) {
            if (mi->num_uses > 0) {
                asm_append(A, " ");
                if (mi->operands[mi->num_defs].kind == MOP_SGPR)
                    print_sgpr_pair(A, mi->operands[mi->num_defs].reg_num);
                else
                    print_operand(A, &mi->operands[mi->num_defs]);
            }
        } else {
            for (uint8_t k = 0; k < total; k++) {
                if (k > 0) asm_append(A, ",");
                asm_append(A, " ");
                print_operand(A, &mi->operands[k]);
            }
        }
        break;
    }
    default: {
        /* Generic: dst, src0, src1, ... */
        for (uint8_t k = 0; k < total; k++) {
            if (k > 0) asm_append(A, ",");
            asm_append(A, " ");
            print_operand(A, &mi->operands[k]);
        }
        break;
    }
    }

    asm_append(A, "\n");
}

static void emit_asm_function(amd_module_t *A, uint32_t mf_idx)
{
    const mfunc_t *F = &A->mfuncs[mf_idx];
    const char *name = A->bir->strings + F->name;

    if (F->is_kernel) {
        asm_append(A, "    .globl %s\n", name);
        asm_append(A, "    .p2align 8\n");
        asm_append(A, "    .type %s,@function\n", name);
    } else {
        asm_append(A, "    .type %s,@function\n", name);
    }
    asm_append(A, "%s:\n", name);
    asm_append(A, "    ; %u SGPRs, %u VGPRs, %u LDS bytes, %u scratch bytes\n",
               F->num_sgprs, F->num_vgprs, F->lds_bytes, F->scratch_bytes);

    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        const mblock_t *MB = &A->mblocks[mb_idx];

        asm_append(A, ".LBB%u:\n", mb_idx);

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            print_minst(A, &A->minsts[mi_idx]);
        }
    }
    asm_append(A, "\n");
}

void amdgpu_regalloc(amd_module_t *A)
{
    amdgpu_phi_elim(A);
    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++)
        regalloc_function(A, fi);
}

void amdgpu_emit_asm(const amd_module_t *amd, FILE *out)
{
    /* We need to cast away const for the asm buffer operations */
    amd_module_t *A = (amd_module_t *)amd;

    A->asm_len = 0;
    asm_append(A, (A->target == AMD_TARGET_GFX1200)
        ? "    .amdgcn_target \"amdgcn-amd-amdhsa--gfx1200\"\n"
        : "    .amdgcn_target \"amdgcn-amd-amdhsa--gfx1100\"\n");
    asm_append(A, "    .text\n\n");

    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        emit_asm_function(A, fi);
    }

    /* Write to output */
    fwrite(A->asm_buf, 1, A->asm_len, out);
}

/* ---- Binary Instruction Encoding ---- */

/* Inline constant range: 0-64, -1..-16, and special float values */
static int fits_inline(int32_t val)
{
    if (val >= 0 && val <= 64) return 1;
    if (val >= -16 && val <= -1) return 1;
    return 0;
}

/* Encode an SGPR/inline-constant/literal source for scalar formats */
static uint8_t encode_ssrc(const moperand_t *op, uint32_t *literal, int *need_lit)
{
    switch (op->kind) {
    case MOP_SGPR:   return (uint8_t)op->reg_num;
    case MOP_SPECIAL:
        switch (op->imm) {
        case AMD_SPEC_VCC:  return 106;  /* VCC_LO */
        case AMD_SPEC_EXEC: return 126;  /* EXEC_LO */
        case AMD_SPEC_SCC:  return 253;
        case AMD_SPEC_M0:   return 124;
        default: return 0;
        }
    case MOP_IMM:
        if (fits_inline(op->imm)) {
            if (op->imm >= 0) return (uint8_t)(128 + op->imm);
            return (uint8_t)(192 + (-op->imm - 1));
        }
        *literal = (uint32_t)op->imm;
        *need_lit = 1;
        return 255; /* literal constant */
    default:
        return 0;
    }
}

/* Encode source for VOP formats (can be SGPR, VGPR, inline, literal) */
static uint16_t encode_vsrc(const moperand_t *op, uint32_t *literal, int *need_lit)
{
    switch (op->kind) {
    case MOP_VGPR:   return (uint16_t)(256 + op->reg_num);
    case MOP_SGPR:   return (uint16_t)op->reg_num;
    case MOP_SPECIAL:
        switch (op->imm) {
        case AMD_SPEC_VCC:  return 106;
        case AMD_SPEC_EXEC: return 126;
        case AMD_SPEC_SCC:  return 253;
        default: return 0;
        }
    case MOP_IMM:
        if (fits_inline(op->imm)) {
            if (op->imm >= 0) return (uint16_t)(128 + op->imm);
            return (uint16_t)(192 + (-op->imm - 1));
        }
        *literal = (uint32_t)op->imm;
        *need_lit = 1;
        return 255;
    default:
        return 0;
    }
}

static void emit_dword(amd_module_t *A, uint32_t dw)
{
    if (A->code_len + 4 > AMD_CODE_SIZE) return;
    memcpy(A->code + A->code_len, &dw, 4);
    A->code_len += 4;
}

/* Encode SDST field: handles both physical SGPRs and special registers */
static uint8_t encode_sdst(const moperand_t *op)
{
    if (op->kind == MOP_SGPR) return (uint8_t)op->reg_num;
    if (op->kind == MOP_SPECIAL) {
        switch (op->imm) {
        case AMD_SPEC_VCC:  return 106;
        case AMD_SPEC_EXEC: return 126;
        case AMD_SPEC_SCC:  return 253;
        case AMD_SPEC_M0:   return 124;
        default: break;
        }
    }
    return 0;
}

static void encode_sop2(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:30]=10 [29:23]=OP [22:16]=SDST [15:8]=SSRC1 [7:0]=SSRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t sdst = (mi->num_defs > 0) ? encode_sdst(&mi->operands[0]) : 0;
    uint8_t ssrc0 = encode_ssrc(&mi->operands[mi->num_defs], &literal, &need_lit);
    uint8_t ssrc1 = (mi->num_uses > 1) ?
                    encode_ssrc(&mi->operands[mi->num_defs + 1], &literal, &need_lit) : 0;

    uint32_t dw = (2u << 30) | ((uint32_t)(hw_op & 0x7F) << 23) |
                  ((uint32_t)sdst << 16) | ((uint32_t)ssrc1 << 8) | ssrc0;
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_sop1(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:23]=101111101 [22:16]=SDST [15:8]=OP [7:0]=SSRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t sdst = (mi->num_defs > 0) ? encode_sdst(&mi->operands[0]) : 0;
    uint8_t ssrc0 = (mi->num_uses > 0) ?
                    encode_ssrc(&mi->operands[mi->num_defs], &literal, &need_lit) : 0;

    uint32_t dw = 0xBE800000u | ((uint32_t)sdst << 16) |
                  ((uint32_t)(hw_op & 0xFF) << 8) | ssrc0;
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_sopc(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:23]=101111110 [22:16]=OP [15:8]=SSRC1 [7:0]=SSRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t ssrc0 = encode_ssrc(&mi->operands[0], &literal, &need_lit);
    uint8_t ssrc1 = (mi->num_uses > 1) ?
                    encode_ssrc(&mi->operands[1], &literal, &need_lit) : 0;

    uint32_t dw = 0xBF000000u | ((uint32_t)(hw_op & 0x7F) << 16) |
                  ((uint32_t)ssrc1 << 8) | ssrc0;
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_sopp(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:23]=101111111 [22:16]=OP [15:0]=SIMM16 */
    uint16_t simm16 = 0;
    if (mi->op == AMD_S_WAITCNT) {
        /* GFX11 waitcnt SIMM16: [15:10]=vmcnt [9:4]=lgkmcnt [3:0]=expcnt
           Counter at 0 = wait for all; counter at max = don't wait */
        uint16_t vm   = 63;   /* max: don't wait */
        uint16_t lgkm = 63;
        uint16_t exp  = 7;
        if (mi->flags & AMD_WAIT_VMCNT0)   vm   = 0;
        if (mi->flags & AMD_WAIT_LGKMCNT0) lgkm = 0;
        simm16 = (uint16_t)((vm << 10) | (lgkm << 4) | exp);
    } else if (mi->num_uses > 0 && mi->operands[0].kind == MOP_LABEL) {
        /* Branch target: offset will be patched in fixup pass */
        simm16 = (uint16_t)mi->operands[0].imm;
    }

    uint32_t dw = 0xBF800000u | ((uint32_t)(hw_op & 0x7F) << 16) | simm16;
    emit_dword(A, dw);
}

static void encode_smem(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    uint8_t sdata = (mi->num_defs > 0 && mi->operands[0].kind == MOP_SGPR) ?
                    (uint8_t)mi->operands[0].reg_num : 0;
    uint8_t sbase = 0;
    if (mi->num_uses > 0 && mi->operands[mi->num_defs].kind == MOP_SGPR)
        sbase = (uint8_t)(mi->operands[mi->num_defs].reg_num / 2);
    int32_t offset = 0;
    if (mi->num_uses > 1)
        offset = mi->operands[mi->num_defs + 1].imm;

    if (A->target >= AMD_TARGET_GFX1200) {
        /* GFX12 SMEM: 2 dwords
           DW0: [31:26]=111101 [24:23]=TH [22:21]=SCOPE [18:13]=OP(6) [12:6]=SDATA [5:0]=SBASE
           DW1: [31:25]=SOFFSET(0x7C=null) [23:0]=IOFFSET(24-bit) */
        uint32_t dw0 = (0x3Du << 26) | ((uint32_t)(hw_op & 0x3F) << 13) |
                       ((uint32_t)(sdata & 0x7F) << 6) | (uint32_t)(sbase & 0x3F);
        uint32_t dw1 = (0x7Cu << 25) | ((uint32_t)offset & 0xFFFFFF);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    } else {
        /* GFX11 SMEM: 2 dwords
           DW0: [31:26]=111101 [25:18]=OP(8) [12:6]=SDATA [5:0]=SBASE
           DW1: [31:25]=SOFFSET(0x7C=null) [20:0]=OFFSET(21-bit) */
        uint32_t dw0 = (0x3Du << 26) | ((uint32_t)(hw_op & 0xFF) << 18) |
                       ((uint32_t)(sdata & 0x7F) << 6) | (uint32_t)(sbase & 0x3F);
        uint32_t dw1 = (0x7Cu << 25) | ((uint32_t)offset & 0x1FFFFF);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    }
}

static void encode_vop1(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:25]=0111111 [24:17]=VDST [16:9]=OP [8:0]=SRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t vdst = (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR) ?
                   (uint8_t)mi->operands[0].reg_num : 0;
    uint16_t src0 = (mi->num_uses > 0) ?
                    encode_vsrc(&mi->operands[mi->num_defs], &literal, &need_lit) : 0;

    uint32_t dw = (0x3Fu << 25) | ((uint32_t)vdst << 17) |
                  ((uint32_t)(hw_op & 0xFF) << 9) | (src0 & 0x1FF);
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_vop2(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31]=0 [30:25]=OP [24:17]=VDST [16:9]=VSRC1 [8:0]=SRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t vdst = (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR) ?
                   (uint8_t)mi->operands[0].reg_num : 0;
    uint16_t src0 = encode_vsrc(&mi->operands[mi->num_defs], &literal, &need_lit);
    uint8_t vsrc1 = 0;
    if (mi->num_uses > 1 && mi->operands[mi->num_defs + 1].kind == MOP_VGPR)
        vsrc1 = (uint8_t)mi->operands[mi->num_defs + 1].reg_num;

    uint32_t dw = ((uint32_t)(hw_op & 0x3F) << 25) | ((uint32_t)vdst << 17) |
                  ((uint32_t)vsrc1 << 9) | (src0 & 0x1FF);
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_vop3(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* VOP3: 2 dwords
       DW0: [31:26]=110100 [25:16]=OP [15:8]=VDST [7:0]=ABS/CLAMP
       DW1: [31]=0 [30:29]=OMOD [28:27]=NEG [26:18]=SRC2 [17:9]=SRC1 [8:0]=SRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t vdst = (mi->num_defs > 0) ?
                   (uint8_t)mi->operands[0].reg_num : 0;
    uint16_t src0 = (mi->num_uses > 0) ?
                    encode_vsrc(&mi->operands[mi->num_defs], &literal, &need_lit) : 0;
    uint16_t src1 = (mi->num_uses > 1) ?
                    encode_vsrc(&mi->operands[mi->num_defs + 1], &literal, &need_lit) : 0;
    uint16_t src2 = (mi->num_uses > 2) ?
                    encode_vsrc(&mi->operands[mi->num_defs + 2], &literal, &need_lit) : 0;

    uint32_t dw0 = (0x35u << 26) | ((uint32_t)(hw_op & 0x3FF) << 16) |
                   (uint32_t)vdst;
    uint32_t dw1 = ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF);
    emit_dword(A, dw0);
    emit_dword(A, dw1);
}

static void encode_vopc(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:25]=0111110 [24:17]=OP [16:9]=VSRC1 [8:0]=SRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint16_t src0 = (mi->num_uses > 0) ?
                    encode_vsrc(&mi->operands[0], &literal, &need_lit) : 0;
    uint8_t vsrc1 = 0;
    if (mi->num_uses > 1 && mi->operands[1].kind == MOP_VGPR)
        vsrc1 = (uint8_t)mi->operands[1].reg_num;

    uint32_t dw = (0x3Eu << 25) | ((uint32_t)(hw_op & 0xFF) << 17) |
                  ((uint32_t)vsrc1 << 9) | (src0 & 0x1FF);
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_ds(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* DS: 2 dwords
       DW0: [31:26]=110110 [25:18]=OP [17:16]=GDS [15:0]=OFFSET
       DW1: [31:24]=VDST [23:16]=DATA1 [15:8]=DATA0 [7:0]=ADDR */
    uint8_t vdst = 0, addr = 0, data0 = 0;
    if (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR)
        vdst = (uint8_t)mi->operands[0].reg_num;
    /* First use operand is addr */
    uint8_t use_base = mi->num_defs;
    if (mi->num_uses > 0 && mi->operands[use_base].kind == MOP_VGPR)
        addr = (uint8_t)mi->operands[use_base].reg_num;
    if (mi->num_uses > 1 && mi->operands[use_base + 1].kind == MOP_VGPR)
        data0 = (uint8_t)mi->operands[use_base + 1].reg_num;

    uint16_t offset = 0;
    /* Check for immediate offset in last use operand */
    uint8_t last_use = (uint8_t)(use_base + mi->num_uses - 1);
    if (mi->num_uses > 0 && last_use < MINST_MAX_OPS &&
        mi->operands[last_use].kind == MOP_IMM)
        offset = (uint16_t)mi->operands[last_use].imm;

    uint32_t dw0 = (0x36u << 26) | ((uint32_t)(hw_op & 0xFF) << 18) | offset;
    uint32_t dw1 = ((uint32_t)vdst << 24) | ((uint32_t)data0 << 8) | addr;
    emit_dword(A, dw0);
    emit_dword(A, dw1);
}

static void encode_flat_global(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    uint8_t vdst = 0, addr = 0, data = 0;
    int32_t offset = 0;
    int is_scratch = (amd_enc_table[mi->op].fmt == AMD_FMT_FLAT_SCR);

    /* Extract def: VDST */
    if (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR)
        vdst = (uint8_t)mi->operands[0].reg_num;

    /* Default null SADDR */
    uint32_t saddr = 0x7C; /* sgpr_null */

    /* Walk use operands: 1st VGPR=addr, 2nd VGPR=data, SGPR=saddr, IMM=offset */
    uint8_t use_base = mi->num_defs;
    int got_addr = 0;
    for (uint8_t k = use_base; k < (uint8_t)(use_base + mi->num_uses) && k < MINST_MAX_OPS; k++) {
        const moperand_t *op = &mi->operands[k];
        if (op->kind == MOP_VGPR) {
            if (!got_addr) { addr = (uint8_t)op->reg_num; got_addr = 1; }
            else { data = (uint8_t)op->reg_num; }
        } else if (op->kind == MOP_SGPR) {
            saddr = op->reg_num;
        } else if (op->kind == MOP_IMM) {
            offset = op->imm;
        }
    }

    if (A->target >= AMD_TARGET_GFX1200) {
        /* GFX12 FLAT/GLOBAL/SCRATCH: 3 dwords (96-bit)
           DW0: [31:24]=ENCODING [21:14]=OP [7]=NV [6:0]=SADDR
           DW1: [30:23]=VSRC [22:20]=TH [19:18]=SCOPE [17]=SVE [7:0]=VDST
           DW2: [31:8]=IOFFSET(24-bit) [7:0]=VADDR
           Encoding: global=0xEE, scratch=0xED
           (PDF says OP[20:13] but AMD ISA XML has OP[21:14]) */
        uint8_t enc = is_scratch ? 0xED : 0xEE;
        uint8_t sve = is_scratch ? 1 : 0;

        uint32_t dw0 = ((uint32_t)enc << 24) |
                       ((uint32_t)(hw_op & 0xFF) << 14) |
                       (saddr & 0x7F);
        uint32_t dw1 = ((uint32_t)data << 23) |
                       ((uint32_t)sve << 17) |
                       (uint32_t)vdst;
        uint32_t dw2 = (((uint32_t)offset & 0xFFFFFF) << 8) |
                       (uint32_t)addr;

        if (mi->flags & AMD_FLAG_GLC) dw1 |= (1u << 20); /* TH[0] */

        emit_dword(A, dw0);
        emit_dword(A, dw1);
        emit_dword(A, dw2);
    } else {
        /* GFX11 FLAT/GLOBAL/SCRATCH: 2 dwords (64-bit)
           DW0: [31:26]=110111 [25:18]=OP [17:16]=SEG [15:14]=flags [13:0]=OFFSET
           DW1: [31:24]=VDST [23:16]=SADDR [15:8]=DATA [7:0]=ADDR */
        uint8_t seg = is_scratch ? 1 : 2;
        if (is_scratch && saddr == 0x7C) saddr = 0xFC; /* GFX11 scratch null */

        uint32_t off_lo = (uint32_t)offset & 0x1FFF;
        uint32_t dw0 = (0x37u << 26) | ((uint32_t)(hw_op & 0xFF) << 18) |
                       ((uint32_t)seg << 16) | off_lo;
        uint32_t dw1 = ((uint32_t)vdst << 24) | ((saddr & 0xFF) << 16) |
                       ((uint32_t)data << 8) | addr;
        if (mi->flags & AMD_FLAG_GLC) dw0 |= (1u << 14);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    }
}

/* Instruction offsets for branch fixup */
static uint32_t inst_offsets[AMD_MAX_MINSTS];
static uint32_t block_offsets[AMD_MAX_MBLOCKS];

static void encode_function(amd_module_t *A, uint32_t mf_idx)
{
    const mfunc_t *F = &A->mfuncs[mf_idx];

    /* Pass 1: compute instruction byte offsets */
    uint32_t offset = A->code_len;
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        if (mb_idx >= AMD_MAX_MBLOCKS) break;
        const mblock_t *MB = &A->mblocks[mb_idx];
        block_offsets[mb_idx] = offset;

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            if (mi_idx >= AMD_MAX_MINSTS) break;
            const minst_t *mi = &A->minsts[mi_idx];
            inst_offsets[mi_idx] = offset;

            if (mi->op >= AMD_OP_COUNT) { offset += 4; continue; }
            const amd_enc_entry_t *enc = &amd_enc_table[mi->op];

            switch (enc->fmt) {
            case AMD_FMT_SOP2: case AMD_FMT_SOP1: case AMD_FMT_SOPC:
            case AMD_FMT_SOPP: case AMD_FMT_VOP1: case AMD_FMT_VOP2:
            case AMD_FMT_VOPC:
                offset += 4;
                /* Check if a literal constant is needed */
                for (uint8_t k = 0; k < mi->num_defs + mi->num_uses && k < MINST_MAX_OPS; k++) {
                    if (mi->operands[k].kind == MOP_IMM && !fits_inline(mi->operands[k].imm)) {
                        offset += 4;
                        break;
                    }
                }
                break;
            case AMD_FMT_SMEM: case AMD_FMT_VOP3: case AMD_FMT_DS:
                offset += 8;
                break;
            case AMD_FMT_FLAT_GBL: case AMD_FMT_FLAT_SCR:
                offset += (A->target >= AMD_TARGET_GFX1200) ? 12 : 8;
                break;
            case AMD_FMT_PSEUDO:
                break; /* no bytes */
            default:
                offset += 4;
                break;
            }
        }
    }

    /* Pass 2: encode with branch fixup */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        if (mb_idx >= AMD_MAX_MBLOCKS) break;
        const mblock_t *MB = &A->mblocks[mb_idx];

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            if (mi_idx >= AMD_MAX_MINSTS) break;
            minst_t *mi = &A->minsts[mi_idx]; /* non-const for branch fixup */

            if (mi->op >= AMD_OP_COUNT) continue;
            const amd_enc_entry_t *enc = &amd_enc_table[mi->op];

            /* Fix up branch targets: convert block index to PC-relative offset */
            if (enc->fmt == AMD_FMT_SOPP && mi->num_uses > 0 &&
                mi->operands[0].kind == MOP_LABEL) {
                uint32_t target_mb = (uint32_t)mi->operands[0].imm;
                if (target_mb < AMD_MAX_MBLOCKS) {
                    int32_t pc_offset = (int32_t)(block_offsets[target_mb] -
                                                  (inst_offsets[mi_idx] + 4));
                    pc_offset /= 4; /* SOPP offset is in dwords */
                    mi->operands[0].imm = pc_offset;
                }
            }

            switch (enc->fmt) {
            case AMD_FMT_SOP2: encode_sop2(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SOP1: encode_sop1(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SOPC: encode_sopc(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SOPP: encode_sopp(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SMEM: encode_smem(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOP1: encode_vop1(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOP2: encode_vop2(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOP3: encode_vop3(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOPC: encode_vopc(A, mi, enc->hw_opcode); break;
            case AMD_FMT_DS:   encode_ds(A, mi, enc->hw_opcode);   break;
            case AMD_FMT_FLAT_GBL:
            case AMD_FMT_FLAT_SCR:
                encode_flat_global(A, mi, enc->hw_opcode);
                break;
            case AMD_FMT_PSEUDO:
                break;
            default:
                emit_dword(A, 0); /* placeholder */
                break;
            }
        }
    }
}

/* ---- Msgpack Encoder (minimal, bounded) ---- */

#define MP_BUF_MAX 8192

static void mp_write(uint8_t *buf, uint32_t *pos, const void *data, uint32_t len)
{
    if (*pos + len > MP_BUF_MAX) return;
    memcpy(buf + *pos, data, len);
    *pos += len;
}

static void mp_fixmap(uint8_t *buf, uint32_t *pos, uint8_t count)
{
    if (*pos >= MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0x80 | count);
}

static void mp_fixarray(uint8_t *buf, uint32_t *pos, uint8_t count)
{
    if (*pos >= MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0x90 | count);
}

static void mp_fixstr(uint8_t *buf, uint32_t *pos, const char *s)
{
    uint8_t len = (uint8_t)strlen(s);
    if (len > 31) len = 31;
    if (*pos + 1 + len > MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0xA0 | len);
    mp_write(buf, pos, s, len);
}

static void mp_str(uint8_t *buf, uint32_t *pos, const char *s)
{
    uint32_t len = (uint32_t)strlen(s);
    if (len <= 31) {
        mp_fixstr(buf, pos, s);
    } else if (len <= 255) {
        if (*pos + 2 + len > MP_BUF_MAX) return;
        buf[(*pos)++] = 0xD9;
        buf[(*pos)++] = (uint8_t)len;
        mp_write(buf, pos, s, len);
    } else {
        if (*pos + 3 + len > MP_BUF_MAX) return;
        buf[(*pos)++] = 0xDA;
        buf[(*pos)++] = (uint8_t)(len >> 8);
        buf[(*pos)++] = (uint8_t)(len);
        mp_write(buf, pos, s, len);
    }
}

static void mp_uint(uint8_t *buf, uint32_t *pos, uint32_t val)
{
    if (*pos >= MP_BUF_MAX - 5) return;  /* worst case: 5 bytes */
    if (val <= 127) {
        buf[(*pos)++] = (uint8_t)val;
    } else if (val <= 0xFF) {
        buf[(*pos)++] = 0xCC;
        buf[(*pos)++] = (uint8_t)val;
    } else if (val <= 0xFFFF) {
        buf[(*pos)++] = 0xCD;
        buf[(*pos)++] = (uint8_t)(val >> 8);
        buf[(*pos)++] = (uint8_t)val;
    } else {
        buf[(*pos)++] = 0xCE;
        buf[(*pos)++] = (uint8_t)(val >> 24);
        buf[(*pos)++] = (uint8_t)(val >> 16);
        buf[(*pos)++] = (uint8_t)(val >> 8);
        buf[(*pos)++] = (uint8_t)val;
    }
}

/* ---- ELF Code Object Writer ---- */

/* ELF64 types */
typedef struct {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} elf64_ehdr_t;   /* 64 bytes */

typedef struct {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} elf64_shdr_t;   /* 64 bytes */

typedef struct {
    uint32_t st_name;
    uint8_t  st_info;
    uint8_t  st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} elf64_sym_t;    /* 24 bytes */

typedef struct {
    uint32_t n_namesz;
    uint32_t n_descsz;
    uint32_t n_type;
} elf64_nhdr_t;   /* 12 bytes */

#define SHT_NULL     0
#define SHT_PROGBITS 1
#define SHT_SYMTAB   2
#define SHT_STRTAB   3
#define SHT_NOTE     7
#define SHF_ALLOC    2
#define SHF_EXECINSTR 4
#define STB_GLOBAL   1
#define STT_FUNC     2
#define STT_OBJECT   1
#define NT_AMDGPU_METADATA 32

/* Pad file to alignment boundary (bounded to avoid infinite loops) */
static void fwrite_pad(FILE *fp, uint32_t align)
{
    long pos = ftell(fp);
    if (pos < 0) return;
    uint32_t pad = ((uint32_t)pos + align - 1) & ~(align - 1);
    uint32_t n = pad - (uint32_t)pos;
    if (n > 256) n = 256; /* sanity cap */
    for (uint32_t i = 0; i < n; i++)
        fputc(0, fp);
}

int amdgpu_emit_elf(amd_module_t *A, const char *path)
{
    /* First, encode all functions to binary */
    A->code_len = 0;

    /* Track kernel descriptor positions */
    static uint32_t kd_offsets[AMD_MAX_MFUNCS];
    static uint32_t code_offsets[AMD_MAX_MFUNCS];
    uint32_t num_kernels = 0;

    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;

        /* Align to 256 bytes for kernel descriptor */
        for (uint32_t pad = 0; A->code_len % 256 != 0 && pad < 64; pad++)
            emit_dword(A, 0);
        kd_offsets[num_kernels] = A->code_len;

        /* Write kernel descriptor (64 bytes) */
        amd_kernel_descriptor_t kd;
        memset(&kd, 0, sizeof(kd));

        mfunc_t *F = &A->mfuncs[fi];
        kd.group_segment_fixed_size = F->lds_bytes;
        kd.private_segment_fixed_size = F->scratch_bytes;
        kd.kernarg_size = F->kernarg_bytes;
        kd.kernel_code_entry_byte_offset = 256; /* descriptor is 64 bytes, padded to 256 */

        /* compute_pgm_rsrc1 */
        uint32_t vgpr_blocks = (F->num_vgprs > 0) ? (uint32_t)((F->num_vgprs + 7) / 8 - 1) : 0;
        uint32_t sgpr_blocks = (F->num_sgprs > 0) ? (uint32_t)((F->num_sgprs + 7) / 8 - 1) : 0;
        kd.compute_pgm_rsrc1 = (vgpr_blocks & 0x3F) |
                               ((sgpr_blocks & 0xF) << 6) |
                               (1u << 20) |  /* IEEE_MODE */
                               (1u << 26) |  /* WGP_MODE */
                               (1u << 27);   /* MEM_ORDERED */

        /* compute_pgm_rsrc2 */
        kd.compute_pgm_rsrc2 = ((F->scratch_bytes > 0) ? 1u : 0u) | /* SCRATCH_EN */
                               (4u << 1) |   /* USER_SGPR_COUNT = 4 (dispatch_ptr + kernarg_ptr) */
                               (1u << 7) |   /* TGID_X_EN */
                               (1u << 8) |   /* TGID_Y_EN */
                               (1u << 9);    /* TGID_Z_EN */

        /* kernel_code_properties */
        kd.kernel_code_properties = (1u << 0) |  /* ENABLE_SGPR_DISPATCH_PTR */
                                    (1u << 1);   /* ENABLE_SGPR_KERNARG_PTR */

        /* Write the 64 bytes */
        if (A->code_len + 64 <= AMD_CODE_SIZE) {
            memcpy(A->code + A->code_len, &kd, 64);
            A->code_len += 64;
        }

        /* Pad to 256 bytes after descriptor */
        for (uint32_t pad = 0; A->code_len % 256 != 0 && A->code_len < AMD_CODE_SIZE && pad < 256; pad++)
            A->code[A->code_len++] = 0;

        code_offsets[num_kernels] = A->code_len;
        num_kernels++;

        /* Encode the function's instructions */
        encode_function(A, fi);
    }

    /* Also encode device functions */
    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        if (A->mfuncs[fi].is_kernel) continue;
        encode_function(A, fi);
    }

    /* Build note section (msgpack metadata) */
    static uint8_t note_buf[16384];
    uint32_t note_len = 0;

    /* Note header: name = "AMDGPU", type = NT_AMDGPU_METADATA */
    const char *note_name = "AMDGPU\0\0"; /* 8 bytes aligned */
    uint32_t note_name_len = 7; /* including null */

    /* Build msgpack payload */
    static uint8_t mp_buf[8192];
    uint32_t mp_pos = 0;

    mp_fixmap(mp_buf, &mp_pos, 3);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.version");
    mp_fixarray(mp_buf, &mp_pos, 2);
    mp_uint(mp_buf, &mp_pos, 1);
    mp_uint(mp_buf, &mp_pos, 2);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.target");
    mp_str(mp_buf, &mp_pos, "amdgcn-amd-amdhsa--gfx1100");

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.kernels");
    uint8_t nk = (num_kernels > 15) ? 15 : (uint8_t)num_kernels;
    mp_fixarray(mp_buf, &mp_pos, nk);

    uint32_t ki = 0;
    for (uint32_t fi = 0; fi < A->num_mfuncs && ki < nk; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;
        mfunc_t *F = &A->mfuncs[fi];
        const char *name = A->bir->strings + F->name;

        /* Build symbol name: "name.kd" */
        char kd_name[256];
        snprintf(kd_name, sizeof(kd_name), "%s.kd", name);

        mp_fixmap(mp_buf, &mp_pos, 10);

        mp_fixstr(mp_buf, &mp_pos, ".name");
        mp_str(mp_buf, &mp_pos, name);

        mp_fixstr(mp_buf, &mp_pos, ".symbol");
        mp_str(mp_buf, &mp_pos, kd_name);

        mp_str(mp_buf, &mp_pos, ".kernarg_segment_size");
        mp_uint(mp_buf, &mp_pos, F->kernarg_bytes);

        mp_str(mp_buf, &mp_pos, ".group_segment_fixed_size");
        mp_uint(mp_buf, &mp_pos, F->lds_bytes);

        mp_str(mp_buf, &mp_pos, ".private_segment_fixed_size");
        mp_uint(mp_buf, &mp_pos, F->scratch_bytes);

        mp_str(mp_buf, &mp_pos, ".wavefront_size");
        mp_uint(mp_buf, &mp_pos, F->wavefront_size);

        mp_fixstr(mp_buf, &mp_pos, ".sgpr_count");
        mp_uint(mp_buf, &mp_pos, F->num_sgprs);

        mp_fixstr(mp_buf, &mp_pos, ".vgpr_count");
        mp_uint(mp_buf, &mp_pos, F->num_vgprs);

        mp_str(mp_buf, &mp_pos, ".max_flat_workgroup_size");
        mp_uint(mp_buf, &mp_pos, 256);

        mp_fixstr(mp_buf, &mp_pos, ".args");
        mp_fixarray(mp_buf, &mp_pos, 0); /* empty args for now */

        ki++;
    }

    /* Assemble note section */
    elf64_nhdr_t nhdr;
    nhdr.n_namesz = note_name_len;
    nhdr.n_descsz = mp_pos;
    nhdr.n_type = NT_AMDGPU_METADATA;
    memcpy(note_buf + note_len, &nhdr, 12);
    note_len += 12;
    memcpy(note_buf + note_len, note_name, 8); /* padded to 4-byte align */
    note_len += 8;
    memcpy(note_buf + note_len, mp_buf, mp_pos);
    note_len += mp_pos;
    /* Pad to 4 bytes */
    while (note_len % 4 != 0 && note_len < sizeof(note_buf))
        note_buf[note_len++] = 0;

    /* Build string tables */
    /* .shstrtab: section names (total < 50 bytes, 256 is generous) */
    #define SHSTRTAB_MAX 256
    static char shstrtab[SHSTRTAB_MAX];
    uint32_t shstrtab_len = 0;
    shstrtab[shstrtab_len++] = '\0';
    uint32_t text_name_off = shstrtab_len;
    if (shstrtab_len + 6 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".text", 6); shstrtab_len += 6; }
    uint32_t note_name_off = shstrtab_len;
    if (shstrtab_len + 6 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".note", 6); shstrtab_len += 6; }
    uint32_t symtab_name_off = shstrtab_len;
    if (shstrtab_len + 8 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".symtab", 8); shstrtab_len += 8; }
    uint32_t strtab_name_off = shstrtab_len;
    if (shstrtab_len + 8 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".strtab", 8); shstrtab_len += 8; }
    uint32_t shstrtab_name_off = shstrtab_len;
    if (shstrtab_len + 10 <= SHSTRTAB_MAX) { memcpy(shstrtab + shstrtab_len, ".shstrtab", 10); shstrtab_len += 10; }

    /* .strtab: symbol names */
    #define STRTAB_MAX 4096
    static char strtab[STRTAB_MAX];
    uint32_t strtab_len = 0;
    strtab[strtab_len++] = '\0';

    /* Build symbol table */
    static elf64_sym_t symtab[256];
    uint32_t num_syms = 1; /* null symbol at index 0 */
    memset(&symtab[0], 0, sizeof(elf64_sym_t));

    ki = 0;
    for (uint32_t fi = 0; fi < A->num_mfuncs && ki < num_kernels; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;
        const char *name = A->bir->strings + A->mfuncs[fi].name;

        /* Kernel descriptor symbol (STT_OBJECT) */
        char kd_sym[256];
        snprintf(kd_sym, sizeof(kd_sym), "%s.kd", name);
        uint32_t kd_name_idx = strtab_len;
        uint32_t kd_sym_len = (uint32_t)strlen(kd_sym) + 1;
        if (strtab_len + kd_sym_len <= STRTAB_MAX) {
            memcpy(strtab + strtab_len, kd_sym, kd_sym_len);
            strtab_len += kd_sym_len;
        }

        if (num_syms < 256) {
            symtab[num_syms].st_name = kd_name_idx;
            symtab[num_syms].st_info = (STB_GLOBAL << 4) | STT_OBJECT;
            symtab[num_syms].st_other = 0;
            symtab[num_syms].st_shndx = 1; /* .text section */
            symtab[num_syms].st_value = kd_offsets[ki];
            symtab[num_syms].st_size = 64;
            num_syms++;
        }

        /* Function symbol (STT_FUNC) */
        uint32_t func_name_idx = strtab_len;
        uint32_t name_len = (uint32_t)strlen(name) + 1;
        if (strtab_len + name_len <= STRTAB_MAX) {
            memcpy(strtab + strtab_len, name, name_len);
            strtab_len += name_len;
        }

        if (num_syms < 256) {
            symtab[num_syms].st_name = func_name_idx;
            symtab[num_syms].st_info = (STB_GLOBAL << 4) | STT_FUNC;
            symtab[num_syms].st_other = 0;
            symtab[num_syms].st_shndx = 1;
            symtab[num_syms].st_value = code_offsets[ki];
            symtab[num_syms].st_size = A->code_len - code_offsets[ki];
            num_syms++;
        }

        ki++;
    }

    /* Write ELF file */
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }

    /* Compute layout */
    uint64_t text_off = 64; /* after ELF header */
    uint64_t text_size = A->code_len;
    uint64_t note_off = text_off + text_size;
    /* Align note to 4 bytes */
    note_off = (note_off + 3) & ~3ULL;
    uint64_t symtab_off = note_off + note_len;
    symtab_off = (symtab_off + 7) & ~7ULL;
    uint64_t symtab_size = num_syms * 24;
    uint64_t strtab_off = symtab_off + symtab_size;
    uint64_t shstrtab_off = strtab_off + strtab_len;
    uint64_t shdr_off = (shstrtab_off + shstrtab_len + 7) & ~7ULL;

    /* Sections: 0=NULL, 1=.text, 2=.note, 3=.symtab, 4=.strtab, 5=.shstrtab */
    uint16_t num_sections = 6;

    /* ELF header */
    elf64_ehdr_t ehdr;
    memset(&ehdr, 0, sizeof(ehdr));
    ehdr.e_ident[0] = 0x7F;
    ehdr.e_ident[1] = 'E';
    ehdr.e_ident[2] = 'L';
    ehdr.e_ident[3] = 'F';
    ehdr.e_ident[4] = 2;    /* ELFCLASS64 */
    ehdr.e_ident[5] = 1;    /* ELFDATA2LSB */
    ehdr.e_ident[6] = 1;    /* EV_CURRENT */
    ehdr.e_ident[7] = ELFOSABI_AMDGPU_HSA;
    ehdr.e_type = 2;        /* ET_EXEC (shared object would be 3) */
    ehdr.e_machine = EM_AMDGPU;
    ehdr.e_version = 1;
    ehdr.e_entry = 0;
    ehdr.e_phoff = 0;
    ehdr.e_shoff = shdr_off;
    ehdr.e_flags = (A->target == AMD_TARGET_GFX1200)
                   ? EF_AMDGPU_MACH_AMDGCN_GFX1200
                   : EF_AMDGPU_MACH_AMDGCN_GFX1100;
    ehdr.e_ehsize = 64;
    ehdr.e_phentsize = 0;
    ehdr.e_phnum = 0;
    ehdr.e_shentsize = 64;
    ehdr.e_shnum = num_sections;
    ehdr.e_shstrndx = 5;    /* .shstrtab index */
    fwrite(&ehdr, 1, 64, fp);

    /* .text section */
    fwrite(A->code, 1, A->code_len, fp);
    fwrite_pad(fp, 4);

    /* .note section */
    fwrite(note_buf, 1, note_len, fp);
    fwrite_pad(fp, 8);

    /* .symtab */
    fwrite(symtab, 24, num_syms, fp);

    /* .strtab */
    fwrite(strtab, 1, strtab_len, fp);

    /* .shstrtab */
    fwrite(shstrtab, 1, shstrtab_len, fp);
    fwrite_pad(fp, 8);

    /* Section header table */
    elf64_shdr_t shdrs[6];
    memset(shdrs, 0, sizeof(shdrs));

    /* 0: NULL */
    /* 1: .text */
    shdrs[1].sh_name = text_name_off;
    shdrs[1].sh_type = SHT_PROGBITS;
    shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
    shdrs[1].sh_offset = text_off;
    shdrs[1].sh_size = text_size;
    shdrs[1].sh_addralign = 256;

    /* 2: .note */
    shdrs[2].sh_name = note_name_off;
    shdrs[2].sh_type = SHT_NOTE;
    shdrs[2].sh_offset = note_off;
    shdrs[2].sh_size = note_len;
    shdrs[2].sh_addralign = 4;

    /* 3: .symtab */
    shdrs[3].sh_name = symtab_name_off;
    shdrs[3].sh_type = SHT_SYMTAB;
    shdrs[3].sh_offset = symtab_off;
    shdrs[3].sh_size = symtab_size;
    shdrs[3].sh_link = 4;    /* .strtab */
    shdrs[3].sh_info = 1;    /* first non-local symbol */
    shdrs[3].sh_addralign = 8;
    shdrs[3].sh_entsize = 24;

    /* 4: .strtab */
    shdrs[4].sh_name = strtab_name_off;
    shdrs[4].sh_type = SHT_STRTAB;
    shdrs[4].sh_offset = strtab_off;
    shdrs[4].sh_size = strtab_len;
    shdrs[4].sh_addralign = 1;

    /* 5: .shstrtab */
    shdrs[5].sh_name = shstrtab_name_off;
    shdrs[5].sh_type = SHT_STRTAB;
    shdrs[5].sh_offset = shstrtab_off;
    shdrs[5].sh_size = shstrtab_len;
    shdrs[5].sh_addralign = 1;

    fwrite(shdrs, 64, num_sections, fp);

    fclose(fp);

    fprintf(stderr, "wrote %s (%u bytes code, %u kernels)\n",
            path, A->code_len, num_kernels);
    return BC_OK;
}
