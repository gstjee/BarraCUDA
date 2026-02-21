#include "tensix.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* Thread coarsening + host code generation. Translates "one thread per element"
 * into one core, many tiles, 32 lanes doing the actual work. */

typedef struct {
    uint32_t    tid[3];         /* BIR_THREAD_ID uses per dimension */
    uint32_t    bid[3];         /* BIR_BLOCK_ID uses per dimension */
    uint32_t    bdim[3];        /* BIR_BLOCK_DIM uses per dimension */
    uint32_t    gdim[3];        /* BIR_GRID_DIM uses per dimension */
    uint32_t    num_loads;
    uint32_t    num_stores;
    uint32_t    num_barriers;
    uint32_t    num_atomics;
    uint32_t    num_shuffles;
} coarsen_stats_t;

static void analyse_function(const bir_module_t *bir, uint32_t func_idx,
                              coarsen_stats_t *stats)
{
    memset(stats, 0, sizeof(*stats));

    const bir_func_t *F = &bir->funcs[func_idx];
    int guard = 262144;
    for (uint32_t bi = 0; bi < F->num_blocks && guard > 0; bi++, guard--) {
        const bir_block_t *B = &bir->blocks[F->first_block + bi];

        for (uint32_t ii = 0; ii < B->num_insts && guard > 0; ii++, guard--) {
            const bir_inst_t *I = &bir->insts[B->first_inst + ii];
            uint32_t dim = I->subop < 3 ? I->subop : 0;

            switch (I->op) {
            case BIR_THREAD_ID:     stats->tid[dim]++;  break;
            case BIR_BLOCK_ID:      stats->bid[dim]++;  break;
            case BIR_BLOCK_DIM:     stats->bdim[dim]++; break;
            case BIR_GRID_DIM:      stats->gdim[dim]++; break;
            case BIR_LOAD:          stats->num_loads++;  break;
            case BIR_STORE:         stats->num_stores++; break;
            case BIR_BARRIER:
            case BIR_BARRIER_GROUP: stats->num_barriers++; break;

            case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB:
            case BIR_ATOMIC_AND: case BIR_ATOMIC_OR: case BIR_ATOMIC_XOR:
            case BIR_ATOMIC_MIN: case BIR_ATOMIC_MAX:
            case BIR_ATOMIC_XCHG: case BIR_ATOMIC_CAS:
                stats->num_atomics++; break;

            case BIR_SHFL: case BIR_SHFL_UP:
            case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
            case BIR_BALLOT: case BIR_VOTE_ANY: case BIR_VOTE_ALL:
                stats->num_shuffles++; break;

            default: break;
            }
        }
    }
}

static const char *pattern_names[] = {
    "generic", "element-wise", "reduction"
};

static tt_coarsen_pattern_t classify_pattern(const coarsen_stats_t *stats)
{
    /* Reduction: barriers, atomics, or warp shuffles */
    if (stats->num_barriers > 0 || stats->num_atomics > 0 ||
        stats->num_shuffles > 0)
        return TT_PATTERN_REDUCTION;

    /* Element-wise: 1D thread indexing (tid.x + bid.x * bdim.x), no sync */
    if (stats->tid[0] > 0 && stats->tid[1] == 0 && stats->tid[2] == 0 &&
        stats->bid[1] == 0 && stats->bid[2] == 0)
        return TT_PATTERN_ELEMENTWISE;

    return TT_PATTERN_GENERIC;
}

/* ---- Grid Dimension Mapping ---- */

static void compute_grid_mapping(tt_mfunc_t *MF,
                                  const coarsen_stats_t *stats)
{
    if (MF->launch_bounds_max > 0) {
        /* launch_bounds_max = threads per block. Each Tensix tile = 32 lanes.
         * One block maps to one core, needing this many SFPU passes: */
        MF->tiles_per_core = (MF->launch_bounds_max + TT_SFPU_LANES - 1)
                             / TT_SFPU_LANES;
        if (MF->tiles_per_core < 1) MF->tiles_per_core = 1;
    } else {
        /* No launch bounds — runtime decides via get_arg_val<uint32_t>(0) */
        MF->tiles_per_core = 1;
    }

    (void)stats;
}

/* ---- Public API: Coarsening Analysis ---- */

void tensix_coarsen(tt_module_t *tt)
{
    int guard = 8192;
    for (uint32_t fi = 0; fi < tt->num_mfuncs && guard > 0; fi++, guard--) {
        tt_mfunc_t *MF = &tt->mfuncs[fi];
        if (!MF->is_kernel) continue;
        if (MF->bir_func >= tt->bir->num_funcs) continue;

        coarsen_stats_t stats;
        analyse_function(tt->bir, MF->bir_func, &stats);

        tt_coarsen_pattern_t pattern = classify_pattern(&stats);
        MF->coarsen_pattern = (uint8_t)pattern;

        compute_grid_mapping(MF, &stats);

        const char *name = tt->bir->strings + MF->name;
        printf("  coarsen %s: %s, %u tiles/core, "
               "%u loads, %u stores, %u barriers\n",
               name,
               (unsigned)pattern < TT_PATTERN_COUNT
                   ? pattern_names[pattern] : "???",
               MF->tiles_per_core,
               stats.num_loads, stats.num_stores, stats.num_barriers);
    }
}

/* ---- Host Program Generation ---- */

static void host_write(FILE *fp, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
}

int tensix_emit_host(tt_module_t *tt, const char *host_path,
                     const char *compute_path)
{
    FILE *fp = fopen(host_path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", host_path);
        return BC_ERR_IO;
    }

    host_write(fp,
        "/* Generated by BarraCUDA — Tensix host program skeleton. */\n"
        "/* Adjust buffer sizes, data formats, and reader/writer kernels. */\n"
        "\n"
        "#include \"tt_metal/host_api.hpp\"\n"
        "#include \"tt_metal/common/constants.hpp\"\n"
        "\n"
        "using namespace tt;\n"
        "using namespace tt::tt_metal;\n"
        "\n"
        "int main() {\n"
        "    Device *device = CreateDevice(0);\n"
        "    CommandQueue &cq = device->command_queue();\n"
        "    Program program = CreateProgram();\n"
        "\n"
        "    /* %dx%d Wormhole B0 compute grid */\n"
        "    CoreRange cores({0, 0}, {%d, %d});\n"
        "    constexpr uint32_t num_cores = %d;\n"
        "\n",
        TT_WH_CORES_X, TT_WH_CORES_Y,
        TT_WH_CORES_X - 1, TT_WH_CORES_Y - 1,
        TT_WH_NUM_CORES);

    host_write(fp,
        "    /* Circular buffers — adjust DataFormat and tile_size as needed.\n"
        "     * tile_size = 32*32*2 = 2048 for BF16, 32*32*4 = 4096 for FP32 */\n"
        "    constexpr uint32_t tile_size = 2048;\n"
        "\n"
        "    CircularBufferConfig cb_in_config = CircularBufferConfig(\n"
        "        tile_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})\n"
        "        .set_page_size(tt::CBIndex::c_0, tile_size);\n"
        "    CreateCircularBuffer(program, cores, cb_in_config);\n"
        "\n"
        "    CircularBufferConfig cb_out_config = CircularBufferConfig(\n"
        "        tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})\n"
        "        .set_page_size(tt::CBIndex::c_16, tile_size);\n"
        "    CreateCircularBuffer(program, cores, cb_out_config);\n"
        "\n");

    /* Emit kernel-specific sections */
    int guard = 8192;
    for (uint32_t fi = 0; fi < tt->num_mfuncs && guard > 0; fi++, guard--) {
        tt_mfunc_t *MF = &tt->mfuncs[fi];
        if (!MF->is_kernel) continue;

        const char *name = tt->bir->strings + MF->name;
        const char *pat = (unsigned)MF->coarsen_pattern < TT_PATTERN_COUNT
                          ? pattern_names[MF->coarsen_pattern] : "generic";

        host_write(fp,
            "    /* Kernel: %s (%s) */\n"
            "    /* TODO(tier4): add reader/writer kernels for data movement */\n"
            "\n"
            "    auto compute_%s = CreateKernel(\n"
            "        program, \"%s\", cores,\n"
            "        ComputeConfig{});\n"
            "\n",
            name, pat, name, compute_path);

        /* Runtime args with tiles_per_core formula */
        host_write(fp,
            "    /* Thread coarsening: total_elements / (num_cores * %d) = tiles_per_core.\n"
            "     * Each core runs a tile loop; each tile = %d SFPU lanes. */\n"
            "    uint32_t total_elements_%s = 1024 * 1024;  /* TODO: your data size */\n"
            "    uint32_t tiles_per_core_%s = (total_elements_%s + num_cores * %d - 1)\n"
            "                                / (num_cores * %d);\n"
            "\n",
            TT_SFPU_LANES, TT_SFPU_LANES,
            name, name, name, TT_SFPU_LANES, TT_SFPU_LANES);

        host_write(fp,
            "    for (uint32_t y = 0; y < %d; y++) {\n"
            "        for (uint32_t x = 0; x < %d; x++) {\n"
            "            SetRuntimeArgs(program, compute_%s,\n"
            "                           CoreCoord{x, y}, {tiles_per_core_%s});\n"
            "        }\n"
            "    }\n"
            "\n",
            TT_WH_CORES_Y, TT_WH_CORES_X, name, name);
    }

    host_write(fp,
        "    EnqueueProgram(cq, program, false);\n"
        "    Finish(cq);\n"
        "    CloseDevice(device);\n"
        "    return 0;\n"
        "}\n");

    fclose(fp);
    printf("wrote %s (host program skeleton)\n", host_path);
    return BC_OK;
}
