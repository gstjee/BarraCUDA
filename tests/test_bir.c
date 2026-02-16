#include "../src/bir.h"
#include <stdio.h>
#include <assert.h>

/* Struct size verification — if these fail, the cache-line math is wrong */
typedef char check_inst_size [sizeof(bir_inst_t)   == 32 ? 1 : -1];
typedef char check_type_size [sizeof(bir_type_t)   == 16 ? 1 : -1];
typedef char check_const_size[sizeof(bir_const_t)  == 16 ? 1 : -1];
typedef char check_func_size [sizeof(bir_func_t)   == 32 ? 1 : -1];
typedef char check_global_size[sizeof(bir_global_t) == 16 ? 1 : -1];
typedef char check_block_size[sizeof(bir_block_t)  == 12 ? 1 : -1];

static bir_module_t module;

static void test_type_interning(void)
{
    bir_module_init(&module);

    uint32_t v1 = bir_type_void(&module);
    uint32_t v2 = bir_type_void(&module);
    assert(v1 == v2 && "void should be interned");

    uint32_t i32a = bir_type_int(&module, 32);
    uint32_t i32b = bir_type_int(&module, 32);
    assert(i32a == i32b && "i32 should be interned");

    uint32_t i64 = bir_type_int(&module, 64);
    assert(i64 != i32a && "i64 != i32");

    uint32_t f32a = bir_type_float(&module, 32);
    uint32_t f32b = bir_type_float(&module, 32);
    assert(f32a == f32b && "f32 should be interned");

    uint32_t f64 = bir_type_float(&module, 64);
    assert(f64 != f32a && "f64 != f32");

    uint32_t pg = bir_type_ptr(&module, f32a, BIR_AS_GLOBAL);
    uint32_t pg2 = bir_type_ptr(&module, f32a, BIR_AS_GLOBAL);
    assert(pg == pg2 && "ptr<global, f32> should be interned");

    uint32_t ps = bir_type_ptr(&module, f32a, BIR_AS_SHARED);
    assert(ps != pg && "different addrspace = different type");

    uint32_t arr = bir_type_array(&module, f32a, 256);
    uint32_t arr2 = bir_type_array(&module, f32a, 256);
    assert(arr == arr2 && "[256 x f32] should be interned");

    uint32_t arr3 = bir_type_array(&module, f32a, 512);
    assert(arr3 != arr && "different count = different type");

    uint32_t vec4 = bir_type_vector(&module, f32a, 4);
    uint32_t vec4b = bir_type_vector(&module, f32a, 4);
    assert(vec4 == vec4b && "<4 x f32> should be interned");

    /* Struct: {i32, f32} */
    uint32_t fields1[] = { i32a, f32a };
    uint32_t s1 = bir_type_struct(&module, fields1, 2);
    uint32_t s2 = bir_type_struct(&module, fields1, 2);
    assert(s1 == s2 && "identical struct should be interned");

    uint32_t fields2[] = { f32a, i32a };
    uint32_t s3 = bir_type_struct(&module, fields2, 2);
    assert(s3 != s1 && "different field order = different struct");

    /* Function type: (i32, ptr<global, f32>) -> void */
    uint32_t params[] = { i32a, pg };
    uint32_t fn1 = bir_type_func(&module, v1, params, 2);
    uint32_t fn2 = bir_type_func(&module, v1, params, 2);
    assert(fn1 == fn2 && "identical func type should be interned");

    printf("  types: %u interned, %u type_fields used\n",
           module.num_types, module.num_type_fields);
    printf("  type_interning: PASS\n");
}

static void test_constants(void)
{
    bir_module_init(&module);

    uint32_t i32 = bir_type_int(&module, 32);
    uint32_t f64 = bir_type_float(&module, 64);

    uint32_t c0a = bir_const_int(&module, i32, 0);
    uint32_t c0b = bir_const_int(&module, i32, 0);
    assert(c0a == c0b && "int constant 0 should be deduped");

    uint32_t c42 = bir_const_int(&module, i32, 42);
    assert(c42 != c0a && "different value = different constant");

    uint32_t cf = bir_const_float(&module, f64, 3.14);
    uint32_t cf2 = bir_const_float(&module, f64, 3.14);
    assert(cf == cf2 && "float constant should be deduped");

    uint32_t ptr_t = bir_type_ptr(&module, i32, BIR_AS_GLOBAL);
    uint32_t cn = bir_const_null(&module, ptr_t);
    uint32_t cn2 = bir_const_null(&module, ptr_t);
    assert(cn == cn2 && "null constant should be deduped");

    /* Value reference encoding */
    uint32_t ref = BIR_MAKE_CONST(c42);
    assert(BIR_VAL_IS_CONST(ref) && "should be const ref");
    assert(BIR_VAL_INDEX(ref) == c42 && "index should match");

    uint32_t val_ref = BIR_MAKE_VAL(100);
    assert(!BIR_VAL_IS_CONST(val_ref) && "should not be const ref");
    assert(BIR_VAL_INDEX(val_ref) == 100 && "index should match");

    printf("  constants: %u interned\n", module.num_consts);
    printf("  constants: PASS\n");
}

static void test_strings(void)
{
    bir_module_init(&module);

    uint32_t s1 = bir_add_string(&module, "entry", 5);
    assert(s1 == 0 && "first string at offset 0");
    assert(strcmp(&module.strings[s1], "entry") == 0);

    uint32_t s2 = bir_add_string(&module, "loop", 4);
    assert(s2 == 6 && "second string after null terminator");
    assert(strcmp(&module.strings[s2], "loop") == 0);

    printf("  strings: %u bytes used\n", module.string_len);
    printf("  strings: PASS\n");
}

static void test_name_tables(void)
{
    assert(strcmp(bir_op_name(BIR_ADD), "add") == 0);
    assert(strcmp(bir_op_name(BIR_BARRIER), "barrier") == 0);
    assert(strcmp(bir_op_name(BIR_THREAD_ID), "thread_id") == 0);
    assert(strcmp(bir_op_name(BIR_INLINE_ASM), "inline_asm") == 0);
    assert(strcmp(bir_op_name(9999), "???") == 0);

    assert(strcmp(bir_cmp_name(BIR_ICMP_EQ), "eq") == 0);
    assert(strcmp(bir_cmp_name(BIR_FCMP_OLT), "olt") == 0);

    assert(strcmp(bir_addrspace_name(BIR_AS_GLOBAL), "global") == 0);
    assert(strcmp(bir_addrspace_name(BIR_AS_SHARED), "shared") == 0);

    assert(strcmp(bir_type_kind_name(BIR_TYPE_PTR), "ptr") == 0);

    assert(strcmp(bir_order_name(BIR_ORDER_SEQ_CST), "seq_cst") == 0);

    printf("  name_tables: PASS\n");
}

/* Helper: emit one instruction into the module, returns its global index */
static uint32_t emit(bir_inst_t **slot, uint16_t op, uint32_t type,
                     uint8_t nops, uint8_t subop)
{
    uint32_t idx = module.num_insts++;
    bir_inst_t *I = &module.insts[idx];
    memset(I, 0, sizeof(*I));
    I->op = op;
    I->type = type;
    I->num_operands = nops;
    I->subop = subop;
    if (slot) *slot = I;
    return idx;
}

static void test_printer(void)
{
    bir_module_init(&module);
    bir_inst_t *I;

    /* Types */
    uint32_t t_void = bir_type_void(&module);
    uint32_t t_i1   = bir_type_int(&module, 1);
    uint32_t t_i32  = bir_type_int(&module, 32);
    uint32_t t_f32  = bir_type_float(&module, 32);
    uint32_t t_pg   = bir_type_ptr(&module, t_f32, BIR_AS_GLOBAL);

    /* Function type: (ptr<global>, ptr<global>, ptr<global>, i32) -> void */
    uint32_t fparams[] = { t_pg, t_pg, t_pg, t_i32 };
    uint32_t t_fn = bir_type_func(&module, t_void, fparams, 4);

    /* Strings */
    uint32_t s_name  = bir_add_string(&module, "vectorAdd", 9);
    uint32_t s_entry = bir_add_string(&module, "entry", 5);
    uint32_t s_body  = bir_add_string(&module, "body", 4);
    uint32_t s_exit  = bir_add_string(&module, "exit", 4);

    /* Function */
    uint32_t fi = module.num_funcs++;
    bir_func_t *F = &module.funcs[fi];
    memset(F, 0, sizeof(*F));
    F->name = s_name;
    F->type = t_fn;
    F->cuda_flags = CUDA_GLOBAL;
    F->num_params = 4;
    F->first_block = module.num_blocks;
    F->num_blocks = 3;

    /* === Entry block === */
    uint32_t bi_entry = module.num_blocks++;
    module.blocks[bi_entry].name = s_entry;
    module.blocks[bi_entry].first_inst = module.num_insts;
    uint32_t base = module.num_insts;

    /* %0-%3 = params */
    emit(NULL, BIR_PARAM, t_pg,  0, 0);
    emit(NULL, BIR_PARAM, t_pg,  0, 1);
    emit(NULL, BIR_PARAM, t_pg,  0, 2);
    emit(NULL, BIR_PARAM, t_i32, 0, 3);

    /* %4 = block_id.x */
    emit(NULL, BIR_BLOCK_ID,  t_i32, 0, 0);
    /* %5 = block_dim.x */
    emit(NULL, BIR_BLOCK_DIM, t_i32, 0, 0);

    /* %6 = mul i32 %4, %5 */
    emit(&I, BIR_MUL, t_i32, 2, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 4);
    I->operands[1] = BIR_MAKE_VAL(base + 5);

    /* %7 = thread_id.x */
    emit(NULL, BIR_THREAD_ID, t_i32, 0, 0);

    /* %8 = add i32 %6, %7 */
    emit(&I, BIR_ADD, t_i32, 2, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 6);
    I->operands[1] = BIR_MAKE_VAL(base + 7);

    /* %9 = icmp slt i32 %8, %3 */
    emit(&I, BIR_ICMP, t_i1, 2, BIR_ICMP_SLT);
    I->operands[0] = BIR_MAKE_VAL(base + 8);
    I->operands[1] = BIR_MAKE_VAL(base + 3);

    /* br_cond %9, body, exit */
    emit(&I, BIR_BR_COND, t_void, 3, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 9);
    I->operands[1] = bi_entry + 1;
    I->operands[2] = bi_entry + 2;

    module.blocks[bi_entry].num_insts =
        module.num_insts - module.blocks[bi_entry].first_inst;

    /* === Body block === */
    uint32_t bi_body = module.num_blocks++;
    module.blocks[bi_body].name = s_body;
    module.blocks[bi_body].first_inst = module.num_insts;

    /* %11 = gep ptr<global,f32>, %0, %8 */
    emit(&I, BIR_GEP, t_pg, 2, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 0);
    I->operands[1] = BIR_MAKE_VAL(base + 8);

    /* %12 = load f32, %11 */
    emit(&I, BIR_LOAD, t_f32, 1, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 11);

    /* %13 = gep ptr<global,f32>, %1, %8 */
    emit(&I, BIR_GEP, t_pg, 2, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 1);
    I->operands[1] = BIR_MAKE_VAL(base + 8);

    /* %14 = load f32, %13 */
    emit(&I, BIR_LOAD, t_f32, 1, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 13);

    /* %15 = fadd f32 %12, %14 */
    emit(&I, BIR_FADD, t_f32, 2, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 12);
    I->operands[1] = BIR_MAKE_VAL(base + 14);

    /* %16 = gep ptr<global,f32>, %2, %8 */
    emit(&I, BIR_GEP, t_pg, 2, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 2);
    I->operands[1] = BIR_MAKE_VAL(base + 8);

    /* store f32 %15, %16 */
    emit(&I, BIR_STORE, t_void, 2, 0);
    I->operands[0] = BIR_MAKE_VAL(base + 15);
    I->operands[1] = BIR_MAKE_VAL(base + 16);

    /* br exit */
    emit(&I, BIR_BR, t_void, 1, 0);
    I->operands[0] = bi_entry + 2;

    module.blocks[bi_body].num_insts =
        module.num_insts - module.blocks[bi_body].first_inst;

    /* === Exit block === */
    uint32_t bi_exit = module.num_blocks++;
    module.blocks[bi_exit].name = s_exit;
    module.blocks[bi_exit].first_inst = module.num_insts;

    /* ret void */
    emit(NULL, BIR_RET, t_void, 0, 0);

    module.blocks[bi_exit].num_insts = 1;
    F->total_insts = module.num_insts - base;

    /* Suppress unused variable warnings */
    (void)t_fn; (void)s_entry; (void)s_body; (void)s_exit;

    /* Print the module */
    printf("\n--- BIR output ---\n");
    bir_print_module(&module, stdout);
    printf("--- end ---\n");
    printf("  %u instructions, %u blocks, %u functions\n",
           module.num_insts, module.num_blocks, module.num_funcs);
    printf("  printer: PASS\n");
}

int main(void)
{
    printf("BIR tests:\n");
    printf("  sizeof(bir_inst_t)   = %u\n", (unsigned)sizeof(bir_inst_t));
    printf("  sizeof(bir_type_t)   = %u\n", (unsigned)sizeof(bir_type_t));
    printf("  sizeof(bir_const_t)  = %u\n", (unsigned)sizeof(bir_const_t));
    printf("  sizeof(bir_func_t)   = %u\n", (unsigned)sizeof(bir_func_t));
    printf("  sizeof(bir_global_t) = %u\n", (unsigned)sizeof(bir_global_t));
    printf("  sizeof(bir_block_t)  = %u\n", (unsigned)sizeof(bir_block_t));
    printf("  sizeof(bir_module_t) = %u MB\n",
           (unsigned)(sizeof(bir_module_t) / (1024 * 1024)));
    printf("\n");

    test_type_interning();
    test_constants();
    test_strings();
    test_name_tables();
    test_printer();

    printf("\nAll BIR tests passed.\n");
    return 0;
}
