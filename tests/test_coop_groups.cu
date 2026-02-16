/* test_coop_groups.cu — Verify cooperative_groups lowering. */

namespace cooperative_groups {
    struct thread_block {};
    thread_block this_thread_block();
}

__global__ void coop_kernel(int *out, int n)
{
    cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
    tb.sync();
    int rank = tb.thread_rank();
    int sz = tb.size();
    if (rank < n)
        out[rank] = rank + sz;
}
