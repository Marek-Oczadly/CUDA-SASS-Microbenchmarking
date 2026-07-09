// Probe for TEST_FAIL_ANALYSIS.md Root Cause F: the DefaultInsAsmRepos.sm_70/75/80/86.txt
// "MUFU_R_FI" bucket was only ever trained on MUFU.RCP64H samples, so it doesn't know the
// "0_RSQ" modifier and rejects "MUFU.RSQ Rd, <float-immediate>" with "Unknown modifiers".
//
// That exact instruction shape (MUFU.RSQ Rd, 0fffc00000, i.e. rsqrt of a hardcoded quiet-NaN
// bit pattern) turns out to be boilerplate inside nvcc's precise/IEEE float-division slowpath
// ($__cuda_sm3x_div_rn_noftz_f32_slowpath), used to manufacture a NaN result cheaply via the
// MUFU unit rather than a dedicated "load NaN constant" instruction. It has nothing to do with
// rsqrtf() being called in source -- any kernel with a plain, non-fast-math float division on
// sm_70+ pulls in this slowpath function, so a bare division is enough to reproduce it.
extern "C" __global__ void mufu_rsq_fi_probe(const float* a, const float* b, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}
