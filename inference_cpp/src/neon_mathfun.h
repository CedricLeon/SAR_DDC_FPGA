#pragma once
// neon_mathfun.h — vectorised logf/expf for ARM NEON (float32x4_t), used by the --neon path of
// normalize/denorm (patch_transforms.hpp). Classic Cephes single-precision polynomials (the
// Julien Pommier neon_mathfun implementation, zlib licence); ~1 ULP-class accuracy, far below the
// INT8 quantisation step the DPU applies to the normalize output.
//
// Guarded by __ARM_NEON: on a non-NEON host these are absent and callers fall back to scalar.

#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>

namespace ddc {

// e^x for four lanes. x is clamped to [-88.37, 88.37] (the float32 exp range).
inline float32x4_t exp_ps(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));

    // fx = floor(x * LOG2EF + 0.5)
    float32x4_t fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f));
    float32x4_t ftmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));  // trunc towards zero
    uint32x4_t mask = vandq_u32(vcgtq_f32(ftmp, fx), vreinterpretq_u32_f32(one));
    fx = vsubq_f32(ftmp, vreinterpretq_f32_u32(mask));

    // x -= fx * C1 + fx * C2   (Cephes two-part ln2)
    x = vsubq_f32(x, vmulq_f32(fx, vdupq_n_f32(0.693359375f)));
    x = vsubq_f32(x, vmulq_f32(fx, vdupq_n_f32(-2.12194440e-4f)));
    float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
    y = vmlaq_f32(vdupq_n_f32(1.3981999507E-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(8.3334519073E-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(4.1665795894E-2f), y, x);
    y = vmlaq_f32(vdupq_n_f32(1.6666665459E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(5.0000001201E-1f), y, x);
    y = vmlaq_f32(x, y, z);
    y = vaddq_f32(y, one);

    // build 2^fx by assembling the exponent field
    int32x4_t pow2 = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(0x7f));
    pow2 = vshlq_n_s32(pow2, 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(pow2));
}

// ln(x) for four lanes. Lanes with x <= 0 return NaN (via the invalid mask), matching libm.
inline float32x4_t log_ps(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0.0f));
    x = vmaxq_f32(x, vreinterpretq_f32_u32(vdupq_n_u32(0x00800000)));  // smallest normal

    int32x4_t ix = vreinterpretq_s32_f32(x);
    float32x4_t e = vcvtq_f32_s32(vsubq_s32(vshrq_n_s32(ix, 23), vdupq_n_s32(0x7f)));
    e = vaddq_f32(e, one);
    // keep mantissa in [0.5, 1): clear exponent bits, set exponent to 0.5
    ix = vandq_s32(ix, vdupq_n_s32(~0x7f800000));
    ix = vorrq_s32(ix, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ix);

    // if x < SQRTHF: e -= 1; x += x; then x -= 1
    uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(0.707106781186547524f));
    float32x4_t xtmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, xtmp);
    float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(7.0376836292E-2f);
    y = vmlaq_f32(vdupq_n_f32(-1.1514610310E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(1.1676998740E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(-1.2420140846E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(1.4249322787E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(-1.6668057665E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(2.0000714765E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(-2.4999993993E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(3.3333331174E-1f), y, x);
    y = vmulq_f32(vmulq_f32(y, x), z);

    y = vmlaq_f32(y, e, vdupq_n_f32(-2.12194440e-4f));
    y = vsubq_f32(y, vmulq_f32(z, vdupq_n_f32(0.5f)));
    x = vaddq_f32(x, y);
    x = vmlaq_f32(x, e, vdupq_n_f32(0.693359375f));
    // x <= 0 lanes -> NaN
    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask));
}

}  // namespace ddc

#endif  // __ARM_NEON
