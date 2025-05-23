#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#define CHANNELS 3
namespace cg = cooperative_groups;

__device__ __constant__ float SH_C0 = 0.28209479177387814f;
__device__ __constant__ float SH_C1 = 0.4886025119029199f;
__device__ __constant__ float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
__device__ __constant__ float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
__device__ __constant__ float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

// This function is used in both host and device code
__host__ __device__ unsigned num_sh_bases(const unsigned degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

template <int tile_sz>
inline __device__ float warpSum(cg::thread_block_tile<tile_sz>& tile,float& val){
    val = cg::reduce(tile, val, cg::plus<float>());
    return val;
}

__device__ void sh_coeffs_to_color(
    const unsigned degree,
    const float3 &viewdir,
    const float *coeffs,
    float *colors
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    // expects CHANNELS * num_bases coefficients
    // supports up to num_bases = 25
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * CHANNELS + c] +
                              z * coeffs[2 * CHANNELS + c] -
                              x * coeffs[3 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * CHANNELS + c] +
             SH_C2[1] * yz * coeffs[5 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * coeffs[6 * CHANNELS + c] +
             SH_C2[3] * xz * coeffs[7 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * coeffs[8 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * coeffs[9 * CHANNELS + c] +
             SH_C3[1] * xy * z * coeffs[10 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * coeffs[11 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 coeffs[12 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * coeffs[13 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * coeffs[14 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * coeffs[15 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * coeffs[16 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * coeffs[17 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * coeffs[18 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * coeffs[19 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 coeffs[20 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * coeffs[21 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 coeffs[22 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * coeffs[23 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 coeffs[24 * CHANNELS + c]);
    }
}

__device__ void sh_coeffs_to_color_vjp(
    const unsigned degree,
    const float3 &viewdir,
    const float *v_colors,
    float *v_coeffs
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_coeffs[1 * CHANNELS + c] = v1 * v_colors[c];
        v_coeffs[2 * CHANNELS + c] = v2 * v_colors[c];
        v_coeffs[3 * CHANNELS + c] = v3 * v_colors[c];
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_coeffs[4 * CHANNELS + c] = v4 * v_colors[c];
        v_coeffs[5 * CHANNELS + c] = v5 * v_colors[c];
        v_coeffs[6 * CHANNELS + c] = v6 * v_colors[c];
        v_coeffs[7 * CHANNELS + c] = v7 * v_colors[c];
        v_coeffs[8 * CHANNELS + c] = v8 * v_colors[c];
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_coeffs[9 * CHANNELS + c] = v9 * v_colors[c];
        v_coeffs[10 * CHANNELS + c] = v10 * v_colors[c];
        v_coeffs[11 * CHANNELS + c] = v11 * v_colors[c];
        v_coeffs[12 * CHANNELS + c] = v12 * v_colors[c];
        v_coeffs[13 * CHANNELS + c] = v13 * v_colors[c];
        v_coeffs[14 * CHANNELS + c] = v14 * v_colors[c];
        v_coeffs[15 * CHANNELS + c] = v15 * v_colors[c];
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_coeffs[16 * CHANNELS + c] = v16 * v_colors[c];
        v_coeffs[17 * CHANNELS + c] = v17 * v_colors[c];
        v_coeffs[18 * CHANNELS + c] = v18 * v_colors[c];
        v_coeffs[19 * CHANNELS + c] = v19 * v_colors[c];
        v_coeffs[20 * CHANNELS + c] = v20 * v_colors[c];
        v_coeffs[21 * CHANNELS + c] = v21 * v_colors[c];
        v_coeffs[22 * CHANNELS + c] = v22 * v_colors[c];
        v_coeffs[23 * CHANNELS + c] = v23 * v_colors[c];
        v_coeffs[24 * CHANNELS + c] = v24 * v_colors[c];
    }
}


__device__ void sh_coeffs_to_color_vjp2(
    const unsigned num_bases,
    const float3 &viewdir,
    const float *v_colors,
    float *v_coeffs
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    float coeffs[25][CHANNELS];
    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        coeffs[0][c] = SH_C0 * v_colors[c];        
        if (num_bases > 1) {
            coeffs[1][c] = -SH_C1 * y * v_colors[c];
            coeffs[2][c] = SH_C1 * z * v_colors[c];
            coeffs[3][c] = -SH_C1 * x * v_colors[c];

            if (num_bases > 4) {
                coeffs[4][c] = SH_C2[0] * (x * y) * v_colors[c];
                coeffs[5][c] = SH_C2[1] * (y * z) * v_colors[c];
                coeffs[6][c] = SH_C2[2] * (2 * zz - xx - yy) * v_colors[c];
                coeffs[7][c] = SH_C2[3] * (x * z) * v_colors[c];
                coeffs[8][c] = SH_C2[4] * (xx - yy) * v_colors[c];

                if (num_bases > 9) {
                    // 计算度数为 3 的项
                    coeffs[9][c] = SH_C3[0] * y * (3.f * xx - yy) * v_colors[c];
                    coeffs[10][c] = SH_C3[1] * (x * y) * z * v_colors[c];
                    coeffs[11][c] = SH_C3[2] * y * (4.f * zz - xx - yy) * v_colors[c];
                    coeffs[12][c] = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) * v_colors[c];
                    coeffs[13][c] = SH_C3[4] * x * (4.f * zz - xx - yy) * v_colors[c];
                    coeffs[14][c] = SH_C3[5] * z * (xx - yy) * v_colors[c];
                    coeffs[15][c] = SH_C3[6] * x * (xx - 3.f * yy) * v_colors[c];

                    if (num_bases > 16) {
                        // 计算度数为 4 的项
                        coeffs[16][c] = SH_C4[0] * (x * y) * (xx - yy) * v_colors[c];
                        coeffs[17][c] = SH_C4[1] * (y * z) * (3.f * xx - yy) * v_colors[c];
                        coeffs[18][c] = SH_C4[2] * (x * y) * (7.f * zz - 1.f) * v_colors[c];
                        coeffs[19][c] = SH_C4[3] * (y * z) * (7.f * zz - 3.f) * v_colors[c];
                        coeffs[20][c] = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) * v_colors[c];
                        coeffs[21][c] = SH_C4[5] * (x * z) * (7.f * zz - 3.f) * v_colors[c];
                        coeffs[22][c] = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) * v_colors[c];
                        coeffs[23][c] = SH_C4[7] * (x * z) * (xx - 3.f * yy) * v_colors[c];
                        coeffs[24][c] = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) * v_colors[c];
                    }
                }
            }
        }
    }
    // auto g = cg::this_thread_block();
    // g.sync();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(cg::this_thread_block());
    
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        #pragma unroll
        for (int i = 0; i < num_bases; ++i) {
            // extern __shared__ float sdata[];
            // float block_sum = reduce_sum(g,sdata,coeffs[i][c]);
            // float block_sum = reduce_sum<32>(g, coeffs[i][c]);
            float block_sum = warpSum<32>(g, coeffs[i][c]);
            // printf("%d:%d,%d,%f,%f\n",g.thread_rank(),i,c,coeffs[i][c],block_sum);
            if (g.thread_rank() == 0) {
                atomicAdd(&v_coeffs[i * CHANNELS + c], block_sum);
            }
        }
    }
}

__device__ void sh_coeffs_to_color_fast(const unsigned degree,
                                        const float3& viewdir,
                                        const float* coeffs,
                                        float* colors) {
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = 0.2820947917738781f * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = rsqrtf(viewdir.x * viewdir.x + viewdir.y * viewdir.y +
                        viewdir.z * viewdir.z);
    float x = viewdir.x * norm;
    float y = viewdir.y * norm;
    float z = viewdir.z * norm;

#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float fTmp0A = 0.48860251190292f;
        colors[c] += fTmp0A * (-y * coeffs[1 * CHANNELS + c] +
                               z * coeffs[2 * CHANNELS + c] -
                               x * coeffs[3 * CHANNELS + c]);
    }
    if (degree < 2) {
        return;
    }
    float z2 = z * z;

    float fTmp0B = -1.092548430592079f * z;
    float fTmp1A = 0.5462742152960395f;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = fTmp1A * fC1;
    float pSH4 = fTmp1A * fS1;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] +=
            pSH4 * coeffs[4 * CHANNELS + c] + pSH5 * coeffs[5 * CHANNELS + c] +
            pSH6 * coeffs[6 * CHANNELS + c] + pSH7 * coeffs[7 * CHANNELS + c] +
            pSH8 * coeffs[8 * CHANNELS + c];
    }
    if (degree < 3) {
        return;
    }

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fTmp2A = -0.5900435899266435f;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = fTmp2A * fC2;
    float pSH9 = fTmp2A * fS2;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += pSH9 * coeffs[9 * CHANNELS + c] +
                     pSH10 * coeffs[10 * CHANNELS + c] +
                     pSH11 * coeffs[11 * CHANNELS + c] +
                     pSH12 * coeffs[12 * CHANNELS + c] +
                     pSH13 * coeffs[13 * CHANNELS + c] +
                     pSH14 * coeffs[14 * CHANNELS + c] +
                     pSH15 * coeffs[15 * CHANNELS + c];
    }
    if (degree < 4) {
        return;
    }

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fTmp3A = 0.6258357354491763f;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = fTmp3A * fC3;
    float pSH16 = fTmp3A * fS3;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += pSH16 * coeffs[16 * CHANNELS + c] +
                     pSH17 * coeffs[17 * CHANNELS + c] +
                     pSH18 * coeffs[18 * CHANNELS + c] +
                     pSH19 * coeffs[19 * CHANNELS + c] +
                     pSH20 * coeffs[20 * CHANNELS + c] +
                     pSH21 * coeffs[21 * CHANNELS + c] +
                     pSH22 * coeffs[22 * CHANNELS + c] +
                     pSH23 * coeffs[23 * CHANNELS + c] +
                     pSH24 * coeffs[24 * CHANNELS + c];
    }
}

__device__ void sh_coeffs_to_color_fast_vjp(const unsigned degree,
                                            const float3& viewdir,
                                            const float* v_colors,
                                            float* v_coeffs) {
// Expects v_colors to be len CHANNELS
// and v_coeffs to be num_bases * CHANNELS
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[c] = 0.2820947917738781f * v_colors[c];
    }
    if (degree < 1) {
        return;
    }
    float norm = rsqrtf(viewdir.x * viewdir.x + viewdir.y * viewdir.y +
                        viewdir.z * viewdir.z);
    float x = viewdir.x * norm;
    float y = viewdir.y * norm;
    float z = viewdir.z * norm;

    float fTmp0A = 0.48860251190292f;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[1 * CHANNELS + c] = -fTmp0A * y * v_colors[c];
        v_coeffs[2 * CHANNELS + c] = fTmp0A * z * v_colors[c];
        v_coeffs[3 * CHANNELS + c] = -fTmp0A * x * v_colors[c];
    }
    if (degree < 2) {
        return;
    }

    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fTmp1A = 0.5462742152960395f;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = fTmp1A * fC1;
    float pSH4 = fTmp1A * fS1;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[4 * CHANNELS + c] = pSH4 * v_colors[c];
        v_coeffs[5 * CHANNELS + c] = pSH5 * v_colors[c];
        v_coeffs[6 * CHANNELS + c] = pSH6 * v_colors[c];
        v_coeffs[7 * CHANNELS + c] = pSH7 * v_colors[c];
        v_coeffs[8 * CHANNELS + c] = pSH8 * v_colors[c];
    }
    if (degree < 3) {
        return;
    }

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fTmp2A = -0.5900435899266435f;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = fTmp2A * fC2;
    float pSH9 = fTmp2A * fS2;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[9 * CHANNELS + c] = pSH9 * v_colors[c];
        v_coeffs[10 * CHANNELS + c] = pSH10 * v_colors[c];
        v_coeffs[11 * CHANNELS + c] = pSH11 * v_colors[c];
        v_coeffs[12 * CHANNELS + c] = pSH12 * v_colors[c];
        v_coeffs[13 * CHANNELS + c] = pSH13 * v_colors[c];
        v_coeffs[14 * CHANNELS + c] = pSH14 * v_colors[c];
        v_coeffs[15 * CHANNELS + c] = pSH15 * v_colors[c];
    }
    if (degree < 4) {
        return;
    }

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fTmp3A = 0.6258357354491763f;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = fTmp3A * fC3;
    float pSH16 = fTmp3A * fS3;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[16 * CHANNELS + c] = pSH16 * v_colors[c];
        v_coeffs[17 * CHANNELS + c] = pSH17 * v_colors[c];
        v_coeffs[18 * CHANNELS + c] = pSH18 * v_colors[c];
        v_coeffs[19 * CHANNELS + c] = pSH19 * v_colors[c];
        v_coeffs[20 * CHANNELS + c] = pSH20 * v_colors[c];
        v_coeffs[21 * CHANNELS + c] = pSH21 * v_colors[c];
        v_coeffs[22 * CHANNELS + c] = pSH22 * v_colors[c];
        v_coeffs[23 * CHANNELS + c] = pSH23 * v_colors[c];
        v_coeffs[24 * CHANNELS + c] = pSH24 * v_colors[c];
    }
}

__device__ void sh_coeffs_to_color_fast_vjp2(const unsigned num_bases,
                                             const float3& viewdir,
                                             const float* v_colors,
                                             float* v_coeffs) {
    float coeffs[25 * CHANNELS];
    float norm = rsqrtf(viewdir.x * viewdir.x + viewdir.y * viewdir.y +
                        viewdir.z * viewdir.z);
    float x = viewdir.x * norm;
    float y = viewdir.y * norm;
    float z = viewdir.z * norm;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        coeffs[c] = 0.2820947917738781f * v_colors[c];
        if (num_bases > 1) {
            float fTmp0A = 0.48860251190292f;
#pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                coeffs[1 * CHANNELS + c] = -fTmp0A * y * v_colors[c];
                coeffs[2 * CHANNELS + c] = fTmp0A * z * v_colors[c];
                coeffs[3 * CHANNELS + c] = -fTmp0A * x * v_colors[c];
                if (num_bases > 4) {
                    float z2 = z * z;
                    float fTmp0B = -1.092548430592079f * z;
                    float fTmp1A = 0.5462742152960395f;
                    float fC1 = x * x - y * y;
                    float fS1 = 2.f * x * y;
                    float pSH6 =
                        (0.9461746957575601f * z2 - 0.3153915652525201f);
                    float pSH7 = fTmp0B * x;
                    float pSH5 = fTmp0B * y;
                    float pSH8 = fTmp1A * fC1;
                    float pSH4 = fTmp1A * fS1;
#pragma unroll
                    for (int c = 0; c < CHANNELS; ++c) {
                        coeffs[4 * CHANNELS + c] = pSH4 * v_colors[c];
                        coeffs[5 * CHANNELS + c] = pSH5 * v_colors[c];
                        coeffs[6 * CHANNELS + c] = pSH6 * v_colors[c];
                        coeffs[7 * CHANNELS + c] = pSH7 * v_colors[c];
                        coeffs[8 * CHANNELS + c] = pSH8 * v_colors[c];
                    }
                    if (num_bases > 9) {
                        float fTmp0C =
                            -2.285228997322329f * z2 + 0.4570457994644658f;
                        float fTmp1B = 1.445305721320277f * z;
                        float fTmp2A = -0.5900435899266435f;
                        float fC2 = x * fC1 - y * fS1;
                        float fS2 = x * fS1 + y * fC1;
                        float pSH12 =
                            z * (1.865881662950577f * z2 - 1.119528997770346f);
                        float pSH13 = fTmp0C * x;
                        float pSH11 = fTmp0C * y;
                        float pSH14 = fTmp1B * fC1;
                        float pSH10 = fTmp1B * fS1;
                        float pSH15 = fTmp2A * fC2;
                        float pSH9 = fTmp2A * fS2;
#pragma unroll
                        for (int c = 0; c < CHANNELS; ++c) {
                            coeffs[9 * CHANNELS + c] = pSH9 * v_colors[c];
                            coeffs[10 * CHANNELS + c] = pSH10 * v_colors[c];
                            coeffs[11 * CHANNELS + c] = pSH11 * v_colors[c];
                            coeffs[12 * CHANNELS + c] = pSH12 * v_colors[c];
                            coeffs[13 * CHANNELS + c] = pSH13 * v_colors[c];
                            coeffs[14 * CHANNELS + c] = pSH14 * v_colors[c];
                            coeffs[15 * CHANNELS + c] = pSH15 * v_colors[c];
                        }
                        if (num_bases > 16) {
                            float fTmp0D = z * (-4.683325804901025f * z2 +
                                                2.007139630671868f);
                            float fTmp1C =
                                3.31161143515146f * z2 - 0.47308734787878f;
                            float fTmp2B = -1.770130769779931f * z;
                            float fTmp3A = 0.6258357354491763f;
                            float fC3 = x * fC2 - y * fS2;
                            float fS3 = x * fS2 + y * fC2;
                            float pSH20 = (1.984313483298443f * z * pSH12 +
                                           -1.006230589874905f * pSH6);
                            float pSH21 = fTmp0D * x;
                            float pSH19 = fTmp0D * y;
                            float pSH22 = fTmp1C * fC1;
                            float pSH18 = fTmp1C * fS1;
                            float pSH23 = fTmp2B * fC2;
                            float pSH17 = fTmp2B * fS2;
                            float pSH24 = fTmp3A * fC3;
                            float pSH16 = fTmp3A * fS3;
#pragma unroll
                            for (int c = 0; c < CHANNELS; ++c) {
                                coeffs[16 * CHANNELS + c] = pSH16 * v_colors[c];
                                coeffs[17 * CHANNELS + c] = pSH17 * v_colors[c];
                                coeffs[18 * CHANNELS + c] = pSH18 * v_colors[c];
                                coeffs[19 * CHANNELS + c] = pSH19 * v_colors[c];
                                coeffs[20 * CHANNELS + c] = pSH20 * v_colors[c];
                                coeffs[21 * CHANNELS + c] = pSH21 * v_colors[c];
                                coeffs[22 * CHANNELS + c] = pSH22 * v_colors[c];
                                coeffs[23 * CHANNELS + c] = pSH23 * v_colors[c];
                                coeffs[24 * CHANNELS + c] = pSH24 * v_colors[c];
                            }
                        }
                    }
                }
            }
        }
    }
    cg::thread_block_tile<32> g =
        cg::tiled_partition<32>(cg::this_thread_block());

#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
#pragma unroll
        for (int i = 0; i < num_bases; ++i) {
            // extern __shared__ float sdata[];
            // float block_sum = reduce_sum(g,sdata,coeffs[i][c]);
            // float block_sum = reduce_sum<32>(g, coeffs[i][c]);
            float block_sum = warpSum<32>(g, coeffs[i * CHANNELS + c]);
            // printf("%d:%d,%d,%f,%f\n",g.thread_rank(),i,c,coeffs[i][c],block_sum);
            if (g.thread_rank() == 0) {
                atomicAdd(&v_coeffs[i * CHANNELS + c], block_sum);
            }
        }
    }
}

__global__ void compute_sh_forward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ coeffs,
    float* __restrict__ colors
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color_fast(
        degrees_to_use, viewdirs[idx], &(coeffs[idx_sh]), &(colors[idx_col])
    );
}

__global__ void compute_sh_backward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ v_colors,
    float* __restrict__ v_coeffs
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color_fast_vjp(
        degrees_to_use, viewdirs[idx], &(v_colors[idx_col]), &(v_coeffs[idx_sh])
    );
}



__global__ void sparse_compute_sh_forward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ coeffs,
    float* __restrict__ colors
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    float3 viewdir = viewdirs[idx];
    #pragma unroll
    for(int group = 0; group < 3; ++group){
        sh_coeffs_to_color_fast(
            degrees_to_use, viewdir, &(coeffs[num_bases * num_channels * group]), &(colors[idx_col + num_channels * num_points * group])
        );
    }
}

__global__ void sparse_compute_sh_backward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ v_colors,
    float* __restrict__ v_coeffs
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned num_bases_used = num_sh_bases(degrees_to_use);

    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;
    float3 viewdir = viewdirs[idx];
    #pragma unroll
    for(int group = 0 ; group < 3; ++group ){  
        sh_coeffs_to_color_fast_vjp2(
            num_bases_used, viewdir, &(v_colors[idx_col + num_points * num_channels * group]), &(v_coeffs[num_bases * num_channels * group])
        );
    }
}

