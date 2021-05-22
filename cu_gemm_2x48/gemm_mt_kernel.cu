#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <vector>
#include <assert.h>
#include "slices_6bits_macros.h"
#include "slices_4bits_macros.h"
#include "slices_3bits_macros.h"
#include "slices_2bits_macros.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define DIVCEIL(a,b) (((a)%(b)!=0)?(a/b+1):(a/b))
#define IS_ROUND_UP(a, is_rnd, rnd_bit) (((a & is_rnd) == is_rnd) || ((a & ((a & rnd_bit) - ((a & rnd_bit) != 0))) != 0))

inline unsigned int intDivCeil(const unsigned int &a, const unsigned int &b) { return ( a%b != 0 ) ? (a/b+1) : (a/b); }


__device__ int find_4bits_2op(const int &a, const bool &is_round) {
    bool set1 = a & S4_OP2_SET_1;
    
    int h_1 = a & S4_MASK1;
    int h_2 = a & S4_MASK5;

    if (is_round) {
        bool h1_round_up = IS_ROUND_UP(a, S4_OP2_IS_RND_1, S4_OP2_RND_BIT_1);  
        h_1 += (h1_round_up) ? ((a & S4_OP2_RND_BIT_1) << 1) : 0;
        h_1 = (h_1 == S4_OP2_OVRFLW_1) ? S4_MASK1 : h_1;
    }

    return (set1) ? h_1 : h_2;
}


__device__ int find_4bits_3op(const int &a, const bool &is_round) {
    bool set1 = a & S4_OP3_SET_1;
    bool set2 = a & S4_OP3_SET_2;

    int h_1 = a & S4_MASK1;
    int h_2 = a & S4_MASK3;
    int h_3 = a & S4_MASK5;

    if (is_round) {
        bool h1_round_up = IS_ROUND_UP(a, S4_OP3_IS_RND_1, S4_OP3_RND_BIT_1);
        bool h2_round_up = IS_ROUND_UP(a, S4_OP3_IS_RND_2, S4_OP3_RND_BIT_2);
        
        h_1 += (h1_round_up) ? ((a & S4_OP3_RND_BIT_1) << 1) : 0;
        h_2 += (h2_round_up) ? ((a & S4_OP3_RND_BIT_2) << 1) : 0;
        h_1 = (h_1 == S4_OP3_OVRFLW_1) ? S4_MASK1 : h_1;
        h_2 = (h_2 == S4_OP3_OVRFLW_2) ? S4_MASK3 : h_2;
    }

    return (set1) ? h_1 : (set2) ? h_2 : h_3;
}


__device__ int find_4bits_5op(const int &a, const bool &is_round) {
    bool set1 = a & S4_OP5_SET_1;
    bool set2 = a & S4_OP5_SET_2;
    bool set3 = a & S4_OP5_SET_3;
    bool set4 = a & S4_OP5_SET_4;

    int h_1 = a & S4_MASK1;
    int h_2 = a & S4_MASK2;
    int h_3 = a & S4_MASK3;
    int h_4 = a & S4_MASK4;
    int h_5 = a & S4_MASK5;

    if (is_round) {
        bool h1_round_up = IS_ROUND_UP(a, S4_OP5_IS_RND_1, S4_OP5_RND_BIT_1);
        bool h2_round_up = IS_ROUND_UP(a, S4_OP5_IS_RND_2, S4_OP5_RND_BIT_2);
        bool h3_round_up = IS_ROUND_UP(a, S4_OP5_IS_RND_3, S4_OP5_RND_BIT_3);
        bool h4_round_up = IS_ROUND_UP(a, S4_OP5_IS_RND_4, S4_OP5_RND_BIT_4);
        h_1 += (h1_round_up) ? ((a & S4_OP5_RND_BIT_1) << 1) : 0;
        h_2 += (h2_round_up) ? ((a & S4_OP5_RND_BIT_2) << 1) : 0;
        h_3 += (h3_round_up) ? ((a & S4_OP5_RND_BIT_3) << 1) : 0;
        h_4 += (h4_round_up) ? ((a & S4_OP5_RND_BIT_4) << 1) : 0;
        h_1 = (h_1 == S4_OP5_OVRFLW_1) ? S4_MASK1 : h_1;
        h_2 = (h_2 == S4_OP5_OVRFLW_2) ? S4_MASK2 : h_2;
        h_3 = (h_3 == S4_OP5_OVRFLW_3) ? S4_MASK3 : h_3;
        h_4 = (h_4 == S4_OP5_OVRFLW_4) ? S4_MASK4 : h_4;
    }

    return (set1) ? h_1 : (set2) ? h_2 : (set3) ? h_3 : (set4) ? h_4 : h_5;
}


__device__ int find_6bits_3op(const int &a, const bool &is_round){
    bool set1 = a & S6_OP3_SET_1;
    bool set2 = a & S6_OP3_SET_2;

    int h_1 = a & S6_MASK1;
    int h_2 = a & S6_MASK2;
    int h_3 = a & S6_MASK3;

    if (is_round) {
        bool h1_round_up = IS_ROUND_UP(a, S6_OP3_IS_RND_1, S6_OP3_RND_BIT_1);
        bool h2_round_up = IS_ROUND_UP(a, S6_OP3_IS_RND_2, S6_OP3_RND_BIT_2);
        h_1 += (h1_round_up) ? ((a & S6_OP3_RND_BIT_1) << 1) : 0;
        h_2 += (h2_round_up) ? ((a & S6_OP3_RND_BIT_2) << 1) : 0;
        h_1 = (h_1 == S6_OP3_OVRFLW_1) ? S6_MASK1 : h_1;
        h_2 = (h_2 == S6_OP3_OVRFLW_2) ? S6_MASK2 : h_2;
    }

    return (set1) ? h_1 : (set2) ? h_2 : h_3;
}


__device__ int find_3bits_6op(const int &a, const bool &is_round) {
    bool set1 = a & S3_OP6_SET_1;
    bool set2 = a & S3_OP6_SET_2;
    bool set3 = a & S3_OP6_SET_3;
    bool set4 = a & S3_OP6_SET_4;
    bool set5 = a & S3_OP6_SET_5;

    int h_1 = a & S3_MASK1;
    int h_2 = a & S3_MASK2;
    int h_3 = a & S3_MASK3;
    int h_4 = a & S3_MASK4;
    int h_5 = a & S3_MASK5;
    int h_6 = a & S3_MASK6;

    if (is_round) {
        bool h1_round_up = IS_ROUND_UP(a, S3_OP6_IS_RND_1, S3_OP6_RND_BIT_1);
        bool h2_round_up = IS_ROUND_UP(a, S3_OP6_IS_RND_2, S3_OP6_RND_BIT_2);
        bool h3_round_up = IS_ROUND_UP(a, S3_OP6_IS_RND_3, S3_OP6_RND_BIT_3);
        bool h4_round_up = IS_ROUND_UP(a, S3_OP6_IS_RND_4, S3_OP6_RND_BIT_4);
        bool h5_round_up = IS_ROUND_UP(a, S3_OP6_IS_RND_5, S3_OP6_RND_BIT_5);
        h_1 += (h1_round_up) ? ((a & S3_OP6_RND_BIT_1) << 1) : 0;
        h_2 += (h2_round_up) ? ((a & S3_OP6_RND_BIT_2) << 1) : 0;
        h_3 += (h3_round_up) ? ((a & S3_OP6_RND_BIT_3) << 1) : 0;
        h_4 += (h4_round_up) ? ((a & S3_OP6_RND_BIT_4) << 1) : 0;
        h_5 += (h5_round_up) ? ((a & S3_OP6_RND_BIT_5) << 1) : 0;
        h_1 = (h_1 == S3_OP6_OVRFLW_1) ? S3_MASK1 : h_1;
        h_2 = (h_2 == S3_OP6_OVRFLW_2) ? S3_MASK2 : h_2;
        h_3 = (h_3 == S3_OP6_OVRFLW_3) ? S3_MASK3 : h_3;
        h_4 = (h_4 == S3_OP6_OVRFLW_4) ? S3_MASK4 : h_4;
        h_5 = (h_5 == S3_OP6_OVRFLW_5) ? S3_MASK5 : h_5;
    }

    return (set1) ? h_1 : (set2) ? h_2 : (set3) ? h_3 : (set4) ? h_4 : (set5) ? h_5 : h_6;
}


__device__ int find_2bits_7op(const int &a, const bool &is_round) {
    bool set1 = a & S2_OP7_SET_1;
    bool set2 = a & S2_OP7_SET_2;
    bool set3 = a & S2_OP7_SET_3;
    bool set4 = a & S2_OP7_SET_4;
    bool set5 = a & S2_OP7_SET_5;
    bool set6 = a & S2_OP7_SET_6;

    int h_1 = a & S2_MASK1;
    int h_2 = a & S2_MASK2;
    int h_3 = a & S2_MASK3;
    int h_4 = a & S2_MASK4;
    int h_5 = a & S2_MASK5;
    int h_6 = a & S2_MASK6;
    int h_7 = a & S2_MASK7;

    if (is_round) {
        bool h1_round_up = IS_ROUND_UP(a, S2_OP7_IS_RND_1, S2_OP7_RND_BIT_1);  
        bool h2_round_up = IS_ROUND_UP(a, S2_OP7_IS_RND_2, S2_OP7_RND_BIT_2);  
        bool h3_round_up = IS_ROUND_UP(a, S2_OP7_IS_RND_3, S2_OP7_RND_BIT_3);  
        bool h4_round_up = IS_ROUND_UP(a, S2_OP7_IS_RND_4, S2_OP7_RND_BIT_4);  
        bool h5_round_up = IS_ROUND_UP(a, S2_OP7_IS_RND_5, S2_OP7_RND_BIT_5);  
        bool h6_round_up = IS_ROUND_UP(a, S2_OP7_IS_RND_6, S2_OP7_RND_BIT_6);
        h_1 += (h1_round_up) ? ((a & S2_OP7_RND_BIT_1) << 1) : 0;
        h_2 += (h2_round_up) ? ((a & S2_OP7_RND_BIT_2) << 1) : 0;
        h_3 += (h3_round_up) ? ((a & S2_OP7_RND_BIT_3) << 1) : 0;
        h_4 += (h4_round_up) ? ((a & S2_OP7_RND_BIT_4) << 1) : 0;
        h_5 += (h5_round_up) ? ((a & S2_OP7_RND_BIT_5) << 1) : 0;
        h_6 += (h6_round_up) ? ((a & S2_OP7_RND_BIT_6) << 1) : 0;
        h_1 = (h_1 == S2_OP7_OVRFLW_1) ? S2_MASK1 : h_1;
        h_2 = (h_2 == S2_OP7_OVRFLW_2) ? S2_MASK2 : h_2;
        h_3 = (h_3 == S2_OP7_OVRFLW_3) ? S2_MASK3 : h_3;
        h_4 = (h_4 == S2_OP7_OVRFLW_4) ? S2_MASK4 : h_4;
        h_5 = (h_5 == S2_OP7_OVRFLW_5) ? S2_MASK5 : h_5;
        h_6 = (h_6 == S2_OP7_OVRFLW_6) ? S2_MASK6 : h_6;
    }

    return (set1) ? h_1 : (set2) ? h_2 : (set3) ? h_3 : (set4) ? h_4 : (set5) ? h_5 : (set6) ? h_6 : h_7;
}


__device__ int find_bits_slice(const int &a, const bool &is_round, const int &slice_size, const int &num_slices) {
    if (slice_size == 6) {
        // Window size 6 -> 3 placement options
        assert(num_slices == 3);
        return find_6bits_3op(a, is_round);
    }
    else if (slice_size == 4) {
        // Window size 4 -> 2 / 3 / 5 placement options, depends on the configuration
        if(num_slices == 2) {
            return find_4bits_2op(a, is_round);
        }
        else if (num_slices == 3) {
            return find_4bits_3op(a, is_round);
        }
        else if (num_slices == 5) {
            return find_4bits_5op(a, is_round);
        }
        else {
            assert(0);
        }
    }
    else if (slice_size == 3) {
        // Window size 3 -> 6 placement options (we do not support additional options at the moment)
        assert(num_slices == 6);
        return find_3bits_6op(a, is_round);
    
    }
    else if (slice_size == 2) {
        // Window size 2 -> 7 placement options (we do not support additional options at the moment)
        assert(num_slices == 7);
        return find_2bits_7op(a, is_round);
    }
    else {
        assert(0);
    }

    return 0;
}


template <typename scalar_t>
__global__ void gemm_48_opt(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
    const bool is_round,
    const int shift_opt,
    const bool is_stc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> stats) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockDim.x * bx + tx;
    int col = blockDim.y * by + ty;

    scalar_t psum = 0;

    if (row < C.size(0) && col < C.size(1)){
        for (int k = 0; k < A.size(1); k += ((is_stc) ? 4 : 2)) {
            float a_1, a_2, b_1, b_2;

            // 2:4 multiplexers simulation in case of STC
            if (is_stc) {
                a_1 = a_2 = b_1 = b_2 = 0;
                int non_zero_counter = 0;
                
                for(int r = 0; r < 4; r++) {
                    if(k + r < A.size(1)) {
                        if(B[col][k + r] != 0) {
                            if(non_zero_counter == 0) {
                                a_1 = A[row][k + r];
                                b_1 = B[col][k + r];
                            }
                            else {
                                a_2 = A[row][k + r];
                                b_2 = B[col][k + r];
                            }

                            non_zero_counter++;
                        }
                    }
                }

                // Checking that the model is indeed pruned in 2:4 structured manner
                assert(non_zero_counter <= 2);
            }
            else {
                a_1 = A[row][k];
                b_1 = B[col][k];

                a_2 = (k + 1 < A.size(1)) ? A[row][k+1] : 0;
                b_2 = (k + 1 < A.size(1)) ? B[col][k+1] : 0;
            }
            
            if (a_1 == 0) {
                psum += (int)a_2 * (int)b_2;
            }
            else if (a_2 == 0) {
                psum += (int)a_1 * (int)b_1;
            }
            else {
                int a1 = a_1;
                int a2 = a_2;
                int b1 = b_1;
                int b2 = b_2;
                int h1, h2;

                // Checking quantization is fine
                assert((a1 < 256) && (a1 >= 0));
                assert((a2 < 256) && (a2 >= 0));
                assert((b1 <= 127) && (b1 >= -128));
                assert((b2 <= 127) && (b2 >= -128));

                // Trim and round each activation
                h1 = find_bits_slice(a1, is_round, 4, shift_opt);
                h2 = find_bits_slice(a2, is_round, 4, shift_opt);
                
                psum += h1 * b1;
                psum += h2 * b2; 
            }
        }

        C[row][col] = psum;
    }
}


template <typename scalar_t>
__global__ void gemm_48_opt_bitgroup(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
    const bool is_round,
    const int shift_opt,
    const int bit_group,
    const bool is_stc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> stats) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockDim.x * bx + tx;
    int col = blockDim.y * by + ty;

    scalar_t psum = 0;

    if (row < C.size(0) && col < C.size(1)) {
        for (int k = 0; k < A.size(1); k += ((is_stc) ? 4 : 2)) {
            float a_1, a_2, b_1, b_2;

            if (is_stc) {
                a_1 = a_2 = b_1 = b_2 = 0;
                int non_zero_counter = 0;

                for (int r = 0; r < 4; r++) {
                    if (k + r < A.size(1)) {
                        if (B[col][k + r] != 0) {
                            if (non_zero_counter == 0) {
                                a_1 = A[row][k + r];
                                b_1 = B[col][k + r];
                            }
                            else {
                                a_2 = A[row][k + r];
                                b_2 = B[col][k + r];
                            }

                            non_zero_counter++;
                        }
                    }
                }

                // Checking that the model is indeed pruned in 2:4 structured manner
                assert(non_zero_counter <= 2);
            }
            else {
                a_1 = A[row][k];
                b_1 = B[col][k];

                a_2 = (k + 1 < A.size(1)) ? A[row][k+1] : 0;
                b_2 = (k + 1 < A.size(1)) ? B[col][k+1] : 0;
            }

            int a1 = a_1;
            int a2 = a_2;
            int b1 = b_1;
            int b2 = b_2;
            int h1 = 0;
            int h2 = 0;

            // Checking quantization is fine
            assert((a1 < 256) && (a1 >= 0));
            assert((a2 < 256) && (a2 >= 0));
            assert((b1 <= 127) && (b1 >= -128)); 
            assert((b2 <= 127) && (b2 >= -128)); 

            if (a1 == 0) {
                // 2n bits in 8-2n+1 slices, e.g., 6 bits in 3 slices
                h2 = find_bits_slice(a2, is_round, 2*bit_group, 8 - 2*bit_group + 1);
            }
            else if (a2 == 0) {
                h1 = find_bits_slice(a1, is_round, 2*bit_group, 8 - 2*bit_group + 1);
            }
            else {
                h1 = find_bits_slice(a1, is_round, bit_group, shift_opt);
                h2 = find_bits_slice(a2, is_round, bit_group, shift_opt);
            }
            
            psum += h1 * b1;
            psum += h2 * b2;
        }

        C[row][col] = psum;
    }
}


std::vector<torch::Tensor> gemm_48_opt_cuda(
    torch::Tensor a,
	torch::Tensor b,
    const bool is_round,
    const int shift_opt,
    const bool is_stc) {

    torch::Device device = torch::kCUDA;

    auto output = torch::zeros({a.size(0), b.size(0)}, device);
    auto stats = torch::zeros({8, a.size(0), b.size(0)}, device);

    const int block_size = 16;
    const dim3 threads(block_size, block_size);
    const dim3 grid(DIVCEIL(output.size(0), threads.x), DIVCEIL(output.size(1), threads.y));

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "_gemm", ([&] {
        gemm_48_opt<scalar_t><<< grid, threads >>>(
          a.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          is_round,
          shift_opt,
          is_stc,
          stats.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
        );
    }));

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error: (%d) %s\n", code, errorMessage);
    }

    return {output, stats};
}


std::vector<torch::Tensor> gemm_48_opt_bitgroup_cuda(
	torch::Tensor a,
	torch::Tensor b,
    const bool is_round,
    const int shift_opt,
    const int bit_group,
    const bool is_stc) {

    torch::Device device = torch::kCUDA;

    auto output = torch::zeros({a.size(0), b.size(0)}, device);
    auto stats = torch::zeros({8, a.size(0), b.size(0)}, device);

    const int block_size = 16;
    const dim3 threads(block_size, block_size);
    const dim3 grid(DIVCEIL(output.size(0), threads.x), DIVCEIL(output.size(1), threads.y));

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "_gemm", ([&] {
        gemm_48_opt_bitgroup<scalar_t><<< grid, threads >>>(
          a.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          is_round,
          shift_opt,
          bit_group,
          is_stc,
          stats.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
        );
    }));

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error: (%d) %s\n", code, errorMessage);
    }

    return {output, stats};
}
