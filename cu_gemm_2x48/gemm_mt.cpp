#include <torch/extension.h>

#include <vector>
#include <assert.h>

// CUDA forward declerations

std::vector<torch::Tensor> gemm_48_opt_cuda(
	torch::Tensor a,
	torch::Tensor b,
    const bool is_round,
    const int shift_opt,
    const bool is_stc);


std::vector<torch::Tensor> gemm_48_opt_bitgroup_cuda(
	torch::Tensor a,
	torch::Tensor b,
    const bool is_round,
    const int shift_opt,
    const int bit_group,
    const bool is_stc);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> gemm_nbsmt_48_opt_aux(
	torch::Tensor a,
	torch::Tensor b,
    const bool is_round,
    const int shift_opt,
    const int bit_group,
    const bool is_stc) {

    CHECK_INPUT(a);
    CHECK_INPUT(b);
    if (bit_group == 0 || bit_group == 4){     
        return gemm_48_opt_cuda(a, b, is_round, shift_opt, is_stc);
    }else{
        int shift_sets;
        if(bit_group == 3){
            shift_sets = 6;
        }else if(bit_group == 2){
            shift_sets = 7; 
        }else{
            assert(0);
        }
        return gemm_48_opt_bitgroup_cuda(a, b, is_round, shift_sets, bit_group, is_stc);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &gemm_nbsmt_48_opt_aux, "GEMM 2x4b-8b simulation");
}

