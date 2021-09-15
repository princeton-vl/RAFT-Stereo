#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


std::vector<torch::Tensor>  sampler_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor>  sampler_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sampler_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return sampler_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> sampler_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  auto volume_grad = sampler_cuda_backward(volume, coords, corr_grad, radius);
  return {volume_grad};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sampler_forward, "SAMPLER forward");
  m.def("backward", &sampler_backward, "SAMPLER backward");
}