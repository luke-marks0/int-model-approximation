#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void launch_fp8_e4m3_wgmma_tile(
    const void *a,
    const void *b,
    const void *c,
    void *d,
    cudaStream_t stream);

torch::Tensor fp8_e4m3_wgmma_tile(
    torch::Tensor a,
    torch::Tensor b,
    torch::optional<torch::Tensor> c) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == 64 && a.size(1) == 32, "a must be [64,32]");
  TORCH_CHECK(b.dim() == 2 && b.size(0) == 128 && b.size(1) == 32, "b must be [128,32]");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "a must be float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "b must be float8_e4m3fn");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");

  const void *c_ptr = nullptr;
  if (c.has_value()) {
    torch::Tensor c_tensor = c.value();
    TORCH_CHECK(c_tensor.is_cuda() && c_tensor.is_contiguous(), "c must be CUDA contiguous");
    TORCH_CHECK(c_tensor.device() == a.device(), "c must be on the same device as a");
    TORCH_CHECK(
        c_tensor.dim() == 2 && c_tensor.size(0) == 64 && c_tensor.size(1) == 128,
        "c must be [64,128]");
    TORCH_CHECK(c_tensor.scalar_type() == torch::kFloat32, "c must be float32");
    c_ptr = c_tensor.data_ptr();
  }

  auto d = torch::empty({64, 128}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
  launch_fp8_e4m3_wgmma_tile(
      a.data_ptr(),
      b.data_ptr(),
      c_ptr,
      d.data_ptr(),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fp8_e4m3_wgmma_tile",
      &fp8_e4m3_wgmma_tile,
      "Hopper WGMMA FP8 E4M3 m64n128k32 tile",
      py::arg("a"),
      py::arg("b"),
      py::arg("c") = py::none());
}
