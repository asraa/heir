#include <cerrno>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h>

#include "mnist_openfhe.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template <int N>
int argmax(float *A) {
  int max_idx = 0;
  for (int i = 1; i < N; i++) {
    if (A[i] > A[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}

int main(int argc, char *argv[]) {
  const std::string model_path = "./traced_model.pt";
  const std::string MNIST_data_path = "./data/MNIST/raw";

  std::vector<std::vector<float>> weights;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval(); // Don't forget to set evaluation mode

    std::cout << "Successfully loaded " << model_path << std::endl;

    // Access and print parameter names and shapes
    std::cout << "Model parameters:" << std::endl;
    for (auto pair : module.named_parameters()) {
      const std::string& name = pair.name;
      const torch::Tensor& tensor = pair.value;
      std::cout << "  " << name << ": " << tensor.sizes() << std::endl;

      auto tensorCont = tensor.contiguous();
      int64_t num_elements = tensorCont.numel();
      const float* tensor_data = tensorCont.data_ptr<float>();

      weights.push_back({tensor_data, tensor_data + num_elements});
    }
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model: " << e.msg() << std::endl;
    return -1;
  }

  auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(test_dataset), 1);


  auto cryptoContext = mnist__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = mnist__configure_crypto_context(cryptoContext, secretKey);

  std::cout << *cryptoContext->GetCryptoParameters() << std::endl;

  int total = 10;
  for (auto& batch : *test_loader) {
    if (total == 0) break;
    torch::Tensor input_tensor = batch.data.contiguous();
    float* tensor_data_ptr = input_tensor.data_ptr<float>();

    std::vector<float> input_vector(tensor_data_ptr, tensor_data_ptr + input_tensor.numel());
    auto input_encrypted =
        mnist__encrypt__arg4(cryptoContext, input_vector, publicKey);
    
    auto t1 = high_resolution_clock::now();
    auto output_encrypted = mnist(cryptoContext, weights[0], weights[1], weights[2], weights[3], input_encrypted);
    auto t2 = high_resolution_clock::now();

    std::vector<float> output =
        mnist__decrypt__result0(cryptoContext, output_encrypted, secretKey);


    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "FHE computation time: " << ms_int.count() << "ms\n";

    torch::Tensor label_tensor = batch.target;
    int64_t label = label_tensor.item<int64_t>();
    auto max_id = argmax<10>(output.data());
    std::cout << "max_id: " << max_id << ", label: " << label << std::endl;

    total--;
  }

  return 0;
}