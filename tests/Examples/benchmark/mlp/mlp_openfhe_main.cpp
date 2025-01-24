#include <cerrno>
#include <fstream>
#include <iostream>
#include <vector>

#include "mlp_openfhe.h"

#define DIM 1024

struct Sample {
  int label;
  float image[DIM];
};

using Dataset = std::vector<Sample>;

void load_dataset(Dataset &dataset, const char *filename) {
  std::ifstream file(filename);
  Sample sample;
  while (file >> sample.label) {
    for (int i = 0; i < DIM; i++) {
      file >> sample.image[i];
    }
    dataset.push_back(sample);
  }
}

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
  auto dataset = Dataset();
  load_dataset(dataset, "mnist_test.txt");

  int accurate = 0;
  int total = 10;  // 10000

  auto cryptoContext = mlp__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = mlp__configure_crypto_context(cryptoContext, secretKey);

  std::cout << *cryptoContext->GetCryptoParameters() << std::endl;

  for (int i = 0; i < total; ++i) {
    auto *input = dataset[i].image;

    std::vector<float> input_vector(input, input + DIM);

    auto input_encrypted =
        mlp__encrypt__arg0(cryptoContext, input_vector, publicKey);
    auto output_encrypted = mlp(cryptoContext, input_encrypted);
    std::vector<float> output =
        mlp__decrypt__result0(cryptoContext, output_encrypted, secretKey);

    auto max_id = argmax<1024>(output.data());
    auto label = dataset[i].label;

    std::cout << "max_id: " << max_id << ", label: " << label << std::endl;

    if (max_id == label) {
      accurate++;
    }
    // if (i % 100 == 0) {
    //   std::cout << "accuracy: " << accurate << "/" << i << std::endl;
    // }
  }

  std::cout << "accuracy: " << accurate << "/" << total << std::endl;

  return 0;
}
