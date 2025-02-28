
#include "openfhe/pke/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using NonConstCiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/cereal.hpp"

struct Weights {
  std::map<std::string, std::vector<float>> floats;
  std::map<std::string, std::vector<double>> doubles;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(CEREAL_NVP(floats), CEREAL_NVP(doubles));
  }
};

Weights GetWeightModule(const std::string& filename) {
  Weights obj;
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive archive(file);
  archive(obj);
  file.close();
  return obj;
}

std::vector<double> ReadDoubleVecFromModule(const std::string& name,
                                            Weights module) {
  return module.doubles[name];
}

std::vector<float> ReadFloatVecFromModule(const std::string& name,
                                          Weights module) {
  return module.floats[name];
}

CiphertextT external_relu_secret(CryptoContextT cc, CiphertextT ct,
                                 PrivateKeyT sk, PublicKeyT pk) {
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& v0_cast = pt->GetCKKSPackedValue();
  std::vector<float> v0(v0_cast.size());
  std::transform(std::begin(v0_cast), std::end(v0_cast), std::begin(v0),
                 [](const std::complex<double>& c) {
                   return std::max((double)0, c.real());
                 });
  std::vector<double> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt_new = cc->MakeCKKSPackedPlaintext(v1_filled);
  const auto& ct_new = cc->Encrypt(pk, pt_new);
  return ct_new;
}

CiphertextT mlp(CryptoContextT cc, CiphertextT ct, PrivateKeyT sk,
                PublicKeyT pk) {
  Weights module = GetWeightModule("/tmp/mlp_weights.bin");
  std::vector<double> v0 = ReadDoubleVecFromModule("v0", module);
  std::vector<double> v1(1024, 0.000000e+00);
  std::vector<float> v2 = ReadFloatVecFromModule("v2", module);
  std::vector<float> v3 = ReadFloatVecFromModule("v3", module);
  std::vector<double> v4 = ReadDoubleVecFromModule("v4", module);
  auto v4_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v4_filled = v4;
  v4_filled.clear();
  v4_filled.reserve(v4_filled_n);
  for (auto i = 0; i < v4_filled_n; ++i) {
    v4_filled.push_back(v4[i % v4.size()]);
  }
  const auto& pt = cc->MakeCKKSPackedPlaintext(v4_filled);
  const auto& ct1 = cc->EvalMult(ct, pt);
  auto v1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt1 = cc->MakeCKKSPackedPlaintext(v1_filled);
  const auto& ct2 = cc->EvalAdd(ct1, pt1);
  NonConstCiphertextT ct3 = ct2->Clone();
  NonConstCiphertextT ct4 = ct->Clone();
  for (auto v5 = 1; v5 < 1024; ++v5) {
    ct4 = cc->EvalRotate(ct4, 1);
    std::vector<double> v6(std::begin(v3) + (v5 * 1024),
                           std::end(v3) + (v5 * 1024) + 1024);
    std::vector<double> v7(std::begin(v6), std::end(v6));
    auto v7_filled_n =
        cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
    auto v7_filled = v7;
    v7_filled.clear();
    v7_filled.reserve(v7_filled_n);
    for (auto i = 0; i < v7_filled_n; ++i) {
      v7_filled.push_back(v7[i % v7.size()]);
    }
    const auto& pt2 = cc->MakeCKKSPackedPlaintext(v7_filled);
    const auto& ct8 = cc->EvalMult(ct4, pt2);
    ct3 = cc->EvalAdd(ct3, ct8);
  }
  const auto& ct10 = external_relu_secret(cc, ct3, sk, pk);
  const auto& ct11 = cc->ModReduce(ct10);
  auto v0_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v0_filled = v0;
  v0_filled.clear();
  v0_filled.reserve(v0_filled_n);
  for (auto i = 0; i < v0_filled_n; ++i) {
    v0_filled.push_back(v0[i % v0.size()]);
  }
  const auto& pt3 = cc->MakeCKKSPackedPlaintext(v0_filled);
  const auto& ct12 = cc->EvalMult(ct11, pt3);
  const auto& ct13 = cc->EvalAdd(ct12, pt1);
  NonConstCiphertextT ct14 = ct13->Clone();
  NonConstCiphertextT ct15 = ct10->Clone();
  for (auto v8 = 1; v8 < 1024; ++v8) {
    ct15 = cc->EvalRotate(ct15, 1);
    std::vector<double> v9(std::begin(v2) + v8 * 1024,
                           std::end(v2) + v8 * 1024 + 1024);
    const auto& ct19 = cc->ModReduce(ct15);
    std::vector<double> v10(std::begin(v9), std::end(v9));
    auto v10_filled_n =
        cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
    auto v10_filled = v10;
    v10_filled.clear();
    v10_filled.reserve(v10_filled_n);
    for (auto i = 0; i < v10_filled_n; ++i) {
      v10_filled.push_back(v10[i % v10.size()]);
    }
    const auto& pt4 = cc->MakeCKKSPackedPlaintext(v10_filled);
    const auto& ct20 = cc->EvalMult(ct19, pt4);
    ct14 = cc->EvalAdd(ct14, ct20);
  }
  const auto& ct22 = cc->ModReduce(ct14);
  return ct22;
}
CiphertextT mlp__encrypt__arg0(CryptoContextT cc, std::vector<float> v0,
                               PublicKeyT pk) {
  std::vector<double> v1(std::begin(v0), std::end(v0));
  auto v1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v1_filled = v1;
  v1_filled.clear();
  v1_filled.reserve(v1_filled_n);
  for (auto i = 0; i < v1_filled_n; ++i) {
    v1_filled.push_back(v1[i % v1.size()]);
  }
  const auto& pt = cc->MakeCKKSPackedPlaintext(v1_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  return ct;
}
std::vector<float> mlp__decrypt__result0(CryptoContextT cc, CiphertextT ct,
                                         PrivateKeyT sk) {
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& v0_cast = pt->GetCKKSPackedValue();
  std::vector<float> v0(v0_cast.size());
  std::transform(std::begin(v0_cast), std::end(v0_cast), std::begin(v0),
                 [](const std::complex<double>& c) { return c.real(); });
  return v0;
}
CryptoContextT mlp__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(2);
  params.SetSecurityLevel(HEStd_NotSet);
  params.SetRingDim(1 << 11);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(LEVELEDSHE);
  cc->Enable(KEYSWITCH);
  return cc;
}
CryptoContextT mlp__configure_crypto_context(CryptoContextT cc,
                                             PrivateKeyT sk) {
  cc->EvalRotateKeyGen(sk, {1});
  return cc;
}
