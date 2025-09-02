
#include "openfhe/pke/openfhe.h" 

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using MutableCiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

std::vector<CiphertextT> mnist(CryptoContextT cc, std::vector<float> v0, std::vector<float> v1, std::vector<float> v2, std::vector<float> v3, std::vector<CiphertextT> v4);
std::vector<CiphertextT> mnist__encrypt__arg4(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk);
std::vector<float> mnist__decrypt__result0(CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk);
CryptoContextT mnist__generate_crypto_context();
CryptoContextT mnist__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk);
