
#include "openfhe/pke/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT external_relu_secret();
CiphertextT mlp(CryptoContextT cc, CiphertextT ct);
CiphertextT mlp__encrypt__arg0(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk);
std::vector<float> mlp__decrypt__result0(CryptoContextT cc, CiphertextT ct, PrivateKeyT sk);
CryptoContextT mlp__generate_crypto_context();
CryptoContextT mlp__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk);
