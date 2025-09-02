
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

std::vector<CiphertextT> mnist(CryptoContextT cc, std::vector<float> v0, std::vector<float> v1, std::vector<float> v2, std::vector<float> v3, std::vector<CiphertextT> v4) {
  [[maybe_unused]] size_t v5 = 511;
  [[maybe_unused]] size_t v6 = 783;
  [[maybe_unused]] size_t v7 = 1535;
  [[maybe_unused]] size_t v8 = 6;
  [[maybe_unused]] size_t v9 = 9;
  [[maybe_unused]] size_t v10 = 1018;
  [[maybe_unused]] size_t v11 = 16;
  std::vector<float> v12(16384, 0);
  float v13 = 0.038310006260871887;
  float v14 = 0.5;
  float v15 = 0.93702799081802368;
  float v16 = 3.9769232503716317E-17;
  float v17 = -0.50627744197845459;
  float v18 = -1.5903312252962446E-16;
  float v19 = 0;
  std::vector<float> v20(1024, 0);
  [[maybe_unused]] size_t v21 = 1807;
  [[maybe_unused]] size_t v22 = 240;
  [[maybe_unused]] size_t v23 = 1024;
  [[maybe_unused]] size_t v24 = 512;
  [[maybe_unused]] size_t v25 = 1;
  [[maybe_unused]] size_t v26 = 0;
  std::vector<float> v27(524288, 0);
  std::vector<float> v28 = v27;
  for (auto v29 = 0; v29 < 512; ++v29) {
    for (auto v32 = 0; v32 < 1024; ++v32) {
      size_t v34 = v29 + v32;
      size_t v35 = v34 + v22;
      size_t v36 = v35 % v23;
      bool v37 = v36 >= v22;
      if (v37) {
        size_t v39 = v32 % v24;
        size_t v40 = v26 - v29;
        size_t v41 = v40 - v32;
        size_t v42 = v41 + v21;
        size_t v43 = v42 % v23;
        size_t v44 = v6 - v43;
        float v45 = v0[v44 + 784 * (v39)];
        v28[v32 + 1024 * (v29)] = v45;
      }
    }
  }
  std::vector<float> v47 = v20;
  for (auto v48 = 0; v48 < 1024; ++v48) {
    v47[v48 + 1024 * (0)] = v19;
  }
  const auto& ct = v4[0];
  std::vector<float> v51(std::begin(v28) + 0 * 512, std::begin(v28) + 0 * 512 + 1024);
  std::vector<double> v52(std::begin(v51), std::end(v51));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v52;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (auto i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v52[i % v52.size()]);
  }
  auto pt = cc->MakeCKKSPackedPlaintext(pt_filled);
  const auto& ct1 = cc->EvalMult(ct, pt);
  std::vector<float> v53(std::begin(v28) + 1 * 512, std::begin(v28) + 1 * 512 + 1024);
  const auto& ct2 = cc->EvalRotate(ct, 1);
  std::vector<double> v54(std::begin(v53), std::end(v53));
  auto pt1_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v54;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (auto i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v54[i % v54.size()]);
  }
  auto pt1 = cc->MakeCKKSPackedPlaintext(pt1_filled);
  const auto& ct3 = cc->EvalMult(ct2, pt1);
  std::vector<float> v55(std::begin(v28) + 2 * 512, std::begin(v28) + 2 * 512 + 1024);
  const auto& ct4 = cc->EvalRotate(ct, 2);
  std::vector<double> v56(std::begin(v55), std::end(v55));
  auto pt2_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v56;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (auto i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v56[i % v56.size()]);
  }
  auto pt2 = cc->MakeCKKSPackedPlaintext(pt2_filled);
  const auto& ct5 = cc->EvalMult(ct4, pt2);
  std::vector<float> v57(std::begin(v28) + 3 * 512, std::begin(v28) + 3 * 512 + 1024);
  const auto& ct6 = cc->EvalRotate(ct, 3);
  std::vector<double> v58(std::begin(v57), std::end(v57));
  auto pt3_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt3_filled = v58;
  pt3_filled.clear();
  pt3_filled.reserve(pt3_filled_n);
  for (auto i = 0; i < pt3_filled_n; ++i) {
    pt3_filled.push_back(v58[i % v58.size()]);
  }
  auto pt3 = cc->MakeCKKSPackedPlaintext(pt3_filled);
  const auto& ct7 = cc->EvalMult(ct6, pt3);
  std::vector<float> v59(std::begin(v28) + 4 * 512, std::begin(v28) + 4 * 512 + 1024);
  const auto& ct8 = cc->EvalRotate(ct, 4);
  std::vector<double> v60(std::begin(v59), std::end(v59));
  auto pt4_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt4_filled = v60;
  pt4_filled.clear();
  pt4_filled.reserve(pt4_filled_n);
  for (auto i = 0; i < pt4_filled_n; ++i) {
    pt4_filled.push_back(v60[i % v60.size()]);
  }
  auto pt4 = cc->MakeCKKSPackedPlaintext(pt4_filled);
  const auto& ct9 = cc->EvalMult(ct8, pt4);
  std::vector<float> v61(std::begin(v28) + 5 * 512, std::begin(v28) + 5 * 512 + 1024);
  const auto& ct10 = cc->EvalRotate(ct, 5);
  std::vector<double> v62(std::begin(v61), std::end(v61));
  auto pt5_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt5_filled = v62;
  pt5_filled.clear();
  pt5_filled.reserve(pt5_filled_n);
  for (auto i = 0; i < pt5_filled_n; ++i) {
    pt5_filled.push_back(v62[i % v62.size()]);
  }
  auto pt5 = cc->MakeCKKSPackedPlaintext(pt5_filled);
  const auto& ct11 = cc->EvalMult(ct10, pt5);
  std::vector<float> v63(std::begin(v28) + 6 * 512, std::begin(v28) + 6 * 512 + 1024);
  const auto& ct12 = cc->EvalRotate(ct, 6);
  std::vector<double> v64(std::begin(v63), std::end(v63));
  auto pt6_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt6_filled = v64;
  pt6_filled.clear();
  pt6_filled.reserve(pt6_filled_n);
  for (auto i = 0; i < pt6_filled_n; ++i) {
    pt6_filled.push_back(v64[i % v64.size()]);
  }
  auto pt6 = cc->MakeCKKSPackedPlaintext(pt6_filled);
  const auto& ct13 = cc->EvalMult(ct12, pt6);
  std::vector<float> v65(std::begin(v28) + 7 * 512, std::begin(v28) + 7 * 512 + 1024);
  const auto& ct14 = cc->EvalRotate(ct, 7);
  std::vector<double> v66(std::begin(v65), std::end(v65));
  auto pt7_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt7_filled = v66;
  pt7_filled.clear();
  pt7_filled.reserve(pt7_filled_n);
  for (auto i = 0; i < pt7_filled_n; ++i) {
    pt7_filled.push_back(v66[i % v66.size()]);
  }
  auto pt7 = cc->MakeCKKSPackedPlaintext(pt7_filled);
  const auto& ct15 = cc->EvalMult(ct14, pt7);
  std::vector<float> v67(std::begin(v28) + 8 * 512, std::begin(v28) + 8 * 512 + 1024);
  const auto& ct16 = cc->EvalRotate(ct, 8);
  std::vector<double> v68(std::begin(v67), std::end(v67));
  auto pt8_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt8_filled = v68;
  pt8_filled.clear();
  pt8_filled.reserve(pt8_filled_n);
  for (auto i = 0; i < pt8_filled_n; ++i) {
    pt8_filled.push_back(v68[i % v68.size()]);
  }
  auto pt8 = cc->MakeCKKSPackedPlaintext(pt8_filled);
  const auto& ct17 = cc->EvalMult(ct16, pt8);
  std::vector<float> v69(std::begin(v28) + 9 * 512, std::begin(v28) + 9 * 512 + 1024);
  const auto& ct18 = cc->EvalRotate(ct, 9);
  std::vector<double> v70(std::begin(v69), std::end(v69));
  auto pt9_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt9_filled = v70;
  pt9_filled.clear();
  pt9_filled.reserve(pt9_filled_n);
  for (auto i = 0; i < pt9_filled_n; ++i) {
    pt9_filled.push_back(v70[i % v70.size()]);
  }
  auto pt9 = cc->MakeCKKSPackedPlaintext(pt9_filled);
  const auto& ct19 = cc->EvalMult(ct18, pt9);
  std::vector<float> v71(std::begin(v28) + 10 * 512, std::begin(v28) + 10 * 512 + 1024);
  const auto& ct20 = cc->EvalRotate(ct, 10);
  std::vector<double> v72(std::begin(v71), std::end(v71));
  auto pt10_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt10_filled = v72;
  pt10_filled.clear();
  pt10_filled.reserve(pt10_filled_n);
  for (auto i = 0; i < pt10_filled_n; ++i) {
    pt10_filled.push_back(v72[i % v72.size()]);
  }
  auto pt10 = cc->MakeCKKSPackedPlaintext(pt10_filled);
  const auto& ct21 = cc->EvalMult(ct20, pt10);
  std::vector<float> v73(std::begin(v28) + 11 * 512, std::begin(v28) + 11 * 512 + 1024);
  const auto& ct22 = cc->EvalRotate(ct, 11);
  std::vector<double> v74(std::begin(v73), std::end(v73));
  auto pt11_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt11_filled = v74;
  pt11_filled.clear();
  pt11_filled.reserve(pt11_filled_n);
  for (auto i = 0; i < pt11_filled_n; ++i) {
    pt11_filled.push_back(v74[i % v74.size()]);
  }
  auto pt11 = cc->MakeCKKSPackedPlaintext(pt11_filled);
  const auto& ct23 = cc->EvalMult(ct22, pt11);
  std::vector<float> v75(std::begin(v28) + 12 * 512, std::begin(v28) + 12 * 512 + 1024);
  const auto& ct24 = cc->EvalRotate(ct, 12);
  std::vector<double> v76(std::begin(v75), std::end(v75));
  auto pt12_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt12_filled = v76;
  pt12_filled.clear();
  pt12_filled.reserve(pt12_filled_n);
  for (auto i = 0; i < pt12_filled_n; ++i) {
    pt12_filled.push_back(v76[i % v76.size()]);
  }
  auto pt12 = cc->MakeCKKSPackedPlaintext(pt12_filled);
  const auto& ct25 = cc->EvalMult(ct24, pt12);
  std::vector<float> v77(std::begin(v28) + 13 * 512, std::begin(v28) + 13 * 512 + 1024);
  const auto& ct26 = cc->EvalRotate(ct, 13);
  std::vector<double> v78(std::begin(v77), std::end(v77));
  auto pt13_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt13_filled = v78;
  pt13_filled.clear();
  pt13_filled.reserve(pt13_filled_n);
  for (auto i = 0; i < pt13_filled_n; ++i) {
    pt13_filled.push_back(v78[i % v78.size()]);
  }
  auto pt13 = cc->MakeCKKSPackedPlaintext(pt13_filled);
  const auto& ct27 = cc->EvalMult(ct26, pt13);
  std::vector<float> v79(std::begin(v28) + 14 * 512, std::begin(v28) + 14 * 512 + 1024);
  const auto& ct28 = cc->EvalRotate(ct, 14);
  std::vector<double> v80(std::begin(v79), std::end(v79));
  auto pt14_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt14_filled = v80;
  pt14_filled.clear();
  pt14_filled.reserve(pt14_filled_n);
  for (auto i = 0; i < pt14_filled_n; ++i) {
    pt14_filled.push_back(v80[i % v80.size()]);
  }
  auto pt14 = cc->MakeCKKSPackedPlaintext(pt14_filled);
  const auto& ct29 = cc->EvalMult(ct28, pt14);
  std::vector<float> v81(std::begin(v28) + 15 * 512, std::begin(v28) + 15 * 512 + 1024);
  const auto& ct30 = cc->EvalRotate(ct, 15);
  std::vector<double> v82(std::begin(v81), std::end(v81));
  auto pt15_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt15_filled = v82;
  pt15_filled.clear();
  pt15_filled.reserve(pt15_filled_n);
  for (auto i = 0; i < pt15_filled_n; ++i) {
    pt15_filled.push_back(v82[i % v82.size()]);
  }
  auto pt15 = cc->MakeCKKSPackedPlaintext(pt15_filled);
  const auto& ct31 = cc->EvalMult(ct30, pt15);
  const auto& ct32 = cc->EvalAdd(ct1, ct3);
  const auto& ct33 = cc->EvalAdd(ct5, ct7);
  const auto& ct34 = cc->EvalAdd(ct32, ct33);
  const auto& ct35 = cc->EvalAdd(ct9, ct11);
  const auto& ct36 = cc->EvalAdd(ct13, ct15);
  const auto& ct37 = cc->EvalAdd(ct35, ct36);
  const auto& ct38 = cc->EvalAdd(ct34, ct37);
  const auto& ct39 = cc->EvalAdd(ct17, ct19);
  const auto& ct40 = cc->EvalAdd(ct21, ct23);
  const auto& ct41 = cc->EvalAdd(ct39, ct40);
  const auto& ct42 = cc->EvalAdd(ct25, ct27);
  const auto& ct43 = cc->EvalAdd(ct29, ct31);
  const auto& ct44 = cc->EvalAdd(ct42, ct43);
  const auto& ct45 = cc->EvalAdd(ct41, ct44);
  const auto& ct46 = cc->EvalAdd(ct38, ct45);
  std::vector<float> v83(std::begin(v28) + 16 * 512, std::begin(v28) + 16 * 512 + 1024);
  std::vector<float> v84(1008);
  std::copy(v83.begin() + 0, v83.begin() + 0 + 1008, v84.begin());
  std::vector<float> v85(16);
  std::copy(v83.begin() + 1008, v83.begin() + 1008 + 16, v85.begin());
  std::vector<float> v86(1024);
  std::copy(v84.begin(), v84.end(), v86.begin() + 16);
  std::copy(v85.begin(), v85.end(), v86.begin() + 0);
  std::vector<double> v89(std::begin(v86), std::end(v86));
  auto pt16_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt16_filled = v89;
  pt16_filled.clear();
  pt16_filled.reserve(pt16_filled_n);
  for (auto i = 0; i < pt16_filled_n; ++i) {
    pt16_filled.push_back(v89[i % v89.size()]);
  }
  auto pt16 = cc->MakeCKKSPackedPlaintext(pt16_filled);
  const auto& ct47 = cc->EvalMult(ct, pt16);
  std::vector<float> v90(std::begin(v28) + 17 * 512, std::begin(v28) + 17 * 512 + 1024);
  std::vector<float> v91(1008);
  std::copy(v90.begin() + 0, v90.begin() + 0 + 1008, v91.begin());
  std::vector<float> v92(16);
  std::copy(v90.begin() + 1008, v90.begin() + 1008 + 16, v92.begin());
  std::copy(v91.begin(), v91.end(), v86.begin() + 16);
  std::copy(v92.begin(), v92.end(), v86.begin() + 0);
  std::vector<double> v95(std::begin(v86), std::end(v86));
  auto pt17_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt17_filled = v95;
  pt17_filled.clear();
  pt17_filled.reserve(pt17_filled_n);
  for (auto i = 0; i < pt17_filled_n; ++i) {
    pt17_filled.push_back(v95[i % v95.size()]);
  }
  auto pt17 = cc->MakeCKKSPackedPlaintext(pt17_filled);
  const auto& ct48 = cc->EvalMult(ct2, pt17);
  std::vector<float> v96(std::begin(v28) + 18 * 512, std::begin(v28) + 18 * 512 + 1024);
  std::vector<float> v97(1008);
  std::copy(v96.begin() + 0, v96.begin() + 0 + 1008, v97.begin());
  std::vector<float> v98(16);
  std::copy(v96.begin() + 1008, v96.begin() + 1008 + 16, v98.begin());
  std::copy(v97.begin(), v97.end(), v86.begin() + 16);
  std::copy(v98.begin(), v98.end(), v86.begin() + 0);
  std::vector<double> v101(std::begin(v86), std::end(v86));
  auto pt18_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt18_filled = v101;
  pt18_filled.clear();
  pt18_filled.reserve(pt18_filled_n);
  for (auto i = 0; i < pt18_filled_n; ++i) {
    pt18_filled.push_back(v101[i % v101.size()]);
  }
  auto pt18 = cc->MakeCKKSPackedPlaintext(pt18_filled);
  const auto& ct49 = cc->EvalMult(ct4, pt18);
  std::vector<float> v102(std::begin(v28) + 19 * 512, std::begin(v28) + 19 * 512 + 1024);
  std::vector<float> v103(1008);
  std::copy(v102.begin() + 0, v102.begin() + 0 + 1008, v103.begin());
  std::vector<float> v104(16);
  std::copy(v102.begin() + 1008, v102.begin() + 1008 + 16, v104.begin());
  std::copy(v103.begin(), v103.end(), v86.begin() + 16);
  std::copy(v104.begin(), v104.end(), v86.begin() + 0);
  std::vector<double> v107(std::begin(v86), std::end(v86));
  auto pt19_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt19_filled = v107;
  pt19_filled.clear();
  pt19_filled.reserve(pt19_filled_n);
  for (auto i = 0; i < pt19_filled_n; ++i) {
    pt19_filled.push_back(v107[i % v107.size()]);
  }
  auto pt19 = cc->MakeCKKSPackedPlaintext(pt19_filled);
  const auto& ct50 = cc->EvalMult(ct6, pt19);
  std::vector<float> v108(std::begin(v28) + 20 * 512, std::begin(v28) + 20 * 512 + 1024);
  std::vector<float> v109(1008);
  std::copy(v108.begin() + 0, v108.begin() + 0 + 1008, v109.begin());
  std::vector<float> v110(16);
  std::copy(v108.begin() + 1008, v108.begin() + 1008 + 16, v110.begin());
  std::copy(v109.begin(), v109.end(), v86.begin() + 16);
  std::copy(v110.begin(), v110.end(), v86.begin() + 0);
  std::vector<double> v113(std::begin(v86), std::end(v86));
  auto pt20_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt20_filled = v113;
  pt20_filled.clear();
  pt20_filled.reserve(pt20_filled_n);
  for (auto i = 0; i < pt20_filled_n; ++i) {
    pt20_filled.push_back(v113[i % v113.size()]);
  }
  auto pt20 = cc->MakeCKKSPackedPlaintext(pt20_filled);
  const auto& ct51 = cc->EvalMult(ct8, pt20);
  std::vector<float> v114(std::begin(v28) + 21 * 512, std::begin(v28) + 21 * 512 + 1024);
  std::vector<float> v115(1008);
  std::copy(v114.begin() + 0, v114.begin() + 0 + 1008, v115.begin());
  std::vector<float> v116(16);
  std::copy(v114.begin() + 1008, v114.begin() + 1008 + 16, v116.begin());
  std::copy(v115.begin(), v115.end(), v86.begin() + 16);
  std::copy(v116.begin(), v116.end(), v86.begin() + 0);
  std::vector<double> v119(std::begin(v86), std::end(v86));
  auto pt21_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt21_filled = v119;
  pt21_filled.clear();
  pt21_filled.reserve(pt21_filled_n);
  for (auto i = 0; i < pt21_filled_n; ++i) {
    pt21_filled.push_back(v119[i % v119.size()]);
  }
  auto pt21 = cc->MakeCKKSPackedPlaintext(pt21_filled);
  const auto& ct52 = cc->EvalMult(ct10, pt21);
  std::vector<float> v120(std::begin(v28) + 22 * 512, std::begin(v28) + 22 * 512 + 1024);
  std::vector<float> v121(1008);
  std::copy(v120.begin() + 0, v120.begin() + 0 + 1008, v121.begin());
  std::vector<float> v122(16);
  std::copy(v120.begin() + 1008, v120.begin() + 1008 + 16, v122.begin());
  std::copy(v121.begin(), v121.end(), v86.begin() + 16);
  std::copy(v122.begin(), v122.end(), v86.begin() + 0);
  std::vector<double> v125(std::begin(v86), std::end(v86));
  auto pt22_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt22_filled = v125;
  pt22_filled.clear();
  pt22_filled.reserve(pt22_filled_n);
  for (auto i = 0; i < pt22_filled_n; ++i) {
    pt22_filled.push_back(v125[i % v125.size()]);
  }
  auto pt22 = cc->MakeCKKSPackedPlaintext(pt22_filled);
  const auto& ct53 = cc->EvalMult(ct12, pt22);
  std::vector<float> v126(std::begin(v28) + 23 * 512, std::begin(v28) + 23 * 512 + 1024);
  std::vector<float> v127(1008);
  std::copy(v126.begin() + 0, v126.begin() + 0 + 1008, v127.begin());
  std::vector<float> v128(16);
  std::copy(v126.begin() + 1008, v126.begin() + 1008 + 16, v128.begin());
  std::copy(v127.begin(), v127.end(), v86.begin() + 16);
  std::copy(v128.begin(), v128.end(), v86.begin() + 0);
  std::vector<double> v131(std::begin(v86), std::end(v86));
  auto pt23_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt23_filled = v131;
  pt23_filled.clear();
  pt23_filled.reserve(pt23_filled_n);
  for (auto i = 0; i < pt23_filled_n; ++i) {
    pt23_filled.push_back(v131[i % v131.size()]);
  }
  auto pt23 = cc->MakeCKKSPackedPlaintext(pt23_filled);
  const auto& ct54 = cc->EvalMult(ct14, pt23);
  std::vector<float> v132(std::begin(v28) + 24 * 512, std::begin(v28) + 24 * 512 + 1024);
  std::vector<float> v133(1008);
  std::copy(v132.begin() + 0, v132.begin() + 0 + 1008, v133.begin());
  std::vector<float> v134(16);
  std::copy(v132.begin() + 1008, v132.begin() + 1008 + 16, v134.begin());
  std::copy(v133.begin(), v133.end(), v86.begin() + 16);
  std::copy(v134.begin(), v134.end(), v86.begin() + 0);
  std::vector<double> v137(std::begin(v86), std::end(v86));
  auto pt24_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt24_filled = v137;
  pt24_filled.clear();
  pt24_filled.reserve(pt24_filled_n);
  for (auto i = 0; i < pt24_filled_n; ++i) {
    pt24_filled.push_back(v137[i % v137.size()]);
  }
  auto pt24 = cc->MakeCKKSPackedPlaintext(pt24_filled);
  const auto& ct55 = cc->EvalMult(ct16, pt24);
  std::vector<float> v138(std::begin(v28) + 25 * 512, std::begin(v28) + 25 * 512 + 1024);
  std::vector<float> v139(1008);
  std::copy(v138.begin() + 0, v138.begin() + 0 + 1008, v139.begin());
  std::vector<float> v140(16);
  std::copy(v138.begin() + 1008, v138.begin() + 1008 + 16, v140.begin());
  std::copy(v139.begin(), v139.end(), v86.begin() + 16);
  std::copy(v140.begin(), v140.end(), v86.begin() + 0);
  std::vector<double> v143(std::begin(v86), std::end(v86));
  auto pt25_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt25_filled = v143;
  pt25_filled.clear();
  pt25_filled.reserve(pt25_filled_n);
  for (auto i = 0; i < pt25_filled_n; ++i) {
    pt25_filled.push_back(v143[i % v143.size()]);
  }
  auto pt25 = cc->MakeCKKSPackedPlaintext(pt25_filled);
  const auto& ct56 = cc->EvalMult(ct18, pt25);
  std::vector<float> v144(std::begin(v28) + 26 * 512, std::begin(v28) + 26 * 512 + 1024);
  std::vector<float> v145(1008);
  std::copy(v144.begin() + 0, v144.begin() + 0 + 1008, v145.begin());
  std::vector<float> v146(16);
  std::copy(v144.begin() + 1008, v144.begin() + 1008 + 16, v146.begin());
  std::copy(v145.begin(), v145.end(), v86.begin() + 16);
  std::copy(v146.begin(), v146.end(), v86.begin() + 0);
  std::vector<double> v149(std::begin(v86), std::end(v86));
  auto pt26_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt26_filled = v149;
  pt26_filled.clear();
  pt26_filled.reserve(pt26_filled_n);
  for (auto i = 0; i < pt26_filled_n; ++i) {
    pt26_filled.push_back(v149[i % v149.size()]);
  }
  auto pt26 = cc->MakeCKKSPackedPlaintext(pt26_filled);
  const auto& ct57 = cc->EvalMult(ct20, pt26);
  std::vector<float> v150(std::begin(v28) + 27 * 512, std::begin(v28) + 27 * 512 + 1024);
  std::vector<float> v151(1008);
  std::copy(v150.begin() + 0, v150.begin() + 0 + 1008, v151.begin());
  std::vector<float> v152(16);
  std::copy(v150.begin() + 1008, v150.begin() + 1008 + 16, v152.begin());
  std::copy(v151.begin(), v151.end(), v86.begin() + 16);
  std::copy(v152.begin(), v152.end(), v86.begin() + 0);
  std::vector<double> v155(std::begin(v86), std::end(v86));
  auto pt27_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt27_filled = v155;
  pt27_filled.clear();
  pt27_filled.reserve(pt27_filled_n);
  for (auto i = 0; i < pt27_filled_n; ++i) {
    pt27_filled.push_back(v155[i % v155.size()]);
  }
  auto pt27 = cc->MakeCKKSPackedPlaintext(pt27_filled);
  const auto& ct58 = cc->EvalMult(ct22, pt27);
  std::vector<float> v156(std::begin(v28) + 28 * 512, std::begin(v28) + 28 * 512 + 1024);
  std::vector<float> v157(1008);
  std::copy(v156.begin() + 0, v156.begin() + 0 + 1008, v157.begin());
  std::vector<float> v158(16);
  std::copy(v156.begin() + 1008, v156.begin() + 1008 + 16, v158.begin());
  std::copy(v157.begin(), v157.end(), v86.begin() + 16);
  std::copy(v158.begin(), v158.end(), v86.begin() + 0);
  std::vector<double> v161(std::begin(v86), std::end(v86));
  auto pt28_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt28_filled = v161;
  pt28_filled.clear();
  pt28_filled.reserve(pt28_filled_n);
  for (auto i = 0; i < pt28_filled_n; ++i) {
    pt28_filled.push_back(v161[i % v161.size()]);
  }
  auto pt28 = cc->MakeCKKSPackedPlaintext(pt28_filled);
  const auto& ct59 = cc->EvalMult(ct24, pt28);
  std::vector<float> v162(std::begin(v28) + 29 * 512, std::begin(v28) + 29 * 512 + 1024);
  std::vector<float> v163(1008);
  std::copy(v162.begin() + 0, v162.begin() + 0 + 1008, v163.begin());
  std::vector<float> v164(16);
  std::copy(v162.begin() + 1008, v162.begin() + 1008 + 16, v164.begin());
  std::copy(v163.begin(), v163.end(), v86.begin() + 16);
  std::copy(v164.begin(), v164.end(), v86.begin() + 0);
  std::vector<double> v167(std::begin(v86), std::end(v86));
  auto pt29_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt29_filled = v167;
  pt29_filled.clear();
  pt29_filled.reserve(pt29_filled_n);
  for (auto i = 0; i < pt29_filled_n; ++i) {
    pt29_filled.push_back(v167[i % v167.size()]);
  }
  auto pt29 = cc->MakeCKKSPackedPlaintext(pt29_filled);
  const auto& ct60 = cc->EvalMult(ct26, pt29);
  std::vector<float> v168(std::begin(v28) + 30 * 512, std::begin(v28) + 30 * 512 + 1024);
  std::vector<float> v169(1008);
  std::copy(v168.begin() + 0, v168.begin() + 0 + 1008, v169.begin());
  std::vector<float> v170(16);
  std::copy(v168.begin() + 1008, v168.begin() + 1008 + 16, v170.begin());
  std::copy(v169.begin(), v169.end(), v86.begin() + 16);
  std::copy(v170.begin(), v170.end(), v86.begin() + 0);
  std::vector<double> v173(std::begin(v86), std::end(v86));
  auto pt30_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt30_filled = v173;
  pt30_filled.clear();
  pt30_filled.reserve(pt30_filled_n);
  for (auto i = 0; i < pt30_filled_n; ++i) {
    pt30_filled.push_back(v173[i % v173.size()]);
  }
  auto pt30 = cc->MakeCKKSPackedPlaintext(pt30_filled);
  const auto& ct61 = cc->EvalMult(ct28, pt30);
  std::vector<float> v174(std::begin(v28) + 31 * 512, std::begin(v28) + 31 * 512 + 1024);
  std::vector<float> v175(1008);
  std::copy(v174.begin() + 0, v174.begin() + 0 + 1008, v175.begin());
  std::vector<float> v176(16);
  std::copy(v174.begin() + 1008, v174.begin() + 1008 + 16, v176.begin());
  std::copy(v175.begin(), v175.end(), v86.begin() + 16);
  std::copy(v176.begin(), v176.end(), v86.begin() + 0);
  std::vector<double> v179(std::begin(v86), std::end(v86));
  auto pt31_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt31_filled = v179;
  pt31_filled.clear();
  pt31_filled.reserve(pt31_filled_n);
  for (auto i = 0; i < pt31_filled_n; ++i) {
    pt31_filled.push_back(v179[i % v179.size()]);
  }
  auto pt31 = cc->MakeCKKSPackedPlaintext(pt31_filled);
  const auto& ct62 = cc->EvalMult(ct30, pt31);
  const auto& ct63 = cc->EvalAdd(ct47, ct48);
  const auto& ct64 = cc->EvalAdd(ct49, ct50);
  const auto& ct65 = cc->EvalAdd(ct63, ct64);
  const auto& ct66 = cc->EvalAdd(ct51, ct52);
  const auto& ct67 = cc->EvalAdd(ct53, ct54);
  const auto& ct68 = cc->EvalAdd(ct66, ct67);
  const auto& ct69 = cc->EvalAdd(ct65, ct68);
  const auto& ct70 = cc->EvalAdd(ct55, ct56);
  const auto& ct71 = cc->EvalAdd(ct57, ct58);
  const auto& ct72 = cc->EvalAdd(ct70, ct71);
  const auto& ct73 = cc->EvalAdd(ct59, ct60);
  const auto& ct74 = cc->EvalAdd(ct61, ct62);
  const auto& ct75 = cc->EvalAdd(ct73, ct74);
  const auto& ct76 = cc->EvalAdd(ct72, ct75);
  const auto& ct77 = cc->EvalAdd(ct69, ct76);
  const auto& ct78 = cc->EvalRotate(ct77, 16);
  std::vector<float> v180(std::begin(v28) + 32 * 512, std::begin(v28) + 32 * 512 + 1024);
  std::vector<float> v181(992);
  std::copy(v180.begin() + 0, v180.begin() + 0 + 992, v181.begin());
  std::vector<float> v182(32);
  std::copy(v180.begin() + 992, v180.begin() + 992 + 32, v182.begin());
  std::copy(v181.begin(), v181.end(), v86.begin() + 32);
  std::copy(v182.begin(), v182.end(), v86.begin() + 0);
  std::vector<double> v185(std::begin(v86), std::end(v86));
  auto pt32_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt32_filled = v185;
  pt32_filled.clear();
  pt32_filled.reserve(pt32_filled_n);
  for (auto i = 0; i < pt32_filled_n; ++i) {
    pt32_filled.push_back(v185[i % v185.size()]);
  }
  auto pt32 = cc->MakeCKKSPackedPlaintext(pt32_filled);
  const auto& ct79 = cc->EvalMult(ct, pt32);
  std::vector<float> v186(std::begin(v28) + 33 * 512, std::begin(v28) + 33 * 512 + 1024);
  std::vector<float> v187(992);
  std::copy(v186.begin() + 0, v186.begin() + 0 + 992, v187.begin());
  std::vector<float> v188(32);
  std::copy(v186.begin() + 992, v186.begin() + 992 + 32, v188.begin());
  std::copy(v187.begin(), v187.end(), v86.begin() + 32);
  std::copy(v188.begin(), v188.end(), v86.begin() + 0);
  std::vector<double> v191(std::begin(v86), std::end(v86));
  auto pt33_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt33_filled = v191;
  pt33_filled.clear();
  pt33_filled.reserve(pt33_filled_n);
  for (auto i = 0; i < pt33_filled_n; ++i) {
    pt33_filled.push_back(v191[i % v191.size()]);
  }
  auto pt33 = cc->MakeCKKSPackedPlaintext(pt33_filled);
  const auto& ct80 = cc->EvalMult(ct2, pt33);
  std::vector<float> v192(std::begin(v28) + 34 * 512, std::begin(v28) + 34 * 512 + 1024);
  std::vector<float> v193(992);
  std::copy(v192.begin() + 0, v192.begin() + 0 + 992, v193.begin());
  std::vector<float> v194(32);
  std::copy(v192.begin() + 992, v192.begin() + 992 + 32, v194.begin());
  std::copy(v193.begin(), v193.end(), v86.begin() + 32);
  std::copy(v194.begin(), v194.end(), v86.begin() + 0);
  std::vector<double> v197(std::begin(v86), std::end(v86));
  auto pt34_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt34_filled = v197;
  pt34_filled.clear();
  pt34_filled.reserve(pt34_filled_n);
  for (auto i = 0; i < pt34_filled_n; ++i) {
    pt34_filled.push_back(v197[i % v197.size()]);
  }
  auto pt34 = cc->MakeCKKSPackedPlaintext(pt34_filled);
  const auto& ct81 = cc->EvalMult(ct4, pt34);
  std::vector<float> v198(std::begin(v28) + 35 * 512, std::begin(v28) + 35 * 512 + 1024);
  std::vector<float> v199(992);
  std::copy(v198.begin() + 0, v198.begin() + 0 + 992, v199.begin());
  std::vector<float> v200(32);
  std::copy(v198.begin() + 992, v198.begin() + 992 + 32, v200.begin());
  std::copy(v199.begin(), v199.end(), v86.begin() + 32);
  std::copy(v200.begin(), v200.end(), v86.begin() + 0);
  std::vector<double> v203(std::begin(v86), std::end(v86));
  auto pt35_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt35_filled = v203;
  pt35_filled.clear();
  pt35_filled.reserve(pt35_filled_n);
  for (auto i = 0; i < pt35_filled_n; ++i) {
    pt35_filled.push_back(v203[i % v203.size()]);
  }
  auto pt35 = cc->MakeCKKSPackedPlaintext(pt35_filled);
  const auto& ct82 = cc->EvalMult(ct6, pt35);
  std::vector<float> v204(std::begin(v28) + 36 * 512, std::begin(v28) + 36 * 512 + 1024);
  std::vector<float> v205(992);
  std::copy(v204.begin() + 0, v204.begin() + 0 + 992, v205.begin());
  std::vector<float> v206(32);
  std::copy(v204.begin() + 992, v204.begin() + 992 + 32, v206.begin());
  std::copy(v205.begin(), v205.end(), v86.begin() + 32);
  std::copy(v206.begin(), v206.end(), v86.begin() + 0);
  std::vector<double> v209(std::begin(v86), std::end(v86));
  auto pt36_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt36_filled = v209;
  pt36_filled.clear();
  pt36_filled.reserve(pt36_filled_n);
  for (auto i = 0; i < pt36_filled_n; ++i) {
    pt36_filled.push_back(v209[i % v209.size()]);
  }
  auto pt36 = cc->MakeCKKSPackedPlaintext(pt36_filled);
  const auto& ct83 = cc->EvalMult(ct8, pt36);
  std::vector<float> v210(std::begin(v28) + 37 * 512, std::begin(v28) + 37 * 512 + 1024);
  std::vector<float> v211(992);
  std::copy(v210.begin() + 0, v210.begin() + 0 + 992, v211.begin());
  std::vector<float> v212(32);
  std::copy(v210.begin() + 992, v210.begin() + 992 + 32, v212.begin());
  std::copy(v211.begin(), v211.end(), v86.begin() + 32);
  std::copy(v212.begin(), v212.end(), v86.begin() + 0);
  std::vector<double> v215(std::begin(v86), std::end(v86));
  auto pt37_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt37_filled = v215;
  pt37_filled.clear();
  pt37_filled.reserve(pt37_filled_n);
  for (auto i = 0; i < pt37_filled_n; ++i) {
    pt37_filled.push_back(v215[i % v215.size()]);
  }
  auto pt37 = cc->MakeCKKSPackedPlaintext(pt37_filled);
  const auto& ct84 = cc->EvalMult(ct10, pt37);
  std::vector<float> v216(std::begin(v28) + 38 * 512, std::begin(v28) + 38 * 512 + 1024);
  std::vector<float> v217(992);
  std::copy(v216.begin() + 0, v216.begin() + 0 + 992, v217.begin());
  std::vector<float> v218(32);
  std::copy(v216.begin() + 992, v216.begin() + 992 + 32, v218.begin());
  std::copy(v217.begin(), v217.end(), v86.begin() + 32);
  std::copy(v218.begin(), v218.end(), v86.begin() + 0);
  std::vector<double> v221(std::begin(v86), std::end(v86));
  auto pt38_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt38_filled = v221;
  pt38_filled.clear();
  pt38_filled.reserve(pt38_filled_n);
  for (auto i = 0; i < pt38_filled_n; ++i) {
    pt38_filled.push_back(v221[i % v221.size()]);
  }
  auto pt38 = cc->MakeCKKSPackedPlaintext(pt38_filled);
  const auto& ct85 = cc->EvalMult(ct12, pt38);
  std::vector<float> v222(std::begin(v28) + 39 * 512, std::begin(v28) + 39 * 512 + 1024);
  std::vector<float> v223(992);
  std::copy(v222.begin() + 0, v222.begin() + 0 + 992, v223.begin());
  std::vector<float> v224(32);
  std::copy(v222.begin() + 992, v222.begin() + 992 + 32, v224.begin());
  std::copy(v223.begin(), v223.end(), v86.begin() + 32);
  std::copy(v224.begin(), v224.end(), v86.begin() + 0);
  std::vector<double> v227(std::begin(v86), std::end(v86));
  auto pt39_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt39_filled = v227;
  pt39_filled.clear();
  pt39_filled.reserve(pt39_filled_n);
  for (auto i = 0; i < pt39_filled_n; ++i) {
    pt39_filled.push_back(v227[i % v227.size()]);
  }
  auto pt39 = cc->MakeCKKSPackedPlaintext(pt39_filled);
  const auto& ct86 = cc->EvalMult(ct14, pt39);
  std::vector<float> v228(std::begin(v28) + 40 * 512, std::begin(v28) + 40 * 512 + 1024);
  std::vector<float> v229(992);
  std::copy(v228.begin() + 0, v228.begin() + 0 + 992, v229.begin());
  std::vector<float> v230(32);
  std::copy(v228.begin() + 992, v228.begin() + 992 + 32, v230.begin());
  std::copy(v229.begin(), v229.end(), v86.begin() + 32);
  std::copy(v230.begin(), v230.end(), v86.begin() + 0);
  std::vector<double> v233(std::begin(v86), std::end(v86));
  auto pt40_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt40_filled = v233;
  pt40_filled.clear();
  pt40_filled.reserve(pt40_filled_n);
  for (auto i = 0; i < pt40_filled_n; ++i) {
    pt40_filled.push_back(v233[i % v233.size()]);
  }
  auto pt40 = cc->MakeCKKSPackedPlaintext(pt40_filled);
  const auto& ct87 = cc->EvalMult(ct16, pt40);
  std::vector<float> v234(std::begin(v28) + 41 * 512, std::begin(v28) + 41 * 512 + 1024);
  std::vector<float> v235(992);
  std::copy(v234.begin() + 0, v234.begin() + 0 + 992, v235.begin());
  std::vector<float> v236(32);
  std::copy(v234.begin() + 992, v234.begin() + 992 + 32, v236.begin());
  std::copy(v235.begin(), v235.end(), v86.begin() + 32);
  std::copy(v236.begin(), v236.end(), v86.begin() + 0);
  std::vector<double> v239(std::begin(v86), std::end(v86));
  auto pt41_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt41_filled = v239;
  pt41_filled.clear();
  pt41_filled.reserve(pt41_filled_n);
  for (auto i = 0; i < pt41_filled_n; ++i) {
    pt41_filled.push_back(v239[i % v239.size()]);
  }
  auto pt41 = cc->MakeCKKSPackedPlaintext(pt41_filled);
  const auto& ct88 = cc->EvalMult(ct18, pt41);
  std::vector<float> v240(std::begin(v28) + 42 * 512, std::begin(v28) + 42 * 512 + 1024);
  std::vector<float> v241(992);
  std::copy(v240.begin() + 0, v240.begin() + 0 + 992, v241.begin());
  std::vector<float> v242(32);
  std::copy(v240.begin() + 992, v240.begin() + 992 + 32, v242.begin());
  std::copy(v241.begin(), v241.end(), v86.begin() + 32);
  std::copy(v242.begin(), v242.end(), v86.begin() + 0);
  std::vector<double> v245(std::begin(v86), std::end(v86));
  auto pt42_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt42_filled = v245;
  pt42_filled.clear();
  pt42_filled.reserve(pt42_filled_n);
  for (auto i = 0; i < pt42_filled_n; ++i) {
    pt42_filled.push_back(v245[i % v245.size()]);
  }
  auto pt42 = cc->MakeCKKSPackedPlaintext(pt42_filled);
  const auto& ct89 = cc->EvalMult(ct20, pt42);
  std::vector<float> v246(std::begin(v28) + 43 * 512, std::begin(v28) + 43 * 512 + 1024);
  std::vector<float> v247(992);
  std::copy(v246.begin() + 0, v246.begin() + 0 + 992, v247.begin());
  std::vector<float> v248(32);
  std::copy(v246.begin() + 992, v246.begin() + 992 + 32, v248.begin());
  std::copy(v247.begin(), v247.end(), v86.begin() + 32);
  std::copy(v248.begin(), v248.end(), v86.begin() + 0);
  std::vector<double> v251(std::begin(v86), std::end(v86));
  auto pt43_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt43_filled = v251;
  pt43_filled.clear();
  pt43_filled.reserve(pt43_filled_n);
  for (auto i = 0; i < pt43_filled_n; ++i) {
    pt43_filled.push_back(v251[i % v251.size()]);
  }
  auto pt43 = cc->MakeCKKSPackedPlaintext(pt43_filled);
  const auto& ct90 = cc->EvalMult(ct22, pt43);
  std::vector<float> v252(std::begin(v28) + 44 * 512, std::begin(v28) + 44 * 512 + 1024);
  std::vector<float> v253(992);
  std::copy(v252.begin() + 0, v252.begin() + 0 + 992, v253.begin());
  std::vector<float> v254(32);
  std::copy(v252.begin() + 992, v252.begin() + 992 + 32, v254.begin());
  std::copy(v253.begin(), v253.end(), v86.begin() + 32);
  std::copy(v254.begin(), v254.end(), v86.begin() + 0);
  std::vector<double> v257(std::begin(v86), std::end(v86));
  auto pt44_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt44_filled = v257;
  pt44_filled.clear();
  pt44_filled.reserve(pt44_filled_n);
  for (auto i = 0; i < pt44_filled_n; ++i) {
    pt44_filled.push_back(v257[i % v257.size()]);
  }
  auto pt44 = cc->MakeCKKSPackedPlaintext(pt44_filled);
  const auto& ct91 = cc->EvalMult(ct24, pt44);
  std::vector<float> v258(std::begin(v28) + 45 * 512, std::begin(v28) + 45 * 512 + 1024);
  std::vector<float> v259(992);
  std::copy(v258.begin() + 0, v258.begin() + 0 + 992, v259.begin());
  std::vector<float> v260(32);
  std::copy(v258.begin() + 992, v258.begin() + 992 + 32, v260.begin());
  std::copy(v259.begin(), v259.end(), v86.begin() + 32);
  std::copy(v260.begin(), v260.end(), v86.begin() + 0);
  std::vector<double> v263(std::begin(v86), std::end(v86));
  auto pt45_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt45_filled = v263;
  pt45_filled.clear();
  pt45_filled.reserve(pt45_filled_n);
  for (auto i = 0; i < pt45_filled_n; ++i) {
    pt45_filled.push_back(v263[i % v263.size()]);
  }
  auto pt45 = cc->MakeCKKSPackedPlaintext(pt45_filled);
  const auto& ct92 = cc->EvalMult(ct26, pt45);
  std::vector<float> v264(std::begin(v28) + 46 * 512, std::begin(v28) + 46 * 512 + 1024);
  std::vector<float> v265(992);
  std::copy(v264.begin() + 0, v264.begin() + 0 + 992, v265.begin());
  std::vector<float> v266(32);
  std::copy(v264.begin() + 992, v264.begin() + 992 + 32, v266.begin());
  std::copy(v265.begin(), v265.end(), v86.begin() + 32);
  std::copy(v266.begin(), v266.end(), v86.begin() + 0);
  std::vector<double> v269(std::begin(v86), std::end(v86));
  auto pt46_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt46_filled = v269;
  pt46_filled.clear();
  pt46_filled.reserve(pt46_filled_n);
  for (auto i = 0; i < pt46_filled_n; ++i) {
    pt46_filled.push_back(v269[i % v269.size()]);
  }
  auto pt46 = cc->MakeCKKSPackedPlaintext(pt46_filled);
  const auto& ct93 = cc->EvalMult(ct28, pt46);
  std::vector<float> v270(std::begin(v28) + 47 * 512, std::begin(v28) + 47 * 512 + 1024);
  std::vector<float> v271(992);
  std::copy(v270.begin() + 0, v270.begin() + 0 + 992, v271.begin());
  std::vector<float> v272(32);
  std::copy(v270.begin() + 992, v270.begin() + 992 + 32, v272.begin());
  std::copy(v271.begin(), v271.end(), v86.begin() + 32);
  std::copy(v272.begin(), v272.end(), v86.begin() + 0);
  std::vector<double> v275(std::begin(v86), std::end(v86));
  auto pt47_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt47_filled = v275;
  pt47_filled.clear();
  pt47_filled.reserve(pt47_filled_n);
  for (auto i = 0; i < pt47_filled_n; ++i) {
    pt47_filled.push_back(v275[i % v275.size()]);
  }
  auto pt47 = cc->MakeCKKSPackedPlaintext(pt47_filled);
  const auto& ct94 = cc->EvalMult(ct30, pt47);
  const auto& ct95 = cc->EvalAdd(ct79, ct80);
  const auto& ct96 = cc->EvalAdd(ct81, ct82);
  const auto& ct97 = cc->EvalAdd(ct95, ct96);
  const auto& ct98 = cc->EvalAdd(ct83, ct84);
  const auto& ct99 = cc->EvalAdd(ct85, ct86);
  const auto& ct100 = cc->EvalAdd(ct98, ct99);
  const auto& ct101 = cc->EvalAdd(ct97, ct100);
  const auto& ct102 = cc->EvalAdd(ct87, ct88);
  const auto& ct103 = cc->EvalAdd(ct89, ct90);
  const auto& ct104 = cc->EvalAdd(ct102, ct103);
  const auto& ct105 = cc->EvalAdd(ct91, ct92);
  const auto& ct106 = cc->EvalAdd(ct93, ct94);
  const auto& ct107 = cc->EvalAdd(ct105, ct106);
  const auto& ct108 = cc->EvalAdd(ct104, ct107);
  const auto& ct109 = cc->EvalAdd(ct101, ct108);
  const auto& ct110 = cc->EvalRotate(ct109, 32);
  std::vector<float> v276(std::begin(v28) + 48 * 512, std::begin(v28) + 48 * 512 + 1024);
  std::vector<float> v277(976);
  std::copy(v276.begin() + 0, v276.begin() + 0 + 976, v277.begin());
  std::vector<float> v278(48);
  std::copy(v276.begin() + 976, v276.begin() + 976 + 48, v278.begin());
  std::copy(v277.begin(), v277.end(), v86.begin() + 48);
  std::copy(v278.begin(), v278.end(), v86.begin() + 0);
  std::vector<double> v281(std::begin(v86), std::end(v86));
  auto pt48_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt48_filled = v281;
  pt48_filled.clear();
  pt48_filled.reserve(pt48_filled_n);
  for (auto i = 0; i < pt48_filled_n; ++i) {
    pt48_filled.push_back(v281[i % v281.size()]);
  }
  auto pt48 = cc->MakeCKKSPackedPlaintext(pt48_filled);
  const auto& ct111 = cc->EvalMult(ct, pt48);
  std::vector<float> v282(std::begin(v28) + 49 * 512, std::begin(v28) + 49 * 512 + 1024);
  std::vector<float> v283(976);
  std::copy(v282.begin() + 0, v282.begin() + 0 + 976, v283.begin());
  std::vector<float> v284(48);
  std::copy(v282.begin() + 976, v282.begin() + 976 + 48, v284.begin());
  std::copy(v283.begin(), v283.end(), v86.begin() + 48);
  std::copy(v284.begin(), v284.end(), v86.begin() + 0);
  std::vector<double> v287(std::begin(v86), std::end(v86));
  auto pt49_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt49_filled = v287;
  pt49_filled.clear();
  pt49_filled.reserve(pt49_filled_n);
  for (auto i = 0; i < pt49_filled_n; ++i) {
    pt49_filled.push_back(v287[i % v287.size()]);
  }
  auto pt49 = cc->MakeCKKSPackedPlaintext(pt49_filled);
  const auto& ct112 = cc->EvalMult(ct2, pt49);
  std::vector<float> v288(std::begin(v28) + 50 * 512, std::begin(v28) + 50 * 512 + 1024);
  std::vector<float> v289(976);
  std::copy(v288.begin() + 0, v288.begin() + 0 + 976, v289.begin());
  std::vector<float> v290(48);
  std::copy(v288.begin() + 976, v288.begin() + 976 + 48, v290.begin());
  std::copy(v289.begin(), v289.end(), v86.begin() + 48);
  std::copy(v290.begin(), v290.end(), v86.begin() + 0);
  std::vector<double> v293(std::begin(v86), std::end(v86));
  auto pt50_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt50_filled = v293;
  pt50_filled.clear();
  pt50_filled.reserve(pt50_filled_n);
  for (auto i = 0; i < pt50_filled_n; ++i) {
    pt50_filled.push_back(v293[i % v293.size()]);
  }
  auto pt50 = cc->MakeCKKSPackedPlaintext(pt50_filled);
  const auto& ct113 = cc->EvalMult(ct4, pt50);
  std::vector<float> v294(std::begin(v28) + 51 * 512, std::begin(v28) + 51 * 512 + 1024);
  std::vector<float> v295(976);
  std::copy(v294.begin() + 0, v294.begin() + 0 + 976, v295.begin());
  std::vector<float> v296(48);
  std::copy(v294.begin() + 976, v294.begin() + 976 + 48, v296.begin());
  std::copy(v295.begin(), v295.end(), v86.begin() + 48);
  std::copy(v296.begin(), v296.end(), v86.begin() + 0);
  std::vector<double> v299(std::begin(v86), std::end(v86));
  auto pt51_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt51_filled = v299;
  pt51_filled.clear();
  pt51_filled.reserve(pt51_filled_n);
  for (auto i = 0; i < pt51_filled_n; ++i) {
    pt51_filled.push_back(v299[i % v299.size()]);
  }
  auto pt51 = cc->MakeCKKSPackedPlaintext(pt51_filled);
  const auto& ct114 = cc->EvalMult(ct6, pt51);
  std::vector<float> v300(std::begin(v28) + 52 * 512, std::begin(v28) + 52 * 512 + 1024);
  std::vector<float> v301(976);
  std::copy(v300.begin() + 0, v300.begin() + 0 + 976, v301.begin());
  std::vector<float> v302(48);
  std::copy(v300.begin() + 976, v300.begin() + 976 + 48, v302.begin());
  std::copy(v301.begin(), v301.end(), v86.begin() + 48);
  std::copy(v302.begin(), v302.end(), v86.begin() + 0);
  std::vector<double> v305(std::begin(v86), std::end(v86));
  auto pt52_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt52_filled = v305;
  pt52_filled.clear();
  pt52_filled.reserve(pt52_filled_n);
  for (auto i = 0; i < pt52_filled_n; ++i) {
    pt52_filled.push_back(v305[i % v305.size()]);
  }
  auto pt52 = cc->MakeCKKSPackedPlaintext(pt52_filled);
  const auto& ct115 = cc->EvalMult(ct8, pt52);
  std::vector<float> v306(std::begin(v28) + 53 * 512, std::begin(v28) + 53 * 512 + 1024);
  std::vector<float> v307(976);
  std::copy(v306.begin() + 0, v306.begin() + 0 + 976, v307.begin());
  std::vector<float> v308(48);
  std::copy(v306.begin() + 976, v306.begin() + 976 + 48, v308.begin());
  std::copy(v307.begin(), v307.end(), v86.begin() + 48);
  std::copy(v308.begin(), v308.end(), v86.begin() + 0);
  std::vector<double> v311(std::begin(v86), std::end(v86));
  auto pt53_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt53_filled = v311;
  pt53_filled.clear();
  pt53_filled.reserve(pt53_filled_n);
  for (auto i = 0; i < pt53_filled_n; ++i) {
    pt53_filled.push_back(v311[i % v311.size()]);
  }
  auto pt53 = cc->MakeCKKSPackedPlaintext(pt53_filled);
  const auto& ct116 = cc->EvalMult(ct10, pt53);
  std::vector<float> v312(std::begin(v28) + 54 * 512, std::begin(v28) + 54 * 512 + 1024);
  std::vector<float> v313(976);
  std::copy(v312.begin() + 0, v312.begin() + 0 + 976, v313.begin());
  std::vector<float> v314(48);
  std::copy(v312.begin() + 976, v312.begin() + 976 + 48, v314.begin());
  std::copy(v313.begin(), v313.end(), v86.begin() + 48);
  std::copy(v314.begin(), v314.end(), v86.begin() + 0);
  std::vector<double> v317(std::begin(v86), std::end(v86));
  auto pt54_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt54_filled = v317;
  pt54_filled.clear();
  pt54_filled.reserve(pt54_filled_n);
  for (auto i = 0; i < pt54_filled_n; ++i) {
    pt54_filled.push_back(v317[i % v317.size()]);
  }
  auto pt54 = cc->MakeCKKSPackedPlaintext(pt54_filled);
  const auto& ct117 = cc->EvalMult(ct12, pt54);
  std::vector<float> v318(std::begin(v28) + 55 * 512, std::begin(v28) + 55 * 512 + 1024);
  std::vector<float> v319(976);
  std::copy(v318.begin() + 0, v318.begin() + 0 + 976, v319.begin());
  std::vector<float> v320(48);
  std::copy(v318.begin() + 976, v318.begin() + 976 + 48, v320.begin());
  std::copy(v319.begin(), v319.end(), v86.begin() + 48);
  std::copy(v320.begin(), v320.end(), v86.begin() + 0);
  std::vector<double> v323(std::begin(v86), std::end(v86));
  auto pt55_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt55_filled = v323;
  pt55_filled.clear();
  pt55_filled.reserve(pt55_filled_n);
  for (auto i = 0; i < pt55_filled_n; ++i) {
    pt55_filled.push_back(v323[i % v323.size()]);
  }
  auto pt55 = cc->MakeCKKSPackedPlaintext(pt55_filled);
  const auto& ct118 = cc->EvalMult(ct14, pt55);
  std::vector<float> v324(std::begin(v28) + 56 * 512, std::begin(v28) + 56 * 512 + 1024);
  std::vector<float> v325(976);
  std::copy(v324.begin() + 0, v324.begin() + 0 + 976, v325.begin());
  std::vector<float> v326(48);
  std::copy(v324.begin() + 976, v324.begin() + 976 + 48, v326.begin());
  std::copy(v325.begin(), v325.end(), v86.begin() + 48);
  std::copy(v326.begin(), v326.end(), v86.begin() + 0);
  std::vector<double> v329(std::begin(v86), std::end(v86));
  auto pt56_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt56_filled = v329;
  pt56_filled.clear();
  pt56_filled.reserve(pt56_filled_n);
  for (auto i = 0; i < pt56_filled_n; ++i) {
    pt56_filled.push_back(v329[i % v329.size()]);
  }
  auto pt56 = cc->MakeCKKSPackedPlaintext(pt56_filled);
  const auto& ct119 = cc->EvalMult(ct16, pt56);
  std::vector<float> v330(std::begin(v28) + 57 * 512, std::begin(v28) + 57 * 512 + 1024);
  std::vector<float> v331(976);
  std::copy(v330.begin() + 0, v330.begin() + 0 + 976, v331.begin());
  std::vector<float> v332(48);
  std::copy(v330.begin() + 976, v330.begin() + 976 + 48, v332.begin());
  std::copy(v331.begin(), v331.end(), v86.begin() + 48);
  std::copy(v332.begin(), v332.end(), v86.begin() + 0);
  std::vector<double> v335(std::begin(v86), std::end(v86));
  auto pt57_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt57_filled = v335;
  pt57_filled.clear();
  pt57_filled.reserve(pt57_filled_n);
  for (auto i = 0; i < pt57_filled_n; ++i) {
    pt57_filled.push_back(v335[i % v335.size()]);
  }
  auto pt57 = cc->MakeCKKSPackedPlaintext(pt57_filled);
  const auto& ct120 = cc->EvalMult(ct18, pt57);
  std::vector<float> v336(std::begin(v28) + 58 * 512, std::begin(v28) + 58 * 512 + 1024);
  std::vector<float> v337(976);
  std::copy(v336.begin() + 0, v336.begin() + 0 + 976, v337.begin());
  std::vector<float> v338(48);
  std::copy(v336.begin() + 976, v336.begin() + 976 + 48, v338.begin());
  std::copy(v337.begin(), v337.end(), v86.begin() + 48);
  std::copy(v338.begin(), v338.end(), v86.begin() + 0);
  std::vector<double> v341(std::begin(v86), std::end(v86));
  auto pt58_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt58_filled = v341;
  pt58_filled.clear();
  pt58_filled.reserve(pt58_filled_n);
  for (auto i = 0; i < pt58_filled_n; ++i) {
    pt58_filled.push_back(v341[i % v341.size()]);
  }
  auto pt58 = cc->MakeCKKSPackedPlaintext(pt58_filled);
  const auto& ct121 = cc->EvalMult(ct20, pt58);
  std::vector<float> v342(std::begin(v28) + 59 * 512, std::begin(v28) + 59 * 512 + 1024);
  std::vector<float> v343(976);
  std::copy(v342.begin() + 0, v342.begin() + 0 + 976, v343.begin());
  std::vector<float> v344(48);
  std::copy(v342.begin() + 976, v342.begin() + 976 + 48, v344.begin());
  std::copy(v343.begin(), v343.end(), v86.begin() + 48);
  std::copy(v344.begin(), v344.end(), v86.begin() + 0);
  std::vector<double> v347(std::begin(v86), std::end(v86));
  auto pt59_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt59_filled = v347;
  pt59_filled.clear();
  pt59_filled.reserve(pt59_filled_n);
  for (auto i = 0; i < pt59_filled_n; ++i) {
    pt59_filled.push_back(v347[i % v347.size()]);
  }
  auto pt59 = cc->MakeCKKSPackedPlaintext(pt59_filled);
  const auto& ct122 = cc->EvalMult(ct22, pt59);
  std::vector<float> v348(std::begin(v28) + 60 * 512, std::begin(v28) + 60 * 512 + 1024);
  std::vector<float> v349(976);
  std::copy(v348.begin() + 0, v348.begin() + 0 + 976, v349.begin());
  std::vector<float> v350(48);
  std::copy(v348.begin() + 976, v348.begin() + 976 + 48, v350.begin());
  std::copy(v349.begin(), v349.end(), v86.begin() + 48);
  std::copy(v350.begin(), v350.end(), v86.begin() + 0);
  std::vector<double> v353(std::begin(v86), std::end(v86));
  auto pt60_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt60_filled = v353;
  pt60_filled.clear();
  pt60_filled.reserve(pt60_filled_n);
  for (auto i = 0; i < pt60_filled_n; ++i) {
    pt60_filled.push_back(v353[i % v353.size()]);
  }
  auto pt60 = cc->MakeCKKSPackedPlaintext(pt60_filled);
  const auto& ct123 = cc->EvalMult(ct24, pt60);
  std::vector<float> v354(std::begin(v28) + 61 * 512, std::begin(v28) + 61 * 512 + 1024);
  std::vector<float> v355(976);
  std::copy(v354.begin() + 0, v354.begin() + 0 + 976, v355.begin());
  std::vector<float> v356(48);
  std::copy(v354.begin() + 976, v354.begin() + 976 + 48, v356.begin());
  std::copy(v355.begin(), v355.end(), v86.begin() + 48);
  std::copy(v356.begin(), v356.end(), v86.begin() + 0);
  std::vector<double> v359(std::begin(v86), std::end(v86));
  auto pt61_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt61_filled = v359;
  pt61_filled.clear();
  pt61_filled.reserve(pt61_filled_n);
  for (auto i = 0; i < pt61_filled_n; ++i) {
    pt61_filled.push_back(v359[i % v359.size()]);
  }
  auto pt61 = cc->MakeCKKSPackedPlaintext(pt61_filled);
  const auto& ct124 = cc->EvalMult(ct26, pt61);
  std::vector<float> v360(std::begin(v28) + 62 * 512, std::begin(v28) + 62 * 512 + 1024);
  std::vector<float> v361(976);
  std::copy(v360.begin() + 0, v360.begin() + 0 + 976, v361.begin());
  std::vector<float> v362(48);
  std::copy(v360.begin() + 976, v360.begin() + 976 + 48, v362.begin());
  std::copy(v361.begin(), v361.end(), v86.begin() + 48);
  std::copy(v362.begin(), v362.end(), v86.begin() + 0);
  std::vector<double> v365(std::begin(v86), std::end(v86));
  auto pt62_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt62_filled = v365;
  pt62_filled.clear();
  pt62_filled.reserve(pt62_filled_n);
  for (auto i = 0; i < pt62_filled_n; ++i) {
    pt62_filled.push_back(v365[i % v365.size()]);
  }
  auto pt62 = cc->MakeCKKSPackedPlaintext(pt62_filled);
  const auto& ct125 = cc->EvalMult(ct28, pt62);
  std::vector<float> v366(std::begin(v28) + 63 * 512, std::begin(v28) + 63 * 512 + 1024);
  std::vector<float> v367(976);
  std::copy(v366.begin() + 0, v366.begin() + 0 + 976, v367.begin());
  std::vector<float> v368(48);
  std::copy(v366.begin() + 976, v366.begin() + 976 + 48, v368.begin());
  std::copy(v367.begin(), v367.end(), v86.begin() + 48);
  std::copy(v368.begin(), v368.end(), v86.begin() + 0);
  std::vector<double> v371(std::begin(v86), std::end(v86));
  auto pt63_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt63_filled = v371;
  pt63_filled.clear();
  pt63_filled.reserve(pt63_filled_n);
  for (auto i = 0; i < pt63_filled_n; ++i) {
    pt63_filled.push_back(v371[i % v371.size()]);
  }
  auto pt63 = cc->MakeCKKSPackedPlaintext(pt63_filled);
  const auto& ct126 = cc->EvalMult(ct30, pt63);
  const auto& ct127 = cc->EvalAdd(ct111, ct112);
  const auto& ct128 = cc->EvalAdd(ct113, ct114);
  const auto& ct129 = cc->EvalAdd(ct127, ct128);
  const auto& ct130 = cc->EvalAdd(ct115, ct116);
  const auto& ct131 = cc->EvalAdd(ct117, ct118);
  const auto& ct132 = cc->EvalAdd(ct130, ct131);
  const auto& ct133 = cc->EvalAdd(ct129, ct132);
  const auto& ct134 = cc->EvalAdd(ct119, ct120);
  const auto& ct135 = cc->EvalAdd(ct121, ct122);
  const auto& ct136 = cc->EvalAdd(ct134, ct135);
  const auto& ct137 = cc->EvalAdd(ct123, ct124);
  const auto& ct138 = cc->EvalAdd(ct125, ct126);
  const auto& ct139 = cc->EvalAdd(ct137, ct138);
  const auto& ct140 = cc->EvalAdd(ct136, ct139);
  const auto& ct141 = cc->EvalAdd(ct133, ct140);
  const auto& ct142 = cc->EvalRotate(ct141, 48);
  std::vector<float> v372(std::begin(v28) + 64 * 512, std::begin(v28) + 64 * 512 + 1024);
  std::vector<float> v373(960);
  std::copy(v372.begin() + 0, v372.begin() + 0 + 960, v373.begin());
  std::vector<float> v374(64);
  std::copy(v372.begin() + 960, v372.begin() + 960 + 64, v374.begin());
  std::copy(v373.begin(), v373.end(), v86.begin() + 64);
  std::copy(v374.begin(), v374.end(), v86.begin() + 0);
  std::vector<double> v377(std::begin(v86), std::end(v86));
  auto pt64_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt64_filled = v377;
  pt64_filled.clear();
  pt64_filled.reserve(pt64_filled_n);
  for (auto i = 0; i < pt64_filled_n; ++i) {
    pt64_filled.push_back(v377[i % v377.size()]);
  }
  auto pt64 = cc->MakeCKKSPackedPlaintext(pt64_filled);
  const auto& ct143 = cc->EvalMult(ct, pt64);
  std::vector<float> v378(std::begin(v28) + 65 * 512, std::begin(v28) + 65 * 512 + 1024);
  std::vector<float> v379(960);
  std::copy(v378.begin() + 0, v378.begin() + 0 + 960, v379.begin());
  std::vector<float> v380(64);
  std::copy(v378.begin() + 960, v378.begin() + 960 + 64, v380.begin());
  std::copy(v379.begin(), v379.end(), v86.begin() + 64);
  std::copy(v380.begin(), v380.end(), v86.begin() + 0);
  std::vector<double> v383(std::begin(v86), std::end(v86));
  auto pt65_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt65_filled = v383;
  pt65_filled.clear();
  pt65_filled.reserve(pt65_filled_n);
  for (auto i = 0; i < pt65_filled_n; ++i) {
    pt65_filled.push_back(v383[i % v383.size()]);
  }
  auto pt65 = cc->MakeCKKSPackedPlaintext(pt65_filled);
  const auto& ct144 = cc->EvalMult(ct2, pt65);
  std::vector<float> v384(std::begin(v28) + 66 * 512, std::begin(v28) + 66 * 512 + 1024);
  std::vector<float> v385(960);
  std::copy(v384.begin() + 0, v384.begin() + 0 + 960, v385.begin());
  std::vector<float> v386(64);
  std::copy(v384.begin() + 960, v384.begin() + 960 + 64, v386.begin());
  std::copy(v385.begin(), v385.end(), v86.begin() + 64);
  std::copy(v386.begin(), v386.end(), v86.begin() + 0);
  std::vector<double> v389(std::begin(v86), std::end(v86));
  auto pt66_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt66_filled = v389;
  pt66_filled.clear();
  pt66_filled.reserve(pt66_filled_n);
  for (auto i = 0; i < pt66_filled_n; ++i) {
    pt66_filled.push_back(v389[i % v389.size()]);
  }
  auto pt66 = cc->MakeCKKSPackedPlaintext(pt66_filled);
  const auto& ct145 = cc->EvalMult(ct4, pt66);
  std::vector<float> v390(std::begin(v28) + 67 * 512, std::begin(v28) + 67 * 512 + 1024);
  std::vector<float> v391(960);
  std::copy(v390.begin() + 0, v390.begin() + 0 + 960, v391.begin());
  std::vector<float> v392(64);
  std::copy(v390.begin() + 960, v390.begin() + 960 + 64, v392.begin());
  std::copy(v391.begin(), v391.end(), v86.begin() + 64);
  std::copy(v392.begin(), v392.end(), v86.begin() + 0);
  std::vector<double> v395(std::begin(v86), std::end(v86));
  auto pt67_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt67_filled = v395;
  pt67_filled.clear();
  pt67_filled.reserve(pt67_filled_n);
  for (auto i = 0; i < pt67_filled_n; ++i) {
    pt67_filled.push_back(v395[i % v395.size()]);
  }
  auto pt67 = cc->MakeCKKSPackedPlaintext(pt67_filled);
  const auto& ct146 = cc->EvalMult(ct6, pt67);
  std::vector<float> v396(std::begin(v28) + 68 * 512, std::begin(v28) + 68 * 512 + 1024);
  std::vector<float> v397(960);
  std::copy(v396.begin() + 0, v396.begin() + 0 + 960, v397.begin());
  std::vector<float> v398(64);
  std::copy(v396.begin() + 960, v396.begin() + 960 + 64, v398.begin());
  std::copy(v397.begin(), v397.end(), v86.begin() + 64);
  std::copy(v398.begin(), v398.end(), v86.begin() + 0);
  std::vector<double> v401(std::begin(v86), std::end(v86));
  auto pt68_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt68_filled = v401;
  pt68_filled.clear();
  pt68_filled.reserve(pt68_filled_n);
  for (auto i = 0; i < pt68_filled_n; ++i) {
    pt68_filled.push_back(v401[i % v401.size()]);
  }
  auto pt68 = cc->MakeCKKSPackedPlaintext(pt68_filled);
  const auto& ct147 = cc->EvalMult(ct8, pt68);
  std::vector<float> v402(std::begin(v28) + 69 * 512, std::begin(v28) + 69 * 512 + 1024);
  std::vector<float> v403(960);
  std::copy(v402.begin() + 0, v402.begin() + 0 + 960, v403.begin());
  std::vector<float> v404(64);
  std::copy(v402.begin() + 960, v402.begin() + 960 + 64, v404.begin());
  std::copy(v403.begin(), v403.end(), v86.begin() + 64);
  std::copy(v404.begin(), v404.end(), v86.begin() + 0);
  std::vector<double> v407(std::begin(v86), std::end(v86));
  auto pt69_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt69_filled = v407;
  pt69_filled.clear();
  pt69_filled.reserve(pt69_filled_n);
  for (auto i = 0; i < pt69_filled_n; ++i) {
    pt69_filled.push_back(v407[i % v407.size()]);
  }
  auto pt69 = cc->MakeCKKSPackedPlaintext(pt69_filled);
  const auto& ct148 = cc->EvalMult(ct10, pt69);
  std::vector<float> v408(std::begin(v28) + 70 * 512, std::begin(v28) + 70 * 512 + 1024);
  std::vector<float> v409(960);
  std::copy(v408.begin() + 0, v408.begin() + 0 + 960, v409.begin());
  std::vector<float> v410(64);
  std::copy(v408.begin() + 960, v408.begin() + 960 + 64, v410.begin());
  std::copy(v409.begin(), v409.end(), v86.begin() + 64);
  std::copy(v410.begin(), v410.end(), v86.begin() + 0);
  std::vector<double> v413(std::begin(v86), std::end(v86));
  auto pt70_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt70_filled = v413;
  pt70_filled.clear();
  pt70_filled.reserve(pt70_filled_n);
  for (auto i = 0; i < pt70_filled_n; ++i) {
    pt70_filled.push_back(v413[i % v413.size()]);
  }
  auto pt70 = cc->MakeCKKSPackedPlaintext(pt70_filled);
  const auto& ct149 = cc->EvalMult(ct12, pt70);
  std::vector<float> v414(std::begin(v28) + 71 * 512, std::begin(v28) + 71 * 512 + 1024);
  std::vector<float> v415(960);
  std::copy(v414.begin() + 0, v414.begin() + 0 + 960, v415.begin());
  std::vector<float> v416(64);
  std::copy(v414.begin() + 960, v414.begin() + 960 + 64, v416.begin());
  std::copy(v415.begin(), v415.end(), v86.begin() + 64);
  std::copy(v416.begin(), v416.end(), v86.begin() + 0);
  std::vector<double> v419(std::begin(v86), std::end(v86));
  auto pt71_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt71_filled = v419;
  pt71_filled.clear();
  pt71_filled.reserve(pt71_filled_n);
  for (auto i = 0; i < pt71_filled_n; ++i) {
    pt71_filled.push_back(v419[i % v419.size()]);
  }
  auto pt71 = cc->MakeCKKSPackedPlaintext(pt71_filled);
  const auto& ct150 = cc->EvalMult(ct14, pt71);
  std::vector<float> v420(std::begin(v28) + 72 * 512, std::begin(v28) + 72 * 512 + 1024);
  std::vector<float> v421(960);
  std::copy(v420.begin() + 0, v420.begin() + 0 + 960, v421.begin());
  std::vector<float> v422(64);
  std::copy(v420.begin() + 960, v420.begin() + 960 + 64, v422.begin());
  std::copy(v421.begin(), v421.end(), v86.begin() + 64);
  std::copy(v422.begin(), v422.end(), v86.begin() + 0);
  std::vector<double> v425(std::begin(v86), std::end(v86));
  auto pt72_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt72_filled = v425;
  pt72_filled.clear();
  pt72_filled.reserve(pt72_filled_n);
  for (auto i = 0; i < pt72_filled_n; ++i) {
    pt72_filled.push_back(v425[i % v425.size()]);
  }
  auto pt72 = cc->MakeCKKSPackedPlaintext(pt72_filled);
  const auto& ct151 = cc->EvalMult(ct16, pt72);
  std::vector<float> v426(std::begin(v28) + 73 * 512, std::begin(v28) + 73 * 512 + 1024);
  std::vector<float> v427(960);
  std::copy(v426.begin() + 0, v426.begin() + 0 + 960, v427.begin());
  std::vector<float> v428(64);
  std::copy(v426.begin() + 960, v426.begin() + 960 + 64, v428.begin());
  std::copy(v427.begin(), v427.end(), v86.begin() + 64);
  std::copy(v428.begin(), v428.end(), v86.begin() + 0);
  std::vector<double> v431(std::begin(v86), std::end(v86));
  auto pt73_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt73_filled = v431;
  pt73_filled.clear();
  pt73_filled.reserve(pt73_filled_n);
  for (auto i = 0; i < pt73_filled_n; ++i) {
    pt73_filled.push_back(v431[i % v431.size()]);
  }
  auto pt73 = cc->MakeCKKSPackedPlaintext(pt73_filled);
  const auto& ct152 = cc->EvalMult(ct18, pt73);
  std::vector<float> v432(std::begin(v28) + 74 * 512, std::begin(v28) + 74 * 512 + 1024);
  std::vector<float> v433(960);
  std::copy(v432.begin() + 0, v432.begin() + 0 + 960, v433.begin());
  std::vector<float> v434(64);
  std::copy(v432.begin() + 960, v432.begin() + 960 + 64, v434.begin());
  std::copy(v433.begin(), v433.end(), v86.begin() + 64);
  std::copy(v434.begin(), v434.end(), v86.begin() + 0);
  std::vector<double> v437(std::begin(v86), std::end(v86));
  auto pt74_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt74_filled = v437;
  pt74_filled.clear();
  pt74_filled.reserve(pt74_filled_n);
  for (auto i = 0; i < pt74_filled_n; ++i) {
    pt74_filled.push_back(v437[i % v437.size()]);
  }
  auto pt74 = cc->MakeCKKSPackedPlaintext(pt74_filled);
  const auto& ct153 = cc->EvalMult(ct20, pt74);
  std::vector<float> v438(std::begin(v28) + 75 * 512, std::begin(v28) + 75 * 512 + 1024);
  std::vector<float> v439(960);
  std::copy(v438.begin() + 0, v438.begin() + 0 + 960, v439.begin());
  std::vector<float> v440(64);
  std::copy(v438.begin() + 960, v438.begin() + 960 + 64, v440.begin());
  std::copy(v439.begin(), v439.end(), v86.begin() + 64);
  std::copy(v440.begin(), v440.end(), v86.begin() + 0);
  std::vector<double> v443(std::begin(v86), std::end(v86));
  auto pt75_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt75_filled = v443;
  pt75_filled.clear();
  pt75_filled.reserve(pt75_filled_n);
  for (auto i = 0; i < pt75_filled_n; ++i) {
    pt75_filled.push_back(v443[i % v443.size()]);
  }
  auto pt75 = cc->MakeCKKSPackedPlaintext(pt75_filled);
  const auto& ct154 = cc->EvalMult(ct22, pt75);
  std::vector<float> v444(std::begin(v28) + 76 * 512, std::begin(v28) + 76 * 512 + 1024);
  std::vector<float> v445(960);
  std::copy(v444.begin() + 0, v444.begin() + 0 + 960, v445.begin());
  std::vector<float> v446(64);
  std::copy(v444.begin() + 960, v444.begin() + 960 + 64, v446.begin());
  std::copy(v445.begin(), v445.end(), v86.begin() + 64);
  std::copy(v446.begin(), v446.end(), v86.begin() + 0);
  std::vector<double> v449(std::begin(v86), std::end(v86));
  auto pt76_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt76_filled = v449;
  pt76_filled.clear();
  pt76_filled.reserve(pt76_filled_n);
  for (auto i = 0; i < pt76_filled_n; ++i) {
    pt76_filled.push_back(v449[i % v449.size()]);
  }
  auto pt76 = cc->MakeCKKSPackedPlaintext(pt76_filled);
  const auto& ct155 = cc->EvalMult(ct24, pt76);
  std::vector<float> v450(std::begin(v28) + 77 * 512, std::begin(v28) + 77 * 512 + 1024);
  std::vector<float> v451(960);
  std::copy(v450.begin() + 0, v450.begin() + 0 + 960, v451.begin());
  std::vector<float> v452(64);
  std::copy(v450.begin() + 960, v450.begin() + 960 + 64, v452.begin());
  std::copy(v451.begin(), v451.end(), v86.begin() + 64);
  std::copy(v452.begin(), v452.end(), v86.begin() + 0);
  std::vector<double> v455(std::begin(v86), std::end(v86));
  auto pt77_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt77_filled = v455;
  pt77_filled.clear();
  pt77_filled.reserve(pt77_filled_n);
  for (auto i = 0; i < pt77_filled_n; ++i) {
    pt77_filled.push_back(v455[i % v455.size()]);
  }
  auto pt77 = cc->MakeCKKSPackedPlaintext(pt77_filled);
  const auto& ct156 = cc->EvalMult(ct26, pt77);
  std::vector<float> v456(std::begin(v28) + 78 * 512, std::begin(v28) + 78 * 512 + 1024);
  std::vector<float> v457(960);
  std::copy(v456.begin() + 0, v456.begin() + 0 + 960, v457.begin());
  std::vector<float> v458(64);
  std::copy(v456.begin() + 960, v456.begin() + 960 + 64, v458.begin());
  std::copy(v457.begin(), v457.end(), v86.begin() + 64);
  std::copy(v458.begin(), v458.end(), v86.begin() + 0);
  std::vector<double> v461(std::begin(v86), std::end(v86));
  auto pt78_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt78_filled = v461;
  pt78_filled.clear();
  pt78_filled.reserve(pt78_filled_n);
  for (auto i = 0; i < pt78_filled_n; ++i) {
    pt78_filled.push_back(v461[i % v461.size()]);
  }
  auto pt78 = cc->MakeCKKSPackedPlaintext(pt78_filled);
  const auto& ct157 = cc->EvalMult(ct28, pt78);
  std::vector<float> v462(std::begin(v28) + 79 * 512, std::begin(v28) + 79 * 512 + 1024);
  std::vector<float> v463(960);
  std::copy(v462.begin() + 0, v462.begin() + 0 + 960, v463.begin());
  std::vector<float> v464(64);
  std::copy(v462.begin() + 960, v462.begin() + 960 + 64, v464.begin());
  std::copy(v463.begin(), v463.end(), v86.begin() + 64);
  std::copy(v464.begin(), v464.end(), v86.begin() + 0);
  std::vector<double> v467(std::begin(v86), std::end(v86));
  auto pt79_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt79_filled = v467;
  pt79_filled.clear();
  pt79_filled.reserve(pt79_filled_n);
  for (auto i = 0; i < pt79_filled_n; ++i) {
    pt79_filled.push_back(v467[i % v467.size()]);
  }
  auto pt79 = cc->MakeCKKSPackedPlaintext(pt79_filled);
  const auto& ct158 = cc->EvalMult(ct30, pt79);
  const auto& ct159 = cc->EvalAdd(ct143, ct144);
  const auto& ct160 = cc->EvalAdd(ct145, ct146);
  const auto& ct161 = cc->EvalAdd(ct159, ct160);
  const auto& ct162 = cc->EvalAdd(ct147, ct148);
  const auto& ct163 = cc->EvalAdd(ct149, ct150);
  const auto& ct164 = cc->EvalAdd(ct162, ct163);
  const auto& ct165 = cc->EvalAdd(ct161, ct164);
  const auto& ct166 = cc->EvalAdd(ct151, ct152);
  const auto& ct167 = cc->EvalAdd(ct153, ct154);
  const auto& ct168 = cc->EvalAdd(ct166, ct167);
  const auto& ct169 = cc->EvalAdd(ct155, ct156);
  const auto& ct170 = cc->EvalAdd(ct157, ct158);
  const auto& ct171 = cc->EvalAdd(ct169, ct170);
  const auto& ct172 = cc->EvalAdd(ct168, ct171);
  const auto& ct173 = cc->EvalAdd(ct165, ct172);
  const auto& ct174 = cc->EvalRotate(ct173, 64);
  std::vector<float> v468(std::begin(v28) + 80 * 512, std::begin(v28) + 80 * 512 + 1024);
  std::vector<float> v469(944);
  std::copy(v468.begin() + 0, v468.begin() + 0 + 944, v469.begin());
  std::vector<float> v470(80);
  std::copy(v468.begin() + 944, v468.begin() + 944 + 80, v470.begin());
  std::copy(v469.begin(), v469.end(), v86.begin() + 80);
  std::copy(v470.begin(), v470.end(), v86.begin() + 0);
  std::vector<double> v473(std::begin(v86), std::end(v86));
  auto pt80_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt80_filled = v473;
  pt80_filled.clear();
  pt80_filled.reserve(pt80_filled_n);
  for (auto i = 0; i < pt80_filled_n; ++i) {
    pt80_filled.push_back(v473[i % v473.size()]);
  }
  auto pt80 = cc->MakeCKKSPackedPlaintext(pt80_filled);
  const auto& ct175 = cc->EvalMult(ct, pt80);
  std::vector<float> v474(std::begin(v28) + 81 * 512, std::begin(v28) + 81 * 512 + 1024);
  std::vector<float> v475(944);
  std::copy(v474.begin() + 0, v474.begin() + 0 + 944, v475.begin());
  std::vector<float> v476(80);
  std::copy(v474.begin() + 944, v474.begin() + 944 + 80, v476.begin());
  std::copy(v475.begin(), v475.end(), v86.begin() + 80);
  std::copy(v476.begin(), v476.end(), v86.begin() + 0);
  std::vector<double> v479(std::begin(v86), std::end(v86));
  auto pt81_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt81_filled = v479;
  pt81_filled.clear();
  pt81_filled.reserve(pt81_filled_n);
  for (auto i = 0; i < pt81_filled_n; ++i) {
    pt81_filled.push_back(v479[i % v479.size()]);
  }
  auto pt81 = cc->MakeCKKSPackedPlaintext(pt81_filled);
  const auto& ct176 = cc->EvalMult(ct2, pt81);
  std::vector<float> v480(std::begin(v28) + 82 * 512, std::begin(v28) + 82 * 512 + 1024);
  std::vector<float> v481(944);
  std::copy(v480.begin() + 0, v480.begin() + 0 + 944, v481.begin());
  std::vector<float> v482(80);
  std::copy(v480.begin() + 944, v480.begin() + 944 + 80, v482.begin());
  std::copy(v481.begin(), v481.end(), v86.begin() + 80);
  std::copy(v482.begin(), v482.end(), v86.begin() + 0);
  std::vector<double> v485(std::begin(v86), std::end(v86));
  auto pt82_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt82_filled = v485;
  pt82_filled.clear();
  pt82_filled.reserve(pt82_filled_n);
  for (auto i = 0; i < pt82_filled_n; ++i) {
    pt82_filled.push_back(v485[i % v485.size()]);
  }
  auto pt82 = cc->MakeCKKSPackedPlaintext(pt82_filled);
  const auto& ct177 = cc->EvalMult(ct4, pt82);
  std::vector<float> v486(std::begin(v28) + 83 * 512, std::begin(v28) + 83 * 512 + 1024);
  std::vector<float> v487(944);
  std::copy(v486.begin() + 0, v486.begin() + 0 + 944, v487.begin());
  std::vector<float> v488(80);
  std::copy(v486.begin() + 944, v486.begin() + 944 + 80, v488.begin());
  std::copy(v487.begin(), v487.end(), v86.begin() + 80);
  std::copy(v488.begin(), v488.end(), v86.begin() + 0);
  std::vector<double> v491(std::begin(v86), std::end(v86));
  auto pt83_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt83_filled = v491;
  pt83_filled.clear();
  pt83_filled.reserve(pt83_filled_n);
  for (auto i = 0; i < pt83_filled_n; ++i) {
    pt83_filled.push_back(v491[i % v491.size()]);
  }
  auto pt83 = cc->MakeCKKSPackedPlaintext(pt83_filled);
  const auto& ct178 = cc->EvalMult(ct6, pt83);
  std::vector<float> v492(std::begin(v28) + 84 * 512, std::begin(v28) + 84 * 512 + 1024);
  std::vector<float> v493(944);
  std::copy(v492.begin() + 0, v492.begin() + 0 + 944, v493.begin());
  std::vector<float> v494(80);
  std::copy(v492.begin() + 944, v492.begin() + 944 + 80, v494.begin());
  std::copy(v493.begin(), v493.end(), v86.begin() + 80);
  std::copy(v494.begin(), v494.end(), v86.begin() + 0);
  std::vector<double> v497(std::begin(v86), std::end(v86));
  auto pt84_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt84_filled = v497;
  pt84_filled.clear();
  pt84_filled.reserve(pt84_filled_n);
  for (auto i = 0; i < pt84_filled_n; ++i) {
    pt84_filled.push_back(v497[i % v497.size()]);
  }
  auto pt84 = cc->MakeCKKSPackedPlaintext(pt84_filled);
  const auto& ct179 = cc->EvalMult(ct8, pt84);
  std::vector<float> v498(std::begin(v28) + 85 * 512, std::begin(v28) + 85 * 512 + 1024);
  std::vector<float> v499(944);
  std::copy(v498.begin() + 0, v498.begin() + 0 + 944, v499.begin());
  std::vector<float> v500(80);
  std::copy(v498.begin() + 944, v498.begin() + 944 + 80, v500.begin());
  std::copy(v499.begin(), v499.end(), v86.begin() + 80);
  std::copy(v500.begin(), v500.end(), v86.begin() + 0);
  std::vector<double> v503(std::begin(v86), std::end(v86));
  auto pt85_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt85_filled = v503;
  pt85_filled.clear();
  pt85_filled.reserve(pt85_filled_n);
  for (auto i = 0; i < pt85_filled_n; ++i) {
    pt85_filled.push_back(v503[i % v503.size()]);
  }
  auto pt85 = cc->MakeCKKSPackedPlaintext(pt85_filled);
  const auto& ct180 = cc->EvalMult(ct10, pt85);
  std::vector<float> v504(std::begin(v28) + 86 * 512, std::begin(v28) + 86 * 512 + 1024);
  std::vector<float> v505(944);
  std::copy(v504.begin() + 0, v504.begin() + 0 + 944, v505.begin());
  std::vector<float> v506(80);
  std::copy(v504.begin() + 944, v504.begin() + 944 + 80, v506.begin());
  std::copy(v505.begin(), v505.end(), v86.begin() + 80);
  std::copy(v506.begin(), v506.end(), v86.begin() + 0);
  std::vector<double> v509(std::begin(v86), std::end(v86));
  auto pt86_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt86_filled = v509;
  pt86_filled.clear();
  pt86_filled.reserve(pt86_filled_n);
  for (auto i = 0; i < pt86_filled_n; ++i) {
    pt86_filled.push_back(v509[i % v509.size()]);
  }
  auto pt86 = cc->MakeCKKSPackedPlaintext(pt86_filled);
  const auto& ct181 = cc->EvalMult(ct12, pt86);
  std::vector<float> v510(std::begin(v28) + 87 * 512, std::begin(v28) + 87 * 512 + 1024);
  std::vector<float> v511(944);
  std::copy(v510.begin() + 0, v510.begin() + 0 + 944, v511.begin());
  std::vector<float> v512(80);
  std::copy(v510.begin() + 944, v510.begin() + 944 + 80, v512.begin());
  std::copy(v511.begin(), v511.end(), v86.begin() + 80);
  std::copy(v512.begin(), v512.end(), v86.begin() + 0);
  std::vector<double> v515(std::begin(v86), std::end(v86));
  auto pt87_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt87_filled = v515;
  pt87_filled.clear();
  pt87_filled.reserve(pt87_filled_n);
  for (auto i = 0; i < pt87_filled_n; ++i) {
    pt87_filled.push_back(v515[i % v515.size()]);
  }
  auto pt87 = cc->MakeCKKSPackedPlaintext(pt87_filled);
  const auto& ct182 = cc->EvalMult(ct14, pt87);
  std::vector<float> v516(std::begin(v28) + 88 * 512, std::begin(v28) + 88 * 512 + 1024);
  std::vector<float> v517(944);
  std::copy(v516.begin() + 0, v516.begin() + 0 + 944, v517.begin());
  std::vector<float> v518(80);
  std::copy(v516.begin() + 944, v516.begin() + 944 + 80, v518.begin());
  std::copy(v517.begin(), v517.end(), v86.begin() + 80);
  std::copy(v518.begin(), v518.end(), v86.begin() + 0);
  std::vector<double> v521(std::begin(v86), std::end(v86));
  auto pt88_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt88_filled = v521;
  pt88_filled.clear();
  pt88_filled.reserve(pt88_filled_n);
  for (auto i = 0; i < pt88_filled_n; ++i) {
    pt88_filled.push_back(v521[i % v521.size()]);
  }
  auto pt88 = cc->MakeCKKSPackedPlaintext(pt88_filled);
  const auto& ct183 = cc->EvalMult(ct16, pt88);
  std::vector<float> v522(std::begin(v28) + 89 * 512, std::begin(v28) + 89 * 512 + 1024);
  std::vector<float> v523(944);
  std::copy(v522.begin() + 0, v522.begin() + 0 + 944, v523.begin());
  std::vector<float> v524(80);
  std::copy(v522.begin() + 944, v522.begin() + 944 + 80, v524.begin());
  std::copy(v523.begin(), v523.end(), v86.begin() + 80);
  std::copy(v524.begin(), v524.end(), v86.begin() + 0);
  std::vector<double> v527(std::begin(v86), std::end(v86));
  auto pt89_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt89_filled = v527;
  pt89_filled.clear();
  pt89_filled.reserve(pt89_filled_n);
  for (auto i = 0; i < pt89_filled_n; ++i) {
    pt89_filled.push_back(v527[i % v527.size()]);
  }
  auto pt89 = cc->MakeCKKSPackedPlaintext(pt89_filled);
  const auto& ct184 = cc->EvalMult(ct18, pt89);
  std::vector<float> v528(std::begin(v28) + 90 * 512, std::begin(v28) + 90 * 512 + 1024);
  std::vector<float> v529(944);
  std::copy(v528.begin() + 0, v528.begin() + 0 + 944, v529.begin());
  std::vector<float> v530(80);
  std::copy(v528.begin() + 944, v528.begin() + 944 + 80, v530.begin());
  std::copy(v529.begin(), v529.end(), v86.begin() + 80);
  std::copy(v530.begin(), v530.end(), v86.begin() + 0);
  std::vector<double> v533(std::begin(v86), std::end(v86));
  auto pt90_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt90_filled = v533;
  pt90_filled.clear();
  pt90_filled.reserve(pt90_filled_n);
  for (auto i = 0; i < pt90_filled_n; ++i) {
    pt90_filled.push_back(v533[i % v533.size()]);
  }
  auto pt90 = cc->MakeCKKSPackedPlaintext(pt90_filled);
  const auto& ct185 = cc->EvalMult(ct20, pt90);
  std::vector<float> v534(std::begin(v28) + 91 * 512, std::begin(v28) + 91 * 512 + 1024);
  std::vector<float> v535(944);
  std::copy(v534.begin() + 0, v534.begin() + 0 + 944, v535.begin());
  std::vector<float> v536(80);
  std::copy(v534.begin() + 944, v534.begin() + 944 + 80, v536.begin());
  std::copy(v535.begin(), v535.end(), v86.begin() + 80);
  std::copy(v536.begin(), v536.end(), v86.begin() + 0);
  std::vector<double> v539(std::begin(v86), std::end(v86));
  auto pt91_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt91_filled = v539;
  pt91_filled.clear();
  pt91_filled.reserve(pt91_filled_n);
  for (auto i = 0; i < pt91_filled_n; ++i) {
    pt91_filled.push_back(v539[i % v539.size()]);
  }
  auto pt91 = cc->MakeCKKSPackedPlaintext(pt91_filled);
  const auto& ct186 = cc->EvalMult(ct22, pt91);
  std::vector<float> v540(std::begin(v28) + 92 * 512, std::begin(v28) + 92 * 512 + 1024);
  std::vector<float> v541(944);
  std::copy(v540.begin() + 0, v540.begin() + 0 + 944, v541.begin());
  std::vector<float> v542(80);
  std::copy(v540.begin() + 944, v540.begin() + 944 + 80, v542.begin());
  std::copy(v541.begin(), v541.end(), v86.begin() + 80);
  std::copy(v542.begin(), v542.end(), v86.begin() + 0);
  std::vector<double> v545(std::begin(v86), std::end(v86));
  auto pt92_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt92_filled = v545;
  pt92_filled.clear();
  pt92_filled.reserve(pt92_filled_n);
  for (auto i = 0; i < pt92_filled_n; ++i) {
    pt92_filled.push_back(v545[i % v545.size()]);
  }
  auto pt92 = cc->MakeCKKSPackedPlaintext(pt92_filled);
  const auto& ct187 = cc->EvalMult(ct24, pt92);
  std::vector<float> v546(std::begin(v28) + 93 * 512, std::begin(v28) + 93 * 512 + 1024);
  std::vector<float> v547(944);
  std::copy(v546.begin() + 0, v546.begin() + 0 + 944, v547.begin());
  std::vector<float> v548(80);
  std::copy(v546.begin() + 944, v546.begin() + 944 + 80, v548.begin());
  std::copy(v547.begin(), v547.end(), v86.begin() + 80);
  std::copy(v548.begin(), v548.end(), v86.begin() + 0);
  std::vector<double> v551(std::begin(v86), std::end(v86));
  auto pt93_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt93_filled = v551;
  pt93_filled.clear();
  pt93_filled.reserve(pt93_filled_n);
  for (auto i = 0; i < pt93_filled_n; ++i) {
    pt93_filled.push_back(v551[i % v551.size()]);
  }
  auto pt93 = cc->MakeCKKSPackedPlaintext(pt93_filled);
  const auto& ct188 = cc->EvalMult(ct26, pt93);
  std::vector<float> v552(std::begin(v28) + 94 * 512, std::begin(v28) + 94 * 512 + 1024);
  std::vector<float> v553(944);
  std::copy(v552.begin() + 0, v552.begin() + 0 + 944, v553.begin());
  std::vector<float> v554(80);
  std::copy(v552.begin() + 944, v552.begin() + 944 + 80, v554.begin());
  std::copy(v553.begin(), v553.end(), v86.begin() + 80);
  std::copy(v554.begin(), v554.end(), v86.begin() + 0);
  std::vector<double> v557(std::begin(v86), std::end(v86));
  auto pt94_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt94_filled = v557;
  pt94_filled.clear();
  pt94_filled.reserve(pt94_filled_n);
  for (auto i = 0; i < pt94_filled_n; ++i) {
    pt94_filled.push_back(v557[i % v557.size()]);
  }
  auto pt94 = cc->MakeCKKSPackedPlaintext(pt94_filled);
  const auto& ct189 = cc->EvalMult(ct28, pt94);
  std::vector<float> v558(std::begin(v28) + 95 * 512, std::begin(v28) + 95 * 512 + 1024);
  std::vector<float> v559(944);
  std::copy(v558.begin() + 0, v558.begin() + 0 + 944, v559.begin());
  std::vector<float> v560(80);
  std::copy(v558.begin() + 944, v558.begin() + 944 + 80, v560.begin());
  std::copy(v559.begin(), v559.end(), v86.begin() + 80);
  std::copy(v560.begin(), v560.end(), v86.begin() + 0);
  std::vector<double> v563(std::begin(v86), std::end(v86));
  auto pt95_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt95_filled = v563;
  pt95_filled.clear();
  pt95_filled.reserve(pt95_filled_n);
  for (auto i = 0; i < pt95_filled_n; ++i) {
    pt95_filled.push_back(v563[i % v563.size()]);
  }
  auto pt95 = cc->MakeCKKSPackedPlaintext(pt95_filled);
  const auto& ct190 = cc->EvalMult(ct30, pt95);
  const auto& ct191 = cc->EvalAdd(ct175, ct176);
  const auto& ct192 = cc->EvalAdd(ct177, ct178);
  const auto& ct193 = cc->EvalAdd(ct191, ct192);
  const auto& ct194 = cc->EvalAdd(ct179, ct180);
  const auto& ct195 = cc->EvalAdd(ct181, ct182);
  const auto& ct196 = cc->EvalAdd(ct194, ct195);
  const auto& ct197 = cc->EvalAdd(ct193, ct196);
  const auto& ct198 = cc->EvalAdd(ct183, ct184);
  const auto& ct199 = cc->EvalAdd(ct185, ct186);
  const auto& ct200 = cc->EvalAdd(ct198, ct199);
  const auto& ct201 = cc->EvalAdd(ct187, ct188);
  const auto& ct202 = cc->EvalAdd(ct189, ct190);
  const auto& ct203 = cc->EvalAdd(ct201, ct202);
  const auto& ct204 = cc->EvalAdd(ct200, ct203);
  const auto& ct205 = cc->EvalAdd(ct197, ct204);
  const auto& ct206 = cc->EvalRotate(ct205, 80);
  std::vector<float> v564(std::begin(v28) + 96 * 512, std::begin(v28) + 96 * 512 + 1024);
  std::vector<float> v565(928);
  std::copy(v564.begin() + 0, v564.begin() + 0 + 928, v565.begin());
  std::vector<float> v566(96);
  std::copy(v564.begin() + 928, v564.begin() + 928 + 96, v566.begin());
  std::copy(v565.begin(), v565.end(), v86.begin() + 96);
  std::copy(v566.begin(), v566.end(), v86.begin() + 0);
  std::vector<double> v569(std::begin(v86), std::end(v86));
  auto pt96_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt96_filled = v569;
  pt96_filled.clear();
  pt96_filled.reserve(pt96_filled_n);
  for (auto i = 0; i < pt96_filled_n; ++i) {
    pt96_filled.push_back(v569[i % v569.size()]);
  }
  auto pt96 = cc->MakeCKKSPackedPlaintext(pt96_filled);
  const auto& ct207 = cc->EvalMult(ct, pt96);
  std::vector<float> v570(std::begin(v28) + 97 * 512, std::begin(v28) + 97 * 512 + 1024);
  std::vector<float> v571(928);
  std::copy(v570.begin() + 0, v570.begin() + 0 + 928, v571.begin());
  std::vector<float> v572(96);
  std::copy(v570.begin() + 928, v570.begin() + 928 + 96, v572.begin());
  std::copy(v571.begin(), v571.end(), v86.begin() + 96);
  std::copy(v572.begin(), v572.end(), v86.begin() + 0);
  std::vector<double> v575(std::begin(v86), std::end(v86));
  auto pt97_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt97_filled = v575;
  pt97_filled.clear();
  pt97_filled.reserve(pt97_filled_n);
  for (auto i = 0; i < pt97_filled_n; ++i) {
    pt97_filled.push_back(v575[i % v575.size()]);
  }
  auto pt97 = cc->MakeCKKSPackedPlaintext(pt97_filled);
  const auto& ct208 = cc->EvalMult(ct2, pt97);
  std::vector<float> v576(std::begin(v28) + 98 * 512, std::begin(v28) + 98 * 512 + 1024);
  std::vector<float> v577(928);
  std::copy(v576.begin() + 0, v576.begin() + 0 + 928, v577.begin());
  std::vector<float> v578(96);
  std::copy(v576.begin() + 928, v576.begin() + 928 + 96, v578.begin());
  std::copy(v577.begin(), v577.end(), v86.begin() + 96);
  std::copy(v578.begin(), v578.end(), v86.begin() + 0);
  std::vector<double> v581(std::begin(v86), std::end(v86));
  auto pt98_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt98_filled = v581;
  pt98_filled.clear();
  pt98_filled.reserve(pt98_filled_n);
  for (auto i = 0; i < pt98_filled_n; ++i) {
    pt98_filled.push_back(v581[i % v581.size()]);
  }
  auto pt98 = cc->MakeCKKSPackedPlaintext(pt98_filled);
  const auto& ct209 = cc->EvalMult(ct4, pt98);
  std::vector<float> v582(std::begin(v28) + 99 * 512, std::begin(v28) + 99 * 512 + 1024);
  std::vector<float> v583(928);
  std::copy(v582.begin() + 0, v582.begin() + 0 + 928, v583.begin());
  std::vector<float> v584(96);
  std::copy(v582.begin() + 928, v582.begin() + 928 + 96, v584.begin());
  std::copy(v583.begin(), v583.end(), v86.begin() + 96);
  std::copy(v584.begin(), v584.end(), v86.begin() + 0);
  std::vector<double> v587(std::begin(v86), std::end(v86));
  auto pt99_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt99_filled = v587;
  pt99_filled.clear();
  pt99_filled.reserve(pt99_filled_n);
  for (auto i = 0; i < pt99_filled_n; ++i) {
    pt99_filled.push_back(v587[i % v587.size()]);
  }
  auto pt99 = cc->MakeCKKSPackedPlaintext(pt99_filled);
  const auto& ct210 = cc->EvalMult(ct6, pt99);
  std::vector<float> v588(std::begin(v28) + 100 * 512, std::begin(v28) + 100 * 512 + 1024);
  std::vector<float> v589(928);
  std::copy(v588.begin() + 0, v588.begin() + 0 + 928, v589.begin());
  std::vector<float> v590(96);
  std::copy(v588.begin() + 928, v588.begin() + 928 + 96, v590.begin());
  std::copy(v589.begin(), v589.end(), v86.begin() + 96);
  std::copy(v590.begin(), v590.end(), v86.begin() + 0);
  std::vector<double> v593(std::begin(v86), std::end(v86));
  auto pt100_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt100_filled = v593;
  pt100_filled.clear();
  pt100_filled.reserve(pt100_filled_n);
  for (auto i = 0; i < pt100_filled_n; ++i) {
    pt100_filled.push_back(v593[i % v593.size()]);
  }
  auto pt100 = cc->MakeCKKSPackedPlaintext(pt100_filled);
  const auto& ct211 = cc->EvalMult(ct8, pt100);
  std::vector<float> v594(std::begin(v28) + 101 * 512, std::begin(v28) + 101 * 512 + 1024);
  std::vector<float> v595(928);
  std::copy(v594.begin() + 0, v594.begin() + 0 + 928, v595.begin());
  std::vector<float> v596(96);
  std::copy(v594.begin() + 928, v594.begin() + 928 + 96, v596.begin());
  std::copy(v595.begin(), v595.end(), v86.begin() + 96);
  std::copy(v596.begin(), v596.end(), v86.begin() + 0);
  std::vector<double> v599(std::begin(v86), std::end(v86));
  auto pt101_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt101_filled = v599;
  pt101_filled.clear();
  pt101_filled.reserve(pt101_filled_n);
  for (auto i = 0; i < pt101_filled_n; ++i) {
    pt101_filled.push_back(v599[i % v599.size()]);
  }
  auto pt101 = cc->MakeCKKSPackedPlaintext(pt101_filled);
  const auto& ct212 = cc->EvalMult(ct10, pt101);
  std::vector<float> v600(std::begin(v28) + 102 * 512, std::begin(v28) + 102 * 512 + 1024);
  std::vector<float> v601(928);
  std::copy(v600.begin() + 0, v600.begin() + 0 + 928, v601.begin());
  std::vector<float> v602(96);
  std::copy(v600.begin() + 928, v600.begin() + 928 + 96, v602.begin());
  std::copy(v601.begin(), v601.end(), v86.begin() + 96);
  std::copy(v602.begin(), v602.end(), v86.begin() + 0);
  std::vector<double> v605(std::begin(v86), std::end(v86));
  auto pt102_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt102_filled = v605;
  pt102_filled.clear();
  pt102_filled.reserve(pt102_filled_n);
  for (auto i = 0; i < pt102_filled_n; ++i) {
    pt102_filled.push_back(v605[i % v605.size()]);
  }
  auto pt102 = cc->MakeCKKSPackedPlaintext(pt102_filled);
  const auto& ct213 = cc->EvalMult(ct12, pt102);
  std::vector<float> v606(std::begin(v28) + 103 * 512, std::begin(v28) + 103 * 512 + 1024);
  std::vector<float> v607(928);
  std::copy(v606.begin() + 0, v606.begin() + 0 + 928, v607.begin());
  std::vector<float> v608(96);
  std::copy(v606.begin() + 928, v606.begin() + 928 + 96, v608.begin());
  std::copy(v607.begin(), v607.end(), v86.begin() + 96);
  std::copy(v608.begin(), v608.end(), v86.begin() + 0);
  std::vector<double> v611(std::begin(v86), std::end(v86));
  auto pt103_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt103_filled = v611;
  pt103_filled.clear();
  pt103_filled.reserve(pt103_filled_n);
  for (auto i = 0; i < pt103_filled_n; ++i) {
    pt103_filled.push_back(v611[i % v611.size()]);
  }
  auto pt103 = cc->MakeCKKSPackedPlaintext(pt103_filled);
  const auto& ct214 = cc->EvalMult(ct14, pt103);
  std::vector<float> v612(std::begin(v28) + 104 * 512, std::begin(v28) + 104 * 512 + 1024);
  std::vector<float> v613(928);
  std::copy(v612.begin() + 0, v612.begin() + 0 + 928, v613.begin());
  std::vector<float> v614(96);
  std::copy(v612.begin() + 928, v612.begin() + 928 + 96, v614.begin());
  std::copy(v613.begin(), v613.end(), v86.begin() + 96);
  std::copy(v614.begin(), v614.end(), v86.begin() + 0);
  std::vector<double> v617(std::begin(v86), std::end(v86));
  auto pt104_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt104_filled = v617;
  pt104_filled.clear();
  pt104_filled.reserve(pt104_filled_n);
  for (auto i = 0; i < pt104_filled_n; ++i) {
    pt104_filled.push_back(v617[i % v617.size()]);
  }
  auto pt104 = cc->MakeCKKSPackedPlaintext(pt104_filled);
  const auto& ct215 = cc->EvalMult(ct16, pt104);
  std::vector<float> v618(std::begin(v28) + 105 * 512, std::begin(v28) + 105 * 512 + 1024);
  std::vector<float> v619(928);
  std::copy(v618.begin() + 0, v618.begin() + 0 + 928, v619.begin());
  std::vector<float> v620(96);
  std::copy(v618.begin() + 928, v618.begin() + 928 + 96, v620.begin());
  std::copy(v619.begin(), v619.end(), v86.begin() + 96);
  std::copy(v620.begin(), v620.end(), v86.begin() + 0);
  std::vector<double> v623(std::begin(v86), std::end(v86));
  auto pt105_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt105_filled = v623;
  pt105_filled.clear();
  pt105_filled.reserve(pt105_filled_n);
  for (auto i = 0; i < pt105_filled_n; ++i) {
    pt105_filled.push_back(v623[i % v623.size()]);
  }
  auto pt105 = cc->MakeCKKSPackedPlaintext(pt105_filled);
  const auto& ct216 = cc->EvalMult(ct18, pt105);
  std::vector<float> v624(std::begin(v28) + 106 * 512, std::begin(v28) + 106 * 512 + 1024);
  std::vector<float> v625(928);
  std::copy(v624.begin() + 0, v624.begin() + 0 + 928, v625.begin());
  std::vector<float> v626(96);
  std::copy(v624.begin() + 928, v624.begin() + 928 + 96, v626.begin());
  std::copy(v625.begin(), v625.end(), v86.begin() + 96);
  std::copy(v626.begin(), v626.end(), v86.begin() + 0);
  std::vector<double> v629(std::begin(v86), std::end(v86));
  auto pt106_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt106_filled = v629;
  pt106_filled.clear();
  pt106_filled.reserve(pt106_filled_n);
  for (auto i = 0; i < pt106_filled_n; ++i) {
    pt106_filled.push_back(v629[i % v629.size()]);
  }
  auto pt106 = cc->MakeCKKSPackedPlaintext(pt106_filled);
  const auto& ct217 = cc->EvalMult(ct20, pt106);
  std::vector<float> v630(std::begin(v28) + 107 * 512, std::begin(v28) + 107 * 512 + 1024);
  std::vector<float> v631(928);
  std::copy(v630.begin() + 0, v630.begin() + 0 + 928, v631.begin());
  std::vector<float> v632(96);
  std::copy(v630.begin() + 928, v630.begin() + 928 + 96, v632.begin());
  std::copy(v631.begin(), v631.end(), v86.begin() + 96);
  std::copy(v632.begin(), v632.end(), v86.begin() + 0);
  std::vector<double> v635(std::begin(v86), std::end(v86));
  auto pt107_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt107_filled = v635;
  pt107_filled.clear();
  pt107_filled.reserve(pt107_filled_n);
  for (auto i = 0; i < pt107_filled_n; ++i) {
    pt107_filled.push_back(v635[i % v635.size()]);
  }
  auto pt107 = cc->MakeCKKSPackedPlaintext(pt107_filled);
  const auto& ct218 = cc->EvalMult(ct22, pt107);
  std::vector<float> v636(std::begin(v28) + 108 * 512, std::begin(v28) + 108 * 512 + 1024);
  std::vector<float> v637(928);
  std::copy(v636.begin() + 0, v636.begin() + 0 + 928, v637.begin());
  std::vector<float> v638(96);
  std::copy(v636.begin() + 928, v636.begin() + 928 + 96, v638.begin());
  std::copy(v637.begin(), v637.end(), v86.begin() + 96);
  std::copy(v638.begin(), v638.end(), v86.begin() + 0);
  std::vector<double> v641(std::begin(v86), std::end(v86));
  auto pt108_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt108_filled = v641;
  pt108_filled.clear();
  pt108_filled.reserve(pt108_filled_n);
  for (auto i = 0; i < pt108_filled_n; ++i) {
    pt108_filled.push_back(v641[i % v641.size()]);
  }
  auto pt108 = cc->MakeCKKSPackedPlaintext(pt108_filled);
  const auto& ct219 = cc->EvalMult(ct24, pt108);
  std::vector<float> v642(std::begin(v28) + 109 * 512, std::begin(v28) + 109 * 512 + 1024);
  std::vector<float> v643(928);
  std::copy(v642.begin() + 0, v642.begin() + 0 + 928, v643.begin());
  std::vector<float> v644(96);
  std::copy(v642.begin() + 928, v642.begin() + 928 + 96, v644.begin());
  std::copy(v643.begin(), v643.end(), v86.begin() + 96);
  std::copy(v644.begin(), v644.end(), v86.begin() + 0);
  std::vector<double> v647(std::begin(v86), std::end(v86));
  auto pt109_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt109_filled = v647;
  pt109_filled.clear();
  pt109_filled.reserve(pt109_filled_n);
  for (auto i = 0; i < pt109_filled_n; ++i) {
    pt109_filled.push_back(v647[i % v647.size()]);
  }
  auto pt109 = cc->MakeCKKSPackedPlaintext(pt109_filled);
  const auto& ct220 = cc->EvalMult(ct26, pt109);
  std::vector<float> v648(std::begin(v28) + 110 * 512, std::begin(v28) + 110 * 512 + 1024);
  std::vector<float> v649(928);
  std::copy(v648.begin() + 0, v648.begin() + 0 + 928, v649.begin());
  std::vector<float> v650(96);
  std::copy(v648.begin() + 928, v648.begin() + 928 + 96, v650.begin());
  std::copy(v649.begin(), v649.end(), v86.begin() + 96);
  std::copy(v650.begin(), v650.end(), v86.begin() + 0);
  std::vector<double> v653(std::begin(v86), std::end(v86));
  auto pt110_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt110_filled = v653;
  pt110_filled.clear();
  pt110_filled.reserve(pt110_filled_n);
  for (auto i = 0; i < pt110_filled_n; ++i) {
    pt110_filled.push_back(v653[i % v653.size()]);
  }
  auto pt110 = cc->MakeCKKSPackedPlaintext(pt110_filled);
  const auto& ct221 = cc->EvalMult(ct28, pt110);
  std::vector<float> v654(std::begin(v28) + 111 * 512, std::begin(v28) + 111 * 512 + 1024);
  std::vector<float> v655(928);
  std::copy(v654.begin() + 0, v654.begin() + 0 + 928, v655.begin());
  std::vector<float> v656(96);
  std::copy(v654.begin() + 928, v654.begin() + 928 + 96, v656.begin());
  std::copy(v655.begin(), v655.end(), v86.begin() + 96);
  std::copy(v656.begin(), v656.end(), v86.begin() + 0);
  std::vector<double> v659(std::begin(v86), std::end(v86));
  auto pt111_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt111_filled = v659;
  pt111_filled.clear();
  pt111_filled.reserve(pt111_filled_n);
  for (auto i = 0; i < pt111_filled_n; ++i) {
    pt111_filled.push_back(v659[i % v659.size()]);
  }
  auto pt111 = cc->MakeCKKSPackedPlaintext(pt111_filled);
  const auto& ct222 = cc->EvalMult(ct30, pt111);
  const auto& ct223 = cc->EvalAdd(ct207, ct208);
  const auto& ct224 = cc->EvalAdd(ct209, ct210);
  const auto& ct225 = cc->EvalAdd(ct223, ct224);
  const auto& ct226 = cc->EvalAdd(ct211, ct212);
  const auto& ct227 = cc->EvalAdd(ct213, ct214);
  const auto& ct228 = cc->EvalAdd(ct226, ct227);
  const auto& ct229 = cc->EvalAdd(ct225, ct228);
  const auto& ct230 = cc->EvalAdd(ct215, ct216);
  const auto& ct231 = cc->EvalAdd(ct217, ct218);
  const auto& ct232 = cc->EvalAdd(ct230, ct231);
  const auto& ct233 = cc->EvalAdd(ct219, ct220);
  const auto& ct234 = cc->EvalAdd(ct221, ct222);
  const auto& ct235 = cc->EvalAdd(ct233, ct234);
  const auto& ct236 = cc->EvalAdd(ct232, ct235);
  const auto& ct237 = cc->EvalAdd(ct229, ct236);
  const auto& ct238 = cc->EvalRotate(ct237, 96);
  std::vector<float> v660(std::begin(v28) + 112 * 512, std::begin(v28) + 112 * 512 + 1024);
  std::vector<float> v661(912);
  std::copy(v660.begin() + 0, v660.begin() + 0 + 912, v661.begin());
  std::vector<float> v662(112);
  std::copy(v660.begin() + 912, v660.begin() + 912 + 112, v662.begin());
  std::copy(v661.begin(), v661.end(), v86.begin() + 112);
  std::copy(v662.begin(), v662.end(), v86.begin() + 0);
  std::vector<double> v665(std::begin(v86), std::end(v86));
  auto pt112_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt112_filled = v665;
  pt112_filled.clear();
  pt112_filled.reserve(pt112_filled_n);
  for (auto i = 0; i < pt112_filled_n; ++i) {
    pt112_filled.push_back(v665[i % v665.size()]);
  }
  auto pt112 = cc->MakeCKKSPackedPlaintext(pt112_filled);
  const auto& ct239 = cc->EvalMult(ct, pt112);
  std::vector<float> v666(std::begin(v28) + 113 * 512, std::begin(v28) + 113 * 512 + 1024);
  std::vector<float> v667(912);
  std::copy(v666.begin() + 0, v666.begin() + 0 + 912, v667.begin());
  std::vector<float> v668(112);
  std::copy(v666.begin() + 912, v666.begin() + 912 + 112, v668.begin());
  std::copy(v667.begin(), v667.end(), v86.begin() + 112);
  std::copy(v668.begin(), v668.end(), v86.begin() + 0);
  std::vector<double> v671(std::begin(v86), std::end(v86));
  auto pt113_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt113_filled = v671;
  pt113_filled.clear();
  pt113_filled.reserve(pt113_filled_n);
  for (auto i = 0; i < pt113_filled_n; ++i) {
    pt113_filled.push_back(v671[i % v671.size()]);
  }
  auto pt113 = cc->MakeCKKSPackedPlaintext(pt113_filled);
  const auto& ct240 = cc->EvalMult(ct2, pt113);
  std::vector<float> v672(std::begin(v28) + 114 * 512, std::begin(v28) + 114 * 512 + 1024);
  std::vector<float> v673(912);
  std::copy(v672.begin() + 0, v672.begin() + 0 + 912, v673.begin());
  std::vector<float> v674(112);
  std::copy(v672.begin() + 912, v672.begin() + 912 + 112, v674.begin());
  std::copy(v673.begin(), v673.end(), v86.begin() + 112);
  std::copy(v674.begin(), v674.end(), v86.begin() + 0);
  std::vector<double> v677(std::begin(v86), std::end(v86));
  auto pt114_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt114_filled = v677;
  pt114_filled.clear();
  pt114_filled.reserve(pt114_filled_n);
  for (auto i = 0; i < pt114_filled_n; ++i) {
    pt114_filled.push_back(v677[i % v677.size()]);
  }
  auto pt114 = cc->MakeCKKSPackedPlaintext(pt114_filled);
  const auto& ct241 = cc->EvalMult(ct4, pt114);
  std::vector<float> v678(std::begin(v28) + 115 * 512, std::begin(v28) + 115 * 512 + 1024);
  std::vector<float> v679(912);
  std::copy(v678.begin() + 0, v678.begin() + 0 + 912, v679.begin());
  std::vector<float> v680(112);
  std::copy(v678.begin() + 912, v678.begin() + 912 + 112, v680.begin());
  std::copy(v679.begin(), v679.end(), v86.begin() + 112);
  std::copy(v680.begin(), v680.end(), v86.begin() + 0);
  std::vector<double> v683(std::begin(v86), std::end(v86));
  auto pt115_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt115_filled = v683;
  pt115_filled.clear();
  pt115_filled.reserve(pt115_filled_n);
  for (auto i = 0; i < pt115_filled_n; ++i) {
    pt115_filled.push_back(v683[i % v683.size()]);
  }
  auto pt115 = cc->MakeCKKSPackedPlaintext(pt115_filled);
  const auto& ct242 = cc->EvalMult(ct6, pt115);
  std::vector<float> v684(std::begin(v28) + 116 * 512, std::begin(v28) + 116 * 512 + 1024);
  std::vector<float> v685(912);
  std::copy(v684.begin() + 0, v684.begin() + 0 + 912, v685.begin());
  std::vector<float> v686(112);
  std::copy(v684.begin() + 912, v684.begin() + 912 + 112, v686.begin());
  std::copy(v685.begin(), v685.end(), v86.begin() + 112);
  std::copy(v686.begin(), v686.end(), v86.begin() + 0);
  std::vector<double> v689(std::begin(v86), std::end(v86));
  auto pt116_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt116_filled = v689;
  pt116_filled.clear();
  pt116_filled.reserve(pt116_filled_n);
  for (auto i = 0; i < pt116_filled_n; ++i) {
    pt116_filled.push_back(v689[i % v689.size()]);
  }
  auto pt116 = cc->MakeCKKSPackedPlaintext(pt116_filled);
  const auto& ct243 = cc->EvalMult(ct8, pt116);
  std::vector<float> v690(std::begin(v28) + 117 * 512, std::begin(v28) + 117 * 512 + 1024);
  std::vector<float> v691(912);
  std::copy(v690.begin() + 0, v690.begin() + 0 + 912, v691.begin());
  std::vector<float> v692(112);
  std::copy(v690.begin() + 912, v690.begin() + 912 + 112, v692.begin());
  std::copy(v691.begin(), v691.end(), v86.begin() + 112);
  std::copy(v692.begin(), v692.end(), v86.begin() + 0);
  std::vector<double> v695(std::begin(v86), std::end(v86));
  auto pt117_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt117_filled = v695;
  pt117_filled.clear();
  pt117_filled.reserve(pt117_filled_n);
  for (auto i = 0; i < pt117_filled_n; ++i) {
    pt117_filled.push_back(v695[i % v695.size()]);
  }
  auto pt117 = cc->MakeCKKSPackedPlaintext(pt117_filled);
  const auto& ct244 = cc->EvalMult(ct10, pt117);
  std::vector<float> v696(std::begin(v28) + 118 * 512, std::begin(v28) + 118 * 512 + 1024);
  std::vector<float> v697(912);
  std::copy(v696.begin() + 0, v696.begin() + 0 + 912, v697.begin());
  std::vector<float> v698(112);
  std::copy(v696.begin() + 912, v696.begin() + 912 + 112, v698.begin());
  std::copy(v697.begin(), v697.end(), v86.begin() + 112);
  std::copy(v698.begin(), v698.end(), v86.begin() + 0);
  std::vector<double> v701(std::begin(v86), std::end(v86));
  auto pt118_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt118_filled = v701;
  pt118_filled.clear();
  pt118_filled.reserve(pt118_filled_n);
  for (auto i = 0; i < pt118_filled_n; ++i) {
    pt118_filled.push_back(v701[i % v701.size()]);
  }
  auto pt118 = cc->MakeCKKSPackedPlaintext(pt118_filled);
  const auto& ct245 = cc->EvalMult(ct12, pt118);
  std::vector<float> v702(std::begin(v28) + 119 * 512, std::begin(v28) + 119 * 512 + 1024);
  std::vector<float> v703(912);
  std::copy(v702.begin() + 0, v702.begin() + 0 + 912, v703.begin());
  std::vector<float> v704(112);
  std::copy(v702.begin() + 912, v702.begin() + 912 + 112, v704.begin());
  std::copy(v703.begin(), v703.end(), v86.begin() + 112);
  std::copy(v704.begin(), v704.end(), v86.begin() + 0);
  std::vector<double> v707(std::begin(v86), std::end(v86));
  auto pt119_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt119_filled = v707;
  pt119_filled.clear();
  pt119_filled.reserve(pt119_filled_n);
  for (auto i = 0; i < pt119_filled_n; ++i) {
    pt119_filled.push_back(v707[i % v707.size()]);
  }
  auto pt119 = cc->MakeCKKSPackedPlaintext(pt119_filled);
  const auto& ct246 = cc->EvalMult(ct14, pt119);
  std::vector<float> v708(std::begin(v28) + 120 * 512, std::begin(v28) + 120 * 512 + 1024);
  std::vector<float> v709(912);
  std::copy(v708.begin() + 0, v708.begin() + 0 + 912, v709.begin());
  std::vector<float> v710(112);
  std::copy(v708.begin() + 912, v708.begin() + 912 + 112, v710.begin());
  std::copy(v709.begin(), v709.end(), v86.begin() + 112);
  std::copy(v710.begin(), v710.end(), v86.begin() + 0);
  std::vector<double> v713(std::begin(v86), std::end(v86));
  auto pt120_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt120_filled = v713;
  pt120_filled.clear();
  pt120_filled.reserve(pt120_filled_n);
  for (auto i = 0; i < pt120_filled_n; ++i) {
    pt120_filled.push_back(v713[i % v713.size()]);
  }
  auto pt120 = cc->MakeCKKSPackedPlaintext(pt120_filled);
  const auto& ct247 = cc->EvalMult(ct16, pt120);
  std::vector<float> v714(std::begin(v28) + 121 * 512, std::begin(v28) + 121 * 512 + 1024);
  std::vector<float> v715(912);
  std::copy(v714.begin() + 0, v714.begin() + 0 + 912, v715.begin());
  std::vector<float> v716(112);
  std::copy(v714.begin() + 912, v714.begin() + 912 + 112, v716.begin());
  std::copy(v715.begin(), v715.end(), v86.begin() + 112);
  std::copy(v716.begin(), v716.end(), v86.begin() + 0);
  std::vector<double> v719(std::begin(v86), std::end(v86));
  auto pt121_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt121_filled = v719;
  pt121_filled.clear();
  pt121_filled.reserve(pt121_filled_n);
  for (auto i = 0; i < pt121_filled_n; ++i) {
    pt121_filled.push_back(v719[i % v719.size()]);
  }
  auto pt121 = cc->MakeCKKSPackedPlaintext(pt121_filled);
  const auto& ct248 = cc->EvalMult(ct18, pt121);
  std::vector<float> v720(std::begin(v28) + 122 * 512, std::begin(v28) + 122 * 512 + 1024);
  std::vector<float> v721(912);
  std::copy(v720.begin() + 0, v720.begin() + 0 + 912, v721.begin());
  std::vector<float> v722(112);
  std::copy(v720.begin() + 912, v720.begin() + 912 + 112, v722.begin());
  std::copy(v721.begin(), v721.end(), v86.begin() + 112);
  std::copy(v722.begin(), v722.end(), v86.begin() + 0);
  std::vector<double> v725(std::begin(v86), std::end(v86));
  auto pt122_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt122_filled = v725;
  pt122_filled.clear();
  pt122_filled.reserve(pt122_filled_n);
  for (auto i = 0; i < pt122_filled_n; ++i) {
    pt122_filled.push_back(v725[i % v725.size()]);
  }
  auto pt122 = cc->MakeCKKSPackedPlaintext(pt122_filled);
  const auto& ct249 = cc->EvalMult(ct20, pt122);
  std::vector<float> v726(std::begin(v28) + 123 * 512, std::begin(v28) + 123 * 512 + 1024);
  std::vector<float> v727(912);
  std::copy(v726.begin() + 0, v726.begin() + 0 + 912, v727.begin());
  std::vector<float> v728(112);
  std::copy(v726.begin() + 912, v726.begin() + 912 + 112, v728.begin());
  std::copy(v727.begin(), v727.end(), v86.begin() + 112);
  std::copy(v728.begin(), v728.end(), v86.begin() + 0);
  std::vector<double> v731(std::begin(v86), std::end(v86));
  auto pt123_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt123_filled = v731;
  pt123_filled.clear();
  pt123_filled.reserve(pt123_filled_n);
  for (auto i = 0; i < pt123_filled_n; ++i) {
    pt123_filled.push_back(v731[i % v731.size()]);
  }
  auto pt123 = cc->MakeCKKSPackedPlaintext(pt123_filled);
  const auto& ct250 = cc->EvalMult(ct22, pt123);
  std::vector<float> v732(std::begin(v28) + 124 * 512, std::begin(v28) + 124 * 512 + 1024);
  std::vector<float> v733(912);
  std::copy(v732.begin() + 0, v732.begin() + 0 + 912, v733.begin());
  std::vector<float> v734(112);
  std::copy(v732.begin() + 912, v732.begin() + 912 + 112, v734.begin());
  std::copy(v733.begin(), v733.end(), v86.begin() + 112);
  std::copy(v734.begin(), v734.end(), v86.begin() + 0);
  std::vector<double> v737(std::begin(v86), std::end(v86));
  auto pt124_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt124_filled = v737;
  pt124_filled.clear();
  pt124_filled.reserve(pt124_filled_n);
  for (auto i = 0; i < pt124_filled_n; ++i) {
    pt124_filled.push_back(v737[i % v737.size()]);
  }
  auto pt124 = cc->MakeCKKSPackedPlaintext(pt124_filled);
  const auto& ct251 = cc->EvalMult(ct24, pt124);
  std::vector<float> v738(std::begin(v28) + 125 * 512, std::begin(v28) + 125 * 512 + 1024);
  std::vector<float> v739(912);
  std::copy(v738.begin() + 0, v738.begin() + 0 + 912, v739.begin());
  std::vector<float> v740(112);
  std::copy(v738.begin() + 912, v738.begin() + 912 + 112, v740.begin());
  std::copy(v739.begin(), v739.end(), v86.begin() + 112);
  std::copy(v740.begin(), v740.end(), v86.begin() + 0);
  std::vector<double> v743(std::begin(v86), std::end(v86));
  auto pt125_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt125_filled = v743;
  pt125_filled.clear();
  pt125_filled.reserve(pt125_filled_n);
  for (auto i = 0; i < pt125_filled_n; ++i) {
    pt125_filled.push_back(v743[i % v743.size()]);
  }
  auto pt125 = cc->MakeCKKSPackedPlaintext(pt125_filled);
  const auto& ct252 = cc->EvalMult(ct26, pt125);
  std::vector<float> v744(std::begin(v28) + 126 * 512, std::begin(v28) + 126 * 512 + 1024);
  std::vector<float> v745(912);
  std::copy(v744.begin() + 0, v744.begin() + 0 + 912, v745.begin());
  std::vector<float> v746(112);
  std::copy(v744.begin() + 912, v744.begin() + 912 + 112, v746.begin());
  std::copy(v745.begin(), v745.end(), v86.begin() + 112);
  std::copy(v746.begin(), v746.end(), v86.begin() + 0);
  std::vector<double> v749(std::begin(v86), std::end(v86));
  auto pt126_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt126_filled = v749;
  pt126_filled.clear();
  pt126_filled.reserve(pt126_filled_n);
  for (auto i = 0; i < pt126_filled_n; ++i) {
    pt126_filled.push_back(v749[i % v749.size()]);
  }
  auto pt126 = cc->MakeCKKSPackedPlaintext(pt126_filled);
  const auto& ct253 = cc->EvalMult(ct28, pt126);
  std::vector<float> v750(std::begin(v28) + 127 * 512, std::begin(v28) + 127 * 512 + 1024);
  std::vector<float> v751(912);
  std::copy(v750.begin() + 0, v750.begin() + 0 + 912, v751.begin());
  std::vector<float> v752(112);
  std::copy(v750.begin() + 912, v750.begin() + 912 + 112, v752.begin());
  std::copy(v751.begin(), v751.end(), v86.begin() + 112);
  std::copy(v752.begin(), v752.end(), v86.begin() + 0);
  std::vector<double> v755(std::begin(v86), std::end(v86));
  auto pt127_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt127_filled = v755;
  pt127_filled.clear();
  pt127_filled.reserve(pt127_filled_n);
  for (auto i = 0; i < pt127_filled_n; ++i) {
    pt127_filled.push_back(v755[i % v755.size()]);
  }
  auto pt127 = cc->MakeCKKSPackedPlaintext(pt127_filled);
  const auto& ct254 = cc->EvalMult(ct30, pt127);
  const auto& ct255 = cc->EvalAdd(ct239, ct240);
  const auto& ct256 = cc->EvalAdd(ct241, ct242);
  const auto& ct257 = cc->EvalAdd(ct255, ct256);
  const auto& ct258 = cc->EvalAdd(ct243, ct244);
  const auto& ct259 = cc->EvalAdd(ct245, ct246);
  const auto& ct260 = cc->EvalAdd(ct258, ct259);
  const auto& ct261 = cc->EvalAdd(ct257, ct260);
  const auto& ct262 = cc->EvalAdd(ct247, ct248);
  const auto& ct263 = cc->EvalAdd(ct249, ct250);
  const auto& ct264 = cc->EvalAdd(ct262, ct263);
  const auto& ct265 = cc->EvalAdd(ct251, ct252);
  const auto& ct266 = cc->EvalAdd(ct253, ct254);
  const auto& ct267 = cc->EvalAdd(ct265, ct266);
  const auto& ct268 = cc->EvalAdd(ct264, ct267);
  const auto& ct269 = cc->EvalAdd(ct261, ct268);
  const auto& ct270 = cc->EvalRotate(ct269, 112);
  std::vector<float> v756(std::begin(v28) + 128 * 512, std::begin(v28) + 128 * 512 + 1024);
  std::vector<float> v757(896);
  std::copy(v756.begin() + 0, v756.begin() + 0 + 896, v757.begin());
  std::vector<float> v758(128);
  std::copy(v756.begin() + 896, v756.begin() + 896 + 128, v758.begin());
  std::copy(v757.begin(), v757.end(), v86.begin() + 128);
  std::copy(v758.begin(), v758.end(), v86.begin() + 0);
  std::vector<double> v761(std::begin(v86), std::end(v86));
  auto pt128_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt128_filled = v761;
  pt128_filled.clear();
  pt128_filled.reserve(pt128_filled_n);
  for (auto i = 0; i < pt128_filled_n; ++i) {
    pt128_filled.push_back(v761[i % v761.size()]);
  }
  auto pt128 = cc->MakeCKKSPackedPlaintext(pt128_filled);
  const auto& ct271 = cc->EvalMult(ct, pt128);
  std::vector<float> v762(std::begin(v28) + 129 * 512, std::begin(v28) + 129 * 512 + 1024);
  std::vector<float> v763(896);
  std::copy(v762.begin() + 0, v762.begin() + 0 + 896, v763.begin());
  std::vector<float> v764(128);
  std::copy(v762.begin() + 896, v762.begin() + 896 + 128, v764.begin());
  std::copy(v763.begin(), v763.end(), v86.begin() + 128);
  std::copy(v764.begin(), v764.end(), v86.begin() + 0);
  std::vector<double> v767(std::begin(v86), std::end(v86));
  auto pt129_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt129_filled = v767;
  pt129_filled.clear();
  pt129_filled.reserve(pt129_filled_n);
  for (auto i = 0; i < pt129_filled_n; ++i) {
    pt129_filled.push_back(v767[i % v767.size()]);
  }
  auto pt129 = cc->MakeCKKSPackedPlaintext(pt129_filled);
  const auto& ct272 = cc->EvalMult(ct2, pt129);
  std::vector<float> v768(std::begin(v28) + 130 * 512, std::begin(v28) + 130 * 512 + 1024);
  std::vector<float> v769(896);
  std::copy(v768.begin() + 0, v768.begin() + 0 + 896, v769.begin());
  std::vector<float> v770(128);
  std::copy(v768.begin() + 896, v768.begin() + 896 + 128, v770.begin());
  std::copy(v769.begin(), v769.end(), v86.begin() + 128);
  std::copy(v770.begin(), v770.end(), v86.begin() + 0);
  std::vector<double> v773(std::begin(v86), std::end(v86));
  auto pt130_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt130_filled = v773;
  pt130_filled.clear();
  pt130_filled.reserve(pt130_filled_n);
  for (auto i = 0; i < pt130_filled_n; ++i) {
    pt130_filled.push_back(v773[i % v773.size()]);
  }
  auto pt130 = cc->MakeCKKSPackedPlaintext(pt130_filled);
  const auto& ct273 = cc->EvalMult(ct4, pt130);
  std::vector<float> v774(std::begin(v28) + 131 * 512, std::begin(v28) + 131 * 512 + 1024);
  std::vector<float> v775(896);
  std::copy(v774.begin() + 0, v774.begin() + 0 + 896, v775.begin());
  std::vector<float> v776(128);
  std::copy(v774.begin() + 896, v774.begin() + 896 + 128, v776.begin());
  std::copy(v775.begin(), v775.end(), v86.begin() + 128);
  std::copy(v776.begin(), v776.end(), v86.begin() + 0);
  std::vector<double> v779(std::begin(v86), std::end(v86));
  auto pt131_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt131_filled = v779;
  pt131_filled.clear();
  pt131_filled.reserve(pt131_filled_n);
  for (auto i = 0; i < pt131_filled_n; ++i) {
    pt131_filled.push_back(v779[i % v779.size()]);
  }
  auto pt131 = cc->MakeCKKSPackedPlaintext(pt131_filled);
  const auto& ct274 = cc->EvalMult(ct6, pt131);
  std::vector<float> v780(std::begin(v28) + 132 * 512, std::begin(v28) + 132 * 512 + 1024);
  std::vector<float> v781(896);
  std::copy(v780.begin() + 0, v780.begin() + 0 + 896, v781.begin());
  std::vector<float> v782(128);
  std::copy(v780.begin() + 896, v780.begin() + 896 + 128, v782.begin());
  std::copy(v781.begin(), v781.end(), v86.begin() + 128);
  std::copy(v782.begin(), v782.end(), v86.begin() + 0);
  std::vector<double> v785(std::begin(v86), std::end(v86));
  auto pt132_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt132_filled = v785;
  pt132_filled.clear();
  pt132_filled.reserve(pt132_filled_n);
  for (auto i = 0; i < pt132_filled_n; ++i) {
    pt132_filled.push_back(v785[i % v785.size()]);
  }
  auto pt132 = cc->MakeCKKSPackedPlaintext(pt132_filled);
  const auto& ct275 = cc->EvalMult(ct8, pt132);
  std::vector<float> v786(std::begin(v28) + 133 * 512, std::begin(v28) + 133 * 512 + 1024);
  std::vector<float> v787(896);
  std::copy(v786.begin() + 0, v786.begin() + 0 + 896, v787.begin());
  std::vector<float> v788(128);
  std::copy(v786.begin() + 896, v786.begin() + 896 + 128, v788.begin());
  std::copy(v787.begin(), v787.end(), v86.begin() + 128);
  std::copy(v788.begin(), v788.end(), v86.begin() + 0);
  std::vector<double> v791(std::begin(v86), std::end(v86));
  auto pt133_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt133_filled = v791;
  pt133_filled.clear();
  pt133_filled.reserve(pt133_filled_n);
  for (auto i = 0; i < pt133_filled_n; ++i) {
    pt133_filled.push_back(v791[i % v791.size()]);
  }
  auto pt133 = cc->MakeCKKSPackedPlaintext(pt133_filled);
  const auto& ct276 = cc->EvalMult(ct10, pt133);
  std::vector<float> v792(std::begin(v28) + 134 * 512, std::begin(v28) + 134 * 512 + 1024);
  std::vector<float> v793(896);
  std::copy(v792.begin() + 0, v792.begin() + 0 + 896, v793.begin());
  std::vector<float> v794(128);
  std::copy(v792.begin() + 896, v792.begin() + 896 + 128, v794.begin());
  std::copy(v793.begin(), v793.end(), v86.begin() + 128);
  std::copy(v794.begin(), v794.end(), v86.begin() + 0);
  std::vector<double> v797(std::begin(v86), std::end(v86));
  auto pt134_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt134_filled = v797;
  pt134_filled.clear();
  pt134_filled.reserve(pt134_filled_n);
  for (auto i = 0; i < pt134_filled_n; ++i) {
    pt134_filled.push_back(v797[i % v797.size()]);
  }
  auto pt134 = cc->MakeCKKSPackedPlaintext(pt134_filled);
  const auto& ct277 = cc->EvalMult(ct12, pt134);
  std::vector<float> v798(std::begin(v28) + 135 * 512, std::begin(v28) + 135 * 512 + 1024);
  std::vector<float> v799(896);
  std::copy(v798.begin() + 0, v798.begin() + 0 + 896, v799.begin());
  std::vector<float> v800(128);
  std::copy(v798.begin() + 896, v798.begin() + 896 + 128, v800.begin());
  std::copy(v799.begin(), v799.end(), v86.begin() + 128);
  std::copy(v800.begin(), v800.end(), v86.begin() + 0);
  std::vector<double> v803(std::begin(v86), std::end(v86));
  auto pt135_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt135_filled = v803;
  pt135_filled.clear();
  pt135_filled.reserve(pt135_filled_n);
  for (auto i = 0; i < pt135_filled_n; ++i) {
    pt135_filled.push_back(v803[i % v803.size()]);
  }
  auto pt135 = cc->MakeCKKSPackedPlaintext(pt135_filled);
  const auto& ct278 = cc->EvalMult(ct14, pt135);
  std::vector<float> v804(std::begin(v28) + 136 * 512, std::begin(v28) + 136 * 512 + 1024);
  std::vector<float> v805(896);
  std::copy(v804.begin() + 0, v804.begin() + 0 + 896, v805.begin());
  std::vector<float> v806(128);
  std::copy(v804.begin() + 896, v804.begin() + 896 + 128, v806.begin());
  std::copy(v805.begin(), v805.end(), v86.begin() + 128);
  std::copy(v806.begin(), v806.end(), v86.begin() + 0);
  std::vector<double> v809(std::begin(v86), std::end(v86));
  auto pt136_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt136_filled = v809;
  pt136_filled.clear();
  pt136_filled.reserve(pt136_filled_n);
  for (auto i = 0; i < pt136_filled_n; ++i) {
    pt136_filled.push_back(v809[i % v809.size()]);
  }
  auto pt136 = cc->MakeCKKSPackedPlaintext(pt136_filled);
  const auto& ct279 = cc->EvalMult(ct16, pt136);
  std::vector<float> v810(std::begin(v28) + 137 * 512, std::begin(v28) + 137 * 512 + 1024);
  std::vector<float> v811(896);
  std::copy(v810.begin() + 0, v810.begin() + 0 + 896, v811.begin());
  std::vector<float> v812(128);
  std::copy(v810.begin() + 896, v810.begin() + 896 + 128, v812.begin());
  std::copy(v811.begin(), v811.end(), v86.begin() + 128);
  std::copy(v812.begin(), v812.end(), v86.begin() + 0);
  std::vector<double> v815(std::begin(v86), std::end(v86));
  auto pt137_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt137_filled = v815;
  pt137_filled.clear();
  pt137_filled.reserve(pt137_filled_n);
  for (auto i = 0; i < pt137_filled_n; ++i) {
    pt137_filled.push_back(v815[i % v815.size()]);
  }
  auto pt137 = cc->MakeCKKSPackedPlaintext(pt137_filled);
  const auto& ct280 = cc->EvalMult(ct18, pt137);
  std::vector<float> v816(std::begin(v28) + 138 * 512, std::begin(v28) + 138 * 512 + 1024);
  std::vector<float> v817(896);
  std::copy(v816.begin() + 0, v816.begin() + 0 + 896, v817.begin());
  std::vector<float> v818(128);
  std::copy(v816.begin() + 896, v816.begin() + 896 + 128, v818.begin());
  std::copy(v817.begin(), v817.end(), v86.begin() + 128);
  std::copy(v818.begin(), v818.end(), v86.begin() + 0);
  std::vector<double> v821(std::begin(v86), std::end(v86));
  auto pt138_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt138_filled = v821;
  pt138_filled.clear();
  pt138_filled.reserve(pt138_filled_n);
  for (auto i = 0; i < pt138_filled_n; ++i) {
    pt138_filled.push_back(v821[i % v821.size()]);
  }
  auto pt138 = cc->MakeCKKSPackedPlaintext(pt138_filled);
  const auto& ct281 = cc->EvalMult(ct20, pt138);
  std::vector<float> v822(std::begin(v28) + 139 * 512, std::begin(v28) + 139 * 512 + 1024);
  std::vector<float> v823(896);
  std::copy(v822.begin() + 0, v822.begin() + 0 + 896, v823.begin());
  std::vector<float> v824(128);
  std::copy(v822.begin() + 896, v822.begin() + 896 + 128, v824.begin());
  std::copy(v823.begin(), v823.end(), v86.begin() + 128);
  std::copy(v824.begin(), v824.end(), v86.begin() + 0);
  std::vector<double> v827(std::begin(v86), std::end(v86));
  auto pt139_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt139_filled = v827;
  pt139_filled.clear();
  pt139_filled.reserve(pt139_filled_n);
  for (auto i = 0; i < pt139_filled_n; ++i) {
    pt139_filled.push_back(v827[i % v827.size()]);
  }
  auto pt139 = cc->MakeCKKSPackedPlaintext(pt139_filled);
  const auto& ct282 = cc->EvalMult(ct22, pt139);
  std::vector<float> v828(std::begin(v28) + 140 * 512, std::begin(v28) + 140 * 512 + 1024);
  std::vector<float> v829(896);
  std::copy(v828.begin() + 0, v828.begin() + 0 + 896, v829.begin());
  std::vector<float> v830(128);
  std::copy(v828.begin() + 896, v828.begin() + 896 + 128, v830.begin());
  std::copy(v829.begin(), v829.end(), v86.begin() + 128);
  std::copy(v830.begin(), v830.end(), v86.begin() + 0);
  std::vector<double> v833(std::begin(v86), std::end(v86));
  auto pt140_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt140_filled = v833;
  pt140_filled.clear();
  pt140_filled.reserve(pt140_filled_n);
  for (auto i = 0; i < pt140_filled_n; ++i) {
    pt140_filled.push_back(v833[i % v833.size()]);
  }
  auto pt140 = cc->MakeCKKSPackedPlaintext(pt140_filled);
  const auto& ct283 = cc->EvalMult(ct24, pt140);
  std::vector<float> v834(std::begin(v28) + 141 * 512, std::begin(v28) + 141 * 512 + 1024);
  std::vector<float> v835(896);
  std::copy(v834.begin() + 0, v834.begin() + 0 + 896, v835.begin());
  std::vector<float> v836(128);
  std::copy(v834.begin() + 896, v834.begin() + 896 + 128, v836.begin());
  std::copy(v835.begin(), v835.end(), v86.begin() + 128);
  std::copy(v836.begin(), v836.end(), v86.begin() + 0);
  std::vector<double> v839(std::begin(v86), std::end(v86));
  auto pt141_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt141_filled = v839;
  pt141_filled.clear();
  pt141_filled.reserve(pt141_filled_n);
  for (auto i = 0; i < pt141_filled_n; ++i) {
    pt141_filled.push_back(v839[i % v839.size()]);
  }
  auto pt141 = cc->MakeCKKSPackedPlaintext(pt141_filled);
  const auto& ct284 = cc->EvalMult(ct26, pt141);
  std::vector<float> v840(std::begin(v28) + 142 * 512, std::begin(v28) + 142 * 512 + 1024);
  std::vector<float> v841(896);
  std::copy(v840.begin() + 0, v840.begin() + 0 + 896, v841.begin());
  std::vector<float> v842(128);
  std::copy(v840.begin() + 896, v840.begin() + 896 + 128, v842.begin());
  std::copy(v841.begin(), v841.end(), v86.begin() + 128);
  std::copy(v842.begin(), v842.end(), v86.begin() + 0);
  std::vector<double> v845(std::begin(v86), std::end(v86));
  auto pt142_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt142_filled = v845;
  pt142_filled.clear();
  pt142_filled.reserve(pt142_filled_n);
  for (auto i = 0; i < pt142_filled_n; ++i) {
    pt142_filled.push_back(v845[i % v845.size()]);
  }
  auto pt142 = cc->MakeCKKSPackedPlaintext(pt142_filled);
  const auto& ct285 = cc->EvalMult(ct28, pt142);
  std::vector<float> v846(std::begin(v28) + 143 * 512, std::begin(v28) + 143 * 512 + 1024);
  std::vector<float> v847(896);
  std::copy(v846.begin() + 0, v846.begin() + 0 + 896, v847.begin());
  std::vector<float> v848(128);
  std::copy(v846.begin() + 896, v846.begin() + 896 + 128, v848.begin());
  std::copy(v847.begin(), v847.end(), v86.begin() + 128);
  std::copy(v848.begin(), v848.end(), v86.begin() + 0);
  std::vector<double> v851(std::begin(v86), std::end(v86));
  auto pt143_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt143_filled = v851;
  pt143_filled.clear();
  pt143_filled.reserve(pt143_filled_n);
  for (auto i = 0; i < pt143_filled_n; ++i) {
    pt143_filled.push_back(v851[i % v851.size()]);
  }
  auto pt143 = cc->MakeCKKSPackedPlaintext(pt143_filled);
  const auto& ct286 = cc->EvalMult(ct30, pt143);
  const auto& ct287 = cc->EvalAdd(ct271, ct272);
  const auto& ct288 = cc->EvalAdd(ct273, ct274);
  const auto& ct289 = cc->EvalAdd(ct287, ct288);
  const auto& ct290 = cc->EvalAdd(ct275, ct276);
  const auto& ct291 = cc->EvalAdd(ct277, ct278);
  const auto& ct292 = cc->EvalAdd(ct290, ct291);
  const auto& ct293 = cc->EvalAdd(ct289, ct292);
  const auto& ct294 = cc->EvalAdd(ct279, ct280);
  const auto& ct295 = cc->EvalAdd(ct281, ct282);
  const auto& ct296 = cc->EvalAdd(ct294, ct295);
  const auto& ct297 = cc->EvalAdd(ct283, ct284);
  const auto& ct298 = cc->EvalAdd(ct285, ct286);
  const auto& ct299 = cc->EvalAdd(ct297, ct298);
  const auto& ct300 = cc->EvalAdd(ct296, ct299);
  const auto& ct301 = cc->EvalAdd(ct293, ct300);
  const auto& ct302 = cc->EvalRotate(ct301, 128);
  std::vector<float> v852(std::begin(v28) + 144 * 512, std::begin(v28) + 144 * 512 + 1024);
  std::vector<float> v853(880);
  std::copy(v852.begin() + 0, v852.begin() + 0 + 880, v853.begin());
  std::vector<float> v854(144);
  std::copy(v852.begin() + 880, v852.begin() + 880 + 144, v854.begin());
  std::copy(v853.begin(), v853.end(), v86.begin() + 144);
  std::copy(v854.begin(), v854.end(), v86.begin() + 0);
  std::vector<double> v857(std::begin(v86), std::end(v86));
  auto pt144_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt144_filled = v857;
  pt144_filled.clear();
  pt144_filled.reserve(pt144_filled_n);
  for (auto i = 0; i < pt144_filled_n; ++i) {
    pt144_filled.push_back(v857[i % v857.size()]);
  }
  auto pt144 = cc->MakeCKKSPackedPlaintext(pt144_filled);
  const auto& ct303 = cc->EvalMult(ct, pt144);
  std::vector<float> v858(std::begin(v28) + 145 * 512, std::begin(v28) + 145 * 512 + 1024);
  std::vector<float> v859(880);
  std::copy(v858.begin() + 0, v858.begin() + 0 + 880, v859.begin());
  std::vector<float> v860(144);
  std::copy(v858.begin() + 880, v858.begin() + 880 + 144, v860.begin());
  std::copy(v859.begin(), v859.end(), v86.begin() + 144);
  std::copy(v860.begin(), v860.end(), v86.begin() + 0);
  std::vector<double> v863(std::begin(v86), std::end(v86));
  auto pt145_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt145_filled = v863;
  pt145_filled.clear();
  pt145_filled.reserve(pt145_filled_n);
  for (auto i = 0; i < pt145_filled_n; ++i) {
    pt145_filled.push_back(v863[i % v863.size()]);
  }
  auto pt145 = cc->MakeCKKSPackedPlaintext(pt145_filled);
  const auto& ct304 = cc->EvalMult(ct2, pt145);
  std::vector<float> v864(std::begin(v28) + 146 * 512, std::begin(v28) + 146 * 512 + 1024);
  std::vector<float> v865(880);
  std::copy(v864.begin() + 0, v864.begin() + 0 + 880, v865.begin());
  std::vector<float> v866(144);
  std::copy(v864.begin() + 880, v864.begin() + 880 + 144, v866.begin());
  std::copy(v865.begin(), v865.end(), v86.begin() + 144);
  std::copy(v866.begin(), v866.end(), v86.begin() + 0);
  std::vector<double> v869(std::begin(v86), std::end(v86));
  auto pt146_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt146_filled = v869;
  pt146_filled.clear();
  pt146_filled.reserve(pt146_filled_n);
  for (auto i = 0; i < pt146_filled_n; ++i) {
    pt146_filled.push_back(v869[i % v869.size()]);
  }
  auto pt146 = cc->MakeCKKSPackedPlaintext(pt146_filled);
  const auto& ct305 = cc->EvalMult(ct4, pt146);
  std::vector<float> v870(std::begin(v28) + 147 * 512, std::begin(v28) + 147 * 512 + 1024);
  std::vector<float> v871(880);
  std::copy(v870.begin() + 0, v870.begin() + 0 + 880, v871.begin());
  std::vector<float> v872(144);
  std::copy(v870.begin() + 880, v870.begin() + 880 + 144, v872.begin());
  std::copy(v871.begin(), v871.end(), v86.begin() + 144);
  std::copy(v872.begin(), v872.end(), v86.begin() + 0);
  std::vector<double> v875(std::begin(v86), std::end(v86));
  auto pt147_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt147_filled = v875;
  pt147_filled.clear();
  pt147_filled.reserve(pt147_filled_n);
  for (auto i = 0; i < pt147_filled_n; ++i) {
    pt147_filled.push_back(v875[i % v875.size()]);
  }
  auto pt147 = cc->MakeCKKSPackedPlaintext(pt147_filled);
  const auto& ct306 = cc->EvalMult(ct6, pt147);
  std::vector<float> v876(std::begin(v28) + 148 * 512, std::begin(v28) + 148 * 512 + 1024);
  std::vector<float> v877(880);
  std::copy(v876.begin() + 0, v876.begin() + 0 + 880, v877.begin());
  std::vector<float> v878(144);
  std::copy(v876.begin() + 880, v876.begin() + 880 + 144, v878.begin());
  std::copy(v877.begin(), v877.end(), v86.begin() + 144);
  std::copy(v878.begin(), v878.end(), v86.begin() + 0);
  std::vector<double> v881(std::begin(v86), std::end(v86));
  auto pt148_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt148_filled = v881;
  pt148_filled.clear();
  pt148_filled.reserve(pt148_filled_n);
  for (auto i = 0; i < pt148_filled_n; ++i) {
    pt148_filled.push_back(v881[i % v881.size()]);
  }
  auto pt148 = cc->MakeCKKSPackedPlaintext(pt148_filled);
  const auto& ct307 = cc->EvalMult(ct8, pt148);
  std::vector<float> v882(std::begin(v28) + 149 * 512, std::begin(v28) + 149 * 512 + 1024);
  std::vector<float> v883(880);
  std::copy(v882.begin() + 0, v882.begin() + 0 + 880, v883.begin());
  std::vector<float> v884(144);
  std::copy(v882.begin() + 880, v882.begin() + 880 + 144, v884.begin());
  std::copy(v883.begin(), v883.end(), v86.begin() + 144);
  std::copy(v884.begin(), v884.end(), v86.begin() + 0);
  std::vector<double> v887(std::begin(v86), std::end(v86));
  auto pt149_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt149_filled = v887;
  pt149_filled.clear();
  pt149_filled.reserve(pt149_filled_n);
  for (auto i = 0; i < pt149_filled_n; ++i) {
    pt149_filled.push_back(v887[i % v887.size()]);
  }
  auto pt149 = cc->MakeCKKSPackedPlaintext(pt149_filled);
  const auto& ct308 = cc->EvalMult(ct10, pt149);
  std::vector<float> v888(std::begin(v28) + 150 * 512, std::begin(v28) + 150 * 512 + 1024);
  std::vector<float> v889(880);
  std::copy(v888.begin() + 0, v888.begin() + 0 + 880, v889.begin());
  std::vector<float> v890(144);
  std::copy(v888.begin() + 880, v888.begin() + 880 + 144, v890.begin());
  std::copy(v889.begin(), v889.end(), v86.begin() + 144);
  std::copy(v890.begin(), v890.end(), v86.begin() + 0);
  std::vector<double> v893(std::begin(v86), std::end(v86));
  auto pt150_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt150_filled = v893;
  pt150_filled.clear();
  pt150_filled.reserve(pt150_filled_n);
  for (auto i = 0; i < pt150_filled_n; ++i) {
    pt150_filled.push_back(v893[i % v893.size()]);
  }
  auto pt150 = cc->MakeCKKSPackedPlaintext(pt150_filled);
  const auto& ct309 = cc->EvalMult(ct12, pt150);
  std::vector<float> v894(std::begin(v28) + 151 * 512, std::begin(v28) + 151 * 512 + 1024);
  std::vector<float> v895(880);
  std::copy(v894.begin() + 0, v894.begin() + 0 + 880, v895.begin());
  std::vector<float> v896(144);
  std::copy(v894.begin() + 880, v894.begin() + 880 + 144, v896.begin());
  std::copy(v895.begin(), v895.end(), v86.begin() + 144);
  std::copy(v896.begin(), v896.end(), v86.begin() + 0);
  std::vector<double> v899(std::begin(v86), std::end(v86));
  auto pt151_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt151_filled = v899;
  pt151_filled.clear();
  pt151_filled.reserve(pt151_filled_n);
  for (auto i = 0; i < pt151_filled_n; ++i) {
    pt151_filled.push_back(v899[i % v899.size()]);
  }
  auto pt151 = cc->MakeCKKSPackedPlaintext(pt151_filled);
  const auto& ct310 = cc->EvalMult(ct14, pt151);
  std::vector<float> v900(std::begin(v28) + 152 * 512, std::begin(v28) + 152 * 512 + 1024);
  std::vector<float> v901(880);
  std::copy(v900.begin() + 0, v900.begin() + 0 + 880, v901.begin());
  std::vector<float> v902(144);
  std::copy(v900.begin() + 880, v900.begin() + 880 + 144, v902.begin());
  std::copy(v901.begin(), v901.end(), v86.begin() + 144);
  std::copy(v902.begin(), v902.end(), v86.begin() + 0);
  std::vector<double> v905(std::begin(v86), std::end(v86));
  auto pt152_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt152_filled = v905;
  pt152_filled.clear();
  pt152_filled.reserve(pt152_filled_n);
  for (auto i = 0; i < pt152_filled_n; ++i) {
    pt152_filled.push_back(v905[i % v905.size()]);
  }
  auto pt152 = cc->MakeCKKSPackedPlaintext(pt152_filled);
  const auto& ct311 = cc->EvalMult(ct16, pt152);
  std::vector<float> v906(std::begin(v28) + 153 * 512, std::begin(v28) + 153 * 512 + 1024);
  std::vector<float> v907(880);
  std::copy(v906.begin() + 0, v906.begin() + 0 + 880, v907.begin());
  std::vector<float> v908(144);
  std::copy(v906.begin() + 880, v906.begin() + 880 + 144, v908.begin());
  std::copy(v907.begin(), v907.end(), v86.begin() + 144);
  std::copy(v908.begin(), v908.end(), v86.begin() + 0);
  std::vector<double> v911(std::begin(v86), std::end(v86));
  auto pt153_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt153_filled = v911;
  pt153_filled.clear();
  pt153_filled.reserve(pt153_filled_n);
  for (auto i = 0; i < pt153_filled_n; ++i) {
    pt153_filled.push_back(v911[i % v911.size()]);
  }
  auto pt153 = cc->MakeCKKSPackedPlaintext(pt153_filled);
  const auto& ct312 = cc->EvalMult(ct18, pt153);
  std::vector<float> v912(std::begin(v28) + 154 * 512, std::begin(v28) + 154 * 512 + 1024);
  std::vector<float> v913(880);
  std::copy(v912.begin() + 0, v912.begin() + 0 + 880, v913.begin());
  std::vector<float> v914(144);
  std::copy(v912.begin() + 880, v912.begin() + 880 + 144, v914.begin());
  std::copy(v913.begin(), v913.end(), v86.begin() + 144);
  std::copy(v914.begin(), v914.end(), v86.begin() + 0);
  std::vector<double> v917(std::begin(v86), std::end(v86));
  auto pt154_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt154_filled = v917;
  pt154_filled.clear();
  pt154_filled.reserve(pt154_filled_n);
  for (auto i = 0; i < pt154_filled_n; ++i) {
    pt154_filled.push_back(v917[i % v917.size()]);
  }
  auto pt154 = cc->MakeCKKSPackedPlaintext(pt154_filled);
  const auto& ct313 = cc->EvalMult(ct20, pt154);
  std::vector<float> v918(std::begin(v28) + 155 * 512, std::begin(v28) + 155 * 512 + 1024);
  std::vector<float> v919(880);
  std::copy(v918.begin() + 0, v918.begin() + 0 + 880, v919.begin());
  std::vector<float> v920(144);
  std::copy(v918.begin() + 880, v918.begin() + 880 + 144, v920.begin());
  std::copy(v919.begin(), v919.end(), v86.begin() + 144);
  std::copy(v920.begin(), v920.end(), v86.begin() + 0);
  std::vector<double> v923(std::begin(v86), std::end(v86));
  auto pt155_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt155_filled = v923;
  pt155_filled.clear();
  pt155_filled.reserve(pt155_filled_n);
  for (auto i = 0; i < pt155_filled_n; ++i) {
    pt155_filled.push_back(v923[i % v923.size()]);
  }
  auto pt155 = cc->MakeCKKSPackedPlaintext(pt155_filled);
  const auto& ct314 = cc->EvalMult(ct22, pt155);
  std::vector<float> v924(std::begin(v28) + 156 * 512, std::begin(v28) + 156 * 512 + 1024);
  std::vector<float> v925(880);
  std::copy(v924.begin() + 0, v924.begin() + 0 + 880, v925.begin());
  std::vector<float> v926(144);
  std::copy(v924.begin() + 880, v924.begin() + 880 + 144, v926.begin());
  std::copy(v925.begin(), v925.end(), v86.begin() + 144);
  std::copy(v926.begin(), v926.end(), v86.begin() + 0);
  std::vector<double> v929(std::begin(v86), std::end(v86));
  auto pt156_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt156_filled = v929;
  pt156_filled.clear();
  pt156_filled.reserve(pt156_filled_n);
  for (auto i = 0; i < pt156_filled_n; ++i) {
    pt156_filled.push_back(v929[i % v929.size()]);
  }
  auto pt156 = cc->MakeCKKSPackedPlaintext(pt156_filled);
  const auto& ct315 = cc->EvalMult(ct24, pt156);
  std::vector<float> v930(std::begin(v28) + 157 * 512, std::begin(v28) + 157 * 512 + 1024);
  std::vector<float> v931(880);
  std::copy(v930.begin() + 0, v930.begin() + 0 + 880, v931.begin());
  std::vector<float> v932(144);
  std::copy(v930.begin() + 880, v930.begin() + 880 + 144, v932.begin());
  std::copy(v931.begin(), v931.end(), v86.begin() + 144);
  std::copy(v932.begin(), v932.end(), v86.begin() + 0);
  std::vector<double> v935(std::begin(v86), std::end(v86));
  auto pt157_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt157_filled = v935;
  pt157_filled.clear();
  pt157_filled.reserve(pt157_filled_n);
  for (auto i = 0; i < pt157_filled_n; ++i) {
    pt157_filled.push_back(v935[i % v935.size()]);
  }
  auto pt157 = cc->MakeCKKSPackedPlaintext(pt157_filled);
  const auto& ct316 = cc->EvalMult(ct26, pt157);
  std::vector<float> v936(std::begin(v28) + 158 * 512, std::begin(v28) + 158 * 512 + 1024);
  std::vector<float> v937(880);
  std::copy(v936.begin() + 0, v936.begin() + 0 + 880, v937.begin());
  std::vector<float> v938(144);
  std::copy(v936.begin() + 880, v936.begin() + 880 + 144, v938.begin());
  std::copy(v937.begin(), v937.end(), v86.begin() + 144);
  std::copy(v938.begin(), v938.end(), v86.begin() + 0);
  std::vector<double> v941(std::begin(v86), std::end(v86));
  auto pt158_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt158_filled = v941;
  pt158_filled.clear();
  pt158_filled.reserve(pt158_filled_n);
  for (auto i = 0; i < pt158_filled_n; ++i) {
    pt158_filled.push_back(v941[i % v941.size()]);
  }
  auto pt158 = cc->MakeCKKSPackedPlaintext(pt158_filled);
  const auto& ct317 = cc->EvalMult(ct28, pt158);
  std::vector<float> v942(std::begin(v28) + 159 * 512, std::begin(v28) + 159 * 512 + 1024);
  std::vector<float> v943(880);
  std::copy(v942.begin() + 0, v942.begin() + 0 + 880, v943.begin());
  std::vector<float> v944(144);
  std::copy(v942.begin() + 880, v942.begin() + 880 + 144, v944.begin());
  std::copy(v943.begin(), v943.end(), v86.begin() + 144);
  std::copy(v944.begin(), v944.end(), v86.begin() + 0);
  std::vector<double> v947(std::begin(v86), std::end(v86));
  auto pt159_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt159_filled = v947;
  pt159_filled.clear();
  pt159_filled.reserve(pt159_filled_n);
  for (auto i = 0; i < pt159_filled_n; ++i) {
    pt159_filled.push_back(v947[i % v947.size()]);
  }
  auto pt159 = cc->MakeCKKSPackedPlaintext(pt159_filled);
  const auto& ct318 = cc->EvalMult(ct30, pt159);
  const auto& ct319 = cc->EvalAdd(ct303, ct304);
  const auto& ct320 = cc->EvalAdd(ct305, ct306);
  const auto& ct321 = cc->EvalAdd(ct319, ct320);
  const auto& ct322 = cc->EvalAdd(ct307, ct308);
  const auto& ct323 = cc->EvalAdd(ct309, ct310);
  const auto& ct324 = cc->EvalAdd(ct322, ct323);
  const auto& ct325 = cc->EvalAdd(ct321, ct324);
  const auto& ct326 = cc->EvalAdd(ct311, ct312);
  const auto& ct327 = cc->EvalAdd(ct313, ct314);
  const auto& ct328 = cc->EvalAdd(ct326, ct327);
  const auto& ct329 = cc->EvalAdd(ct315, ct316);
  const auto& ct330 = cc->EvalAdd(ct317, ct318);
  const auto& ct331 = cc->EvalAdd(ct329, ct330);
  const auto& ct332 = cc->EvalAdd(ct328, ct331);
  const auto& ct333 = cc->EvalAdd(ct325, ct332);
  const auto& ct334 = cc->EvalRotate(ct333, 144);
  std::vector<float> v948(std::begin(v28) + 160 * 512, std::begin(v28) + 160 * 512 + 1024);
  std::vector<float> v949(864);
  std::copy(v948.begin() + 0, v948.begin() + 0 + 864, v949.begin());
  std::vector<float> v950(160);
  std::copy(v948.begin() + 864, v948.begin() + 864 + 160, v950.begin());
  std::copy(v949.begin(), v949.end(), v86.begin() + 160);
  std::copy(v950.begin(), v950.end(), v86.begin() + 0);
  std::vector<double> v953(std::begin(v86), std::end(v86));
  auto pt160_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt160_filled = v953;
  pt160_filled.clear();
  pt160_filled.reserve(pt160_filled_n);
  for (auto i = 0; i < pt160_filled_n; ++i) {
    pt160_filled.push_back(v953[i % v953.size()]);
  }
  auto pt160 = cc->MakeCKKSPackedPlaintext(pt160_filled);
  const auto& ct335 = cc->EvalMult(ct, pt160);
  std::vector<float> v954(std::begin(v28) + 161 * 512, std::begin(v28) + 161 * 512 + 1024);
  std::vector<float> v955(864);
  std::copy(v954.begin() + 0, v954.begin() + 0 + 864, v955.begin());
  std::vector<float> v956(160);
  std::copy(v954.begin() + 864, v954.begin() + 864 + 160, v956.begin());
  std::copy(v955.begin(), v955.end(), v86.begin() + 160);
  std::copy(v956.begin(), v956.end(), v86.begin() + 0);
  std::vector<double> v959(std::begin(v86), std::end(v86));
  auto pt161_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt161_filled = v959;
  pt161_filled.clear();
  pt161_filled.reserve(pt161_filled_n);
  for (auto i = 0; i < pt161_filled_n; ++i) {
    pt161_filled.push_back(v959[i % v959.size()]);
  }
  auto pt161 = cc->MakeCKKSPackedPlaintext(pt161_filled);
  const auto& ct336 = cc->EvalMult(ct2, pt161);
  std::vector<float> v960(std::begin(v28) + 162 * 512, std::begin(v28) + 162 * 512 + 1024);
  std::vector<float> v961(864);
  std::copy(v960.begin() + 0, v960.begin() + 0 + 864, v961.begin());
  std::vector<float> v962(160);
  std::copy(v960.begin() + 864, v960.begin() + 864 + 160, v962.begin());
  std::copy(v961.begin(), v961.end(), v86.begin() + 160);
  std::copy(v962.begin(), v962.end(), v86.begin() + 0);
  std::vector<double> v965(std::begin(v86), std::end(v86));
  auto pt162_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt162_filled = v965;
  pt162_filled.clear();
  pt162_filled.reserve(pt162_filled_n);
  for (auto i = 0; i < pt162_filled_n; ++i) {
    pt162_filled.push_back(v965[i % v965.size()]);
  }
  auto pt162 = cc->MakeCKKSPackedPlaintext(pt162_filled);
  const auto& ct337 = cc->EvalMult(ct4, pt162);
  std::vector<float> v966(std::begin(v28) + 163 * 512, std::begin(v28) + 163 * 512 + 1024);
  std::vector<float> v967(864);
  std::copy(v966.begin() + 0, v966.begin() + 0 + 864, v967.begin());
  std::vector<float> v968(160);
  std::copy(v966.begin() + 864, v966.begin() + 864 + 160, v968.begin());
  std::copy(v967.begin(), v967.end(), v86.begin() + 160);
  std::copy(v968.begin(), v968.end(), v86.begin() + 0);
  std::vector<double> v971(std::begin(v86), std::end(v86));
  auto pt163_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt163_filled = v971;
  pt163_filled.clear();
  pt163_filled.reserve(pt163_filled_n);
  for (auto i = 0; i < pt163_filled_n; ++i) {
    pt163_filled.push_back(v971[i % v971.size()]);
  }
  auto pt163 = cc->MakeCKKSPackedPlaintext(pt163_filled);
  const auto& ct338 = cc->EvalMult(ct6, pt163);
  std::vector<float> v972(std::begin(v28) + 164 * 512, std::begin(v28) + 164 * 512 + 1024);
  std::vector<float> v973(864);
  std::copy(v972.begin() + 0, v972.begin() + 0 + 864, v973.begin());
  std::vector<float> v974(160);
  std::copy(v972.begin() + 864, v972.begin() + 864 + 160, v974.begin());
  std::copy(v973.begin(), v973.end(), v86.begin() + 160);
  std::copy(v974.begin(), v974.end(), v86.begin() + 0);
  std::vector<double> v977(std::begin(v86), std::end(v86));
  auto pt164_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt164_filled = v977;
  pt164_filled.clear();
  pt164_filled.reserve(pt164_filled_n);
  for (auto i = 0; i < pt164_filled_n; ++i) {
    pt164_filled.push_back(v977[i % v977.size()]);
  }
  auto pt164 = cc->MakeCKKSPackedPlaintext(pt164_filled);
  const auto& ct339 = cc->EvalMult(ct8, pt164);
  std::vector<float> v978(std::begin(v28) + 165 * 512, std::begin(v28) + 165 * 512 + 1024);
  std::vector<float> v979(864);
  std::copy(v978.begin() + 0, v978.begin() + 0 + 864, v979.begin());
  std::vector<float> v980(160);
  std::copy(v978.begin() + 864, v978.begin() + 864 + 160, v980.begin());
  std::copy(v979.begin(), v979.end(), v86.begin() + 160);
  std::copy(v980.begin(), v980.end(), v86.begin() + 0);
  std::vector<double> v983(std::begin(v86), std::end(v86));
  auto pt165_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt165_filled = v983;
  pt165_filled.clear();
  pt165_filled.reserve(pt165_filled_n);
  for (auto i = 0; i < pt165_filled_n; ++i) {
    pt165_filled.push_back(v983[i % v983.size()]);
  }
  auto pt165 = cc->MakeCKKSPackedPlaintext(pt165_filled);
  const auto& ct340 = cc->EvalMult(ct10, pt165);
  std::vector<float> v984(std::begin(v28) + 166 * 512, std::begin(v28) + 166 * 512 + 1024);
  std::vector<float> v985(864);
  std::copy(v984.begin() + 0, v984.begin() + 0 + 864, v985.begin());
  std::vector<float> v986(160);
  std::copy(v984.begin() + 864, v984.begin() + 864 + 160, v986.begin());
  std::copy(v985.begin(), v985.end(), v86.begin() + 160);
  std::copy(v986.begin(), v986.end(), v86.begin() + 0);
  std::vector<double> v989(std::begin(v86), std::end(v86));
  auto pt166_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt166_filled = v989;
  pt166_filled.clear();
  pt166_filled.reserve(pt166_filled_n);
  for (auto i = 0; i < pt166_filled_n; ++i) {
    pt166_filled.push_back(v989[i % v989.size()]);
  }
  auto pt166 = cc->MakeCKKSPackedPlaintext(pt166_filled);
  const auto& ct341 = cc->EvalMult(ct12, pt166);
  std::vector<float> v990(std::begin(v28) + 167 * 512, std::begin(v28) + 167 * 512 + 1024);
  std::vector<float> v991(864);
  std::copy(v990.begin() + 0, v990.begin() + 0 + 864, v991.begin());
  std::vector<float> v992(160);
  std::copy(v990.begin() + 864, v990.begin() + 864 + 160, v992.begin());
  std::copy(v991.begin(), v991.end(), v86.begin() + 160);
  std::copy(v992.begin(), v992.end(), v86.begin() + 0);
  std::vector<double> v995(std::begin(v86), std::end(v86));
  auto pt167_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt167_filled = v995;
  pt167_filled.clear();
  pt167_filled.reserve(pt167_filled_n);
  for (auto i = 0; i < pt167_filled_n; ++i) {
    pt167_filled.push_back(v995[i % v995.size()]);
  }
  auto pt167 = cc->MakeCKKSPackedPlaintext(pt167_filled);
  const auto& ct342 = cc->EvalMult(ct14, pt167);
  std::vector<float> v996(std::begin(v28) + 168 * 512, std::begin(v28) + 168 * 512 + 1024);
  std::vector<float> v997(864);
  std::copy(v996.begin() + 0, v996.begin() + 0 + 864, v997.begin());
  std::vector<float> v998(160);
  std::copy(v996.begin() + 864, v996.begin() + 864 + 160, v998.begin());
  std::copy(v997.begin(), v997.end(), v86.begin() + 160);
  std::copy(v998.begin(), v998.end(), v86.begin() + 0);
  std::vector<double> v1001(std::begin(v86), std::end(v86));
  auto pt168_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt168_filled = v1001;
  pt168_filled.clear();
  pt168_filled.reserve(pt168_filled_n);
  for (auto i = 0; i < pt168_filled_n; ++i) {
    pt168_filled.push_back(v1001[i % v1001.size()]);
  }
  auto pt168 = cc->MakeCKKSPackedPlaintext(pt168_filled);
  const auto& ct343 = cc->EvalMult(ct16, pt168);
  std::vector<float> v1002(std::begin(v28) + 169 * 512, std::begin(v28) + 169 * 512 + 1024);
  std::vector<float> v1003(864);
  std::copy(v1002.begin() + 0, v1002.begin() + 0 + 864, v1003.begin());
  std::vector<float> v1004(160);
  std::copy(v1002.begin() + 864, v1002.begin() + 864 + 160, v1004.begin());
  std::copy(v1003.begin(), v1003.end(), v86.begin() + 160);
  std::copy(v1004.begin(), v1004.end(), v86.begin() + 0);
  std::vector<double> v1007(std::begin(v86), std::end(v86));
  auto pt169_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt169_filled = v1007;
  pt169_filled.clear();
  pt169_filled.reserve(pt169_filled_n);
  for (auto i = 0; i < pt169_filled_n; ++i) {
    pt169_filled.push_back(v1007[i % v1007.size()]);
  }
  auto pt169 = cc->MakeCKKSPackedPlaintext(pt169_filled);
  const auto& ct344 = cc->EvalMult(ct18, pt169);
  std::vector<float> v1008(std::begin(v28) + 170 * 512, std::begin(v28) + 170 * 512 + 1024);
  std::vector<float> v1009(864);
  std::copy(v1008.begin() + 0, v1008.begin() + 0 + 864, v1009.begin());
  std::vector<float> v1010(160);
  std::copy(v1008.begin() + 864, v1008.begin() + 864 + 160, v1010.begin());
  std::copy(v1009.begin(), v1009.end(), v86.begin() + 160);
  std::copy(v1010.begin(), v1010.end(), v86.begin() + 0);
  std::vector<double> v1013(std::begin(v86), std::end(v86));
  auto pt170_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt170_filled = v1013;
  pt170_filled.clear();
  pt170_filled.reserve(pt170_filled_n);
  for (auto i = 0; i < pt170_filled_n; ++i) {
    pt170_filled.push_back(v1013[i % v1013.size()]);
  }
  auto pt170 = cc->MakeCKKSPackedPlaintext(pt170_filled);
  const auto& ct345 = cc->EvalMult(ct20, pt170);
  std::vector<float> v1014(std::begin(v28) + 171 * 512, std::begin(v28) + 171 * 512 + 1024);
  std::vector<float> v1015(864);
  std::copy(v1014.begin() + 0, v1014.begin() + 0 + 864, v1015.begin());
  std::vector<float> v1016(160);
  std::copy(v1014.begin() + 864, v1014.begin() + 864 + 160, v1016.begin());
  std::copy(v1015.begin(), v1015.end(), v86.begin() + 160);
  std::copy(v1016.begin(), v1016.end(), v86.begin() + 0);
  std::vector<double> v1019(std::begin(v86), std::end(v86));
  auto pt171_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt171_filled = v1019;
  pt171_filled.clear();
  pt171_filled.reserve(pt171_filled_n);
  for (auto i = 0; i < pt171_filled_n; ++i) {
    pt171_filled.push_back(v1019[i % v1019.size()]);
  }
  auto pt171 = cc->MakeCKKSPackedPlaintext(pt171_filled);
  const auto& ct346 = cc->EvalMult(ct22, pt171);
  std::vector<float> v1020(std::begin(v28) + 172 * 512, std::begin(v28) + 172 * 512 + 1024);
  std::vector<float> v1021(864);
  std::copy(v1020.begin() + 0, v1020.begin() + 0 + 864, v1021.begin());
  std::vector<float> v1022(160);
  std::copy(v1020.begin() + 864, v1020.begin() + 864 + 160, v1022.begin());
  std::copy(v1021.begin(), v1021.end(), v86.begin() + 160);
  std::copy(v1022.begin(), v1022.end(), v86.begin() + 0);
  std::vector<double> v1025(std::begin(v86), std::end(v86));
  auto pt172_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt172_filled = v1025;
  pt172_filled.clear();
  pt172_filled.reserve(pt172_filled_n);
  for (auto i = 0; i < pt172_filled_n; ++i) {
    pt172_filled.push_back(v1025[i % v1025.size()]);
  }
  auto pt172 = cc->MakeCKKSPackedPlaintext(pt172_filled);
  const auto& ct347 = cc->EvalMult(ct24, pt172);
  std::vector<float> v1026(std::begin(v28) + 173 * 512, std::begin(v28) + 173 * 512 + 1024);
  std::vector<float> v1027(864);
  std::copy(v1026.begin() + 0, v1026.begin() + 0 + 864, v1027.begin());
  std::vector<float> v1028(160);
  std::copy(v1026.begin() + 864, v1026.begin() + 864 + 160, v1028.begin());
  std::copy(v1027.begin(), v1027.end(), v86.begin() + 160);
  std::copy(v1028.begin(), v1028.end(), v86.begin() + 0);
  std::vector<double> v1031(std::begin(v86), std::end(v86));
  auto pt173_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt173_filled = v1031;
  pt173_filled.clear();
  pt173_filled.reserve(pt173_filled_n);
  for (auto i = 0; i < pt173_filled_n; ++i) {
    pt173_filled.push_back(v1031[i % v1031.size()]);
  }
  auto pt173 = cc->MakeCKKSPackedPlaintext(pt173_filled);
  const auto& ct348 = cc->EvalMult(ct26, pt173);
  std::vector<float> v1032(std::begin(v28) + 174 * 512, std::begin(v28) + 174 * 512 + 1024);
  std::vector<float> v1033(864);
  std::copy(v1032.begin() + 0, v1032.begin() + 0 + 864, v1033.begin());
  std::vector<float> v1034(160);
  std::copy(v1032.begin() + 864, v1032.begin() + 864 + 160, v1034.begin());
  std::copy(v1033.begin(), v1033.end(), v86.begin() + 160);
  std::copy(v1034.begin(), v1034.end(), v86.begin() + 0);
  std::vector<double> v1037(std::begin(v86), std::end(v86));
  auto pt174_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt174_filled = v1037;
  pt174_filled.clear();
  pt174_filled.reserve(pt174_filled_n);
  for (auto i = 0; i < pt174_filled_n; ++i) {
    pt174_filled.push_back(v1037[i % v1037.size()]);
  }
  auto pt174 = cc->MakeCKKSPackedPlaintext(pt174_filled);
  const auto& ct349 = cc->EvalMult(ct28, pt174);
  std::vector<float> v1038(std::begin(v28) + 175 * 512, std::begin(v28) + 175 * 512 + 1024);
  std::vector<float> v1039(864);
  std::copy(v1038.begin() + 0, v1038.begin() + 0 + 864, v1039.begin());
  std::vector<float> v1040(160);
  std::copy(v1038.begin() + 864, v1038.begin() + 864 + 160, v1040.begin());
  std::copy(v1039.begin(), v1039.end(), v86.begin() + 160);
  std::copy(v1040.begin(), v1040.end(), v86.begin() + 0);
  std::vector<double> v1043(std::begin(v86), std::end(v86));
  auto pt175_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt175_filled = v1043;
  pt175_filled.clear();
  pt175_filled.reserve(pt175_filled_n);
  for (auto i = 0; i < pt175_filled_n; ++i) {
    pt175_filled.push_back(v1043[i % v1043.size()]);
  }
  auto pt175 = cc->MakeCKKSPackedPlaintext(pt175_filled);
  const auto& ct350 = cc->EvalMult(ct30, pt175);
  const auto& ct351 = cc->EvalAdd(ct335, ct336);
  const auto& ct352 = cc->EvalAdd(ct337, ct338);
  const auto& ct353 = cc->EvalAdd(ct351, ct352);
  const auto& ct354 = cc->EvalAdd(ct339, ct340);
  const auto& ct355 = cc->EvalAdd(ct341, ct342);
  const auto& ct356 = cc->EvalAdd(ct354, ct355);
  const auto& ct357 = cc->EvalAdd(ct353, ct356);
  const auto& ct358 = cc->EvalAdd(ct343, ct344);
  const auto& ct359 = cc->EvalAdd(ct345, ct346);
  const auto& ct360 = cc->EvalAdd(ct358, ct359);
  const auto& ct361 = cc->EvalAdd(ct347, ct348);
  const auto& ct362 = cc->EvalAdd(ct349, ct350);
  const auto& ct363 = cc->EvalAdd(ct361, ct362);
  const auto& ct364 = cc->EvalAdd(ct360, ct363);
  const auto& ct365 = cc->EvalAdd(ct357, ct364);
  const auto& ct366 = cc->EvalRotate(ct365, 160);
  std::vector<float> v1044(std::begin(v28) + 176 * 512, std::begin(v28) + 176 * 512 + 1024);
  std::vector<float> v1045(848);
  std::copy(v1044.begin() + 0, v1044.begin() + 0 + 848, v1045.begin());
  std::vector<float> v1046(176);
  std::copy(v1044.begin() + 848, v1044.begin() + 848 + 176, v1046.begin());
  std::copy(v1045.begin(), v1045.end(), v86.begin() + 176);
  std::copy(v1046.begin(), v1046.end(), v86.begin() + 0);
  std::vector<double> v1049(std::begin(v86), std::end(v86));
  auto pt176_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt176_filled = v1049;
  pt176_filled.clear();
  pt176_filled.reserve(pt176_filled_n);
  for (auto i = 0; i < pt176_filled_n; ++i) {
    pt176_filled.push_back(v1049[i % v1049.size()]);
  }
  auto pt176 = cc->MakeCKKSPackedPlaintext(pt176_filled);
  const auto& ct367 = cc->EvalMult(ct, pt176);
  std::vector<float> v1050(std::begin(v28) + 177 * 512, std::begin(v28) + 177 * 512 + 1024);
  std::vector<float> v1051(848);
  std::copy(v1050.begin() + 0, v1050.begin() + 0 + 848, v1051.begin());
  std::vector<float> v1052(176);
  std::copy(v1050.begin() + 848, v1050.begin() + 848 + 176, v1052.begin());
  std::copy(v1051.begin(), v1051.end(), v86.begin() + 176);
  std::copy(v1052.begin(), v1052.end(), v86.begin() + 0);
  std::vector<double> v1055(std::begin(v86), std::end(v86));
  auto pt177_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt177_filled = v1055;
  pt177_filled.clear();
  pt177_filled.reserve(pt177_filled_n);
  for (auto i = 0; i < pt177_filled_n; ++i) {
    pt177_filled.push_back(v1055[i % v1055.size()]);
  }
  auto pt177 = cc->MakeCKKSPackedPlaintext(pt177_filled);
  const auto& ct368 = cc->EvalMult(ct2, pt177);
  std::vector<float> v1056(std::begin(v28) + 178 * 512, std::begin(v28) + 178 * 512 + 1024);
  std::vector<float> v1057(848);
  std::copy(v1056.begin() + 0, v1056.begin() + 0 + 848, v1057.begin());
  std::vector<float> v1058(176);
  std::copy(v1056.begin() + 848, v1056.begin() + 848 + 176, v1058.begin());
  std::copy(v1057.begin(), v1057.end(), v86.begin() + 176);
  std::copy(v1058.begin(), v1058.end(), v86.begin() + 0);
  std::vector<double> v1061(std::begin(v86), std::end(v86));
  auto pt178_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt178_filled = v1061;
  pt178_filled.clear();
  pt178_filled.reserve(pt178_filled_n);
  for (auto i = 0; i < pt178_filled_n; ++i) {
    pt178_filled.push_back(v1061[i % v1061.size()]);
  }
  auto pt178 = cc->MakeCKKSPackedPlaintext(pt178_filled);
  const auto& ct369 = cc->EvalMult(ct4, pt178);
  std::vector<float> v1062(std::begin(v28) + 179 * 512, std::begin(v28) + 179 * 512 + 1024);
  std::vector<float> v1063(848);
  std::copy(v1062.begin() + 0, v1062.begin() + 0 + 848, v1063.begin());
  std::vector<float> v1064(176);
  std::copy(v1062.begin() + 848, v1062.begin() + 848 + 176, v1064.begin());
  std::copy(v1063.begin(), v1063.end(), v86.begin() + 176);
  std::copy(v1064.begin(), v1064.end(), v86.begin() + 0);
  std::vector<double> v1067(std::begin(v86), std::end(v86));
  auto pt179_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt179_filled = v1067;
  pt179_filled.clear();
  pt179_filled.reserve(pt179_filled_n);
  for (auto i = 0; i < pt179_filled_n; ++i) {
    pt179_filled.push_back(v1067[i % v1067.size()]);
  }
  auto pt179 = cc->MakeCKKSPackedPlaintext(pt179_filled);
  const auto& ct370 = cc->EvalMult(ct6, pt179);
  std::vector<float> v1068(std::begin(v28) + 180 * 512, std::begin(v28) + 180 * 512 + 1024);
  std::vector<float> v1069(848);
  std::copy(v1068.begin() + 0, v1068.begin() + 0 + 848, v1069.begin());
  std::vector<float> v1070(176);
  std::copy(v1068.begin() + 848, v1068.begin() + 848 + 176, v1070.begin());
  std::copy(v1069.begin(), v1069.end(), v86.begin() + 176);
  std::copy(v1070.begin(), v1070.end(), v86.begin() + 0);
  std::vector<double> v1073(std::begin(v86), std::end(v86));
  auto pt180_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt180_filled = v1073;
  pt180_filled.clear();
  pt180_filled.reserve(pt180_filled_n);
  for (auto i = 0; i < pt180_filled_n; ++i) {
    pt180_filled.push_back(v1073[i % v1073.size()]);
  }
  auto pt180 = cc->MakeCKKSPackedPlaintext(pt180_filled);
  const auto& ct371 = cc->EvalMult(ct8, pt180);
  std::vector<float> v1074(std::begin(v28) + 181 * 512, std::begin(v28) + 181 * 512 + 1024);
  std::vector<float> v1075(848);
  std::copy(v1074.begin() + 0, v1074.begin() + 0 + 848, v1075.begin());
  std::vector<float> v1076(176);
  std::copy(v1074.begin() + 848, v1074.begin() + 848 + 176, v1076.begin());
  std::copy(v1075.begin(), v1075.end(), v86.begin() + 176);
  std::copy(v1076.begin(), v1076.end(), v86.begin() + 0);
  std::vector<double> v1079(std::begin(v86), std::end(v86));
  auto pt181_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt181_filled = v1079;
  pt181_filled.clear();
  pt181_filled.reserve(pt181_filled_n);
  for (auto i = 0; i < pt181_filled_n; ++i) {
    pt181_filled.push_back(v1079[i % v1079.size()]);
  }
  auto pt181 = cc->MakeCKKSPackedPlaintext(pt181_filled);
  const auto& ct372 = cc->EvalMult(ct10, pt181);
  std::vector<float> v1080(std::begin(v28) + 182 * 512, std::begin(v28) + 182 * 512 + 1024);
  std::vector<float> v1081(848);
  std::copy(v1080.begin() + 0, v1080.begin() + 0 + 848, v1081.begin());
  std::vector<float> v1082(176);
  std::copy(v1080.begin() + 848, v1080.begin() + 848 + 176, v1082.begin());
  std::copy(v1081.begin(), v1081.end(), v86.begin() + 176);
  std::copy(v1082.begin(), v1082.end(), v86.begin() + 0);
  std::vector<double> v1085(std::begin(v86), std::end(v86));
  auto pt182_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt182_filled = v1085;
  pt182_filled.clear();
  pt182_filled.reserve(pt182_filled_n);
  for (auto i = 0; i < pt182_filled_n; ++i) {
    pt182_filled.push_back(v1085[i % v1085.size()]);
  }
  auto pt182 = cc->MakeCKKSPackedPlaintext(pt182_filled);
  const auto& ct373 = cc->EvalMult(ct12, pt182);
  std::vector<float> v1086(std::begin(v28) + 183 * 512, std::begin(v28) + 183 * 512 + 1024);
  std::vector<float> v1087(848);
  std::copy(v1086.begin() + 0, v1086.begin() + 0 + 848, v1087.begin());
  std::vector<float> v1088(176);
  std::copy(v1086.begin() + 848, v1086.begin() + 848 + 176, v1088.begin());
  std::copy(v1087.begin(), v1087.end(), v86.begin() + 176);
  std::copy(v1088.begin(), v1088.end(), v86.begin() + 0);
  std::vector<double> v1091(std::begin(v86), std::end(v86));
  auto pt183_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt183_filled = v1091;
  pt183_filled.clear();
  pt183_filled.reserve(pt183_filled_n);
  for (auto i = 0; i < pt183_filled_n; ++i) {
    pt183_filled.push_back(v1091[i % v1091.size()]);
  }
  auto pt183 = cc->MakeCKKSPackedPlaintext(pt183_filled);
  const auto& ct374 = cc->EvalMult(ct14, pt183);
  std::vector<float> v1092(std::begin(v28) + 184 * 512, std::begin(v28) + 184 * 512 + 1024);
  std::vector<float> v1093(848);
  std::copy(v1092.begin() + 0, v1092.begin() + 0 + 848, v1093.begin());
  std::vector<float> v1094(176);
  std::copy(v1092.begin() + 848, v1092.begin() + 848 + 176, v1094.begin());
  std::copy(v1093.begin(), v1093.end(), v86.begin() + 176);
  std::copy(v1094.begin(), v1094.end(), v86.begin() + 0);
  std::vector<double> v1097(std::begin(v86), std::end(v86));
  auto pt184_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt184_filled = v1097;
  pt184_filled.clear();
  pt184_filled.reserve(pt184_filled_n);
  for (auto i = 0; i < pt184_filled_n; ++i) {
    pt184_filled.push_back(v1097[i % v1097.size()]);
  }
  auto pt184 = cc->MakeCKKSPackedPlaintext(pt184_filled);
  const auto& ct375 = cc->EvalMult(ct16, pt184);
  std::vector<float> v1098(std::begin(v28) + 185 * 512, std::begin(v28) + 185 * 512 + 1024);
  std::vector<float> v1099(848);
  std::copy(v1098.begin() + 0, v1098.begin() + 0 + 848, v1099.begin());
  std::vector<float> v1100(176);
  std::copy(v1098.begin() + 848, v1098.begin() + 848 + 176, v1100.begin());
  std::copy(v1099.begin(), v1099.end(), v86.begin() + 176);
  std::copy(v1100.begin(), v1100.end(), v86.begin() + 0);
  std::vector<double> v1103(std::begin(v86), std::end(v86));
  auto pt185_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt185_filled = v1103;
  pt185_filled.clear();
  pt185_filled.reserve(pt185_filled_n);
  for (auto i = 0; i < pt185_filled_n; ++i) {
    pt185_filled.push_back(v1103[i % v1103.size()]);
  }
  auto pt185 = cc->MakeCKKSPackedPlaintext(pt185_filled);
  const auto& ct376 = cc->EvalMult(ct18, pt185);
  std::vector<float> v1104(std::begin(v28) + 186 * 512, std::begin(v28) + 186 * 512 + 1024);
  std::vector<float> v1105(848);
  std::copy(v1104.begin() + 0, v1104.begin() + 0 + 848, v1105.begin());
  std::vector<float> v1106(176);
  std::copy(v1104.begin() + 848, v1104.begin() + 848 + 176, v1106.begin());
  std::copy(v1105.begin(), v1105.end(), v86.begin() + 176);
  std::copy(v1106.begin(), v1106.end(), v86.begin() + 0);
  std::vector<double> v1109(std::begin(v86), std::end(v86));
  auto pt186_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt186_filled = v1109;
  pt186_filled.clear();
  pt186_filled.reserve(pt186_filled_n);
  for (auto i = 0; i < pt186_filled_n; ++i) {
    pt186_filled.push_back(v1109[i % v1109.size()]);
  }
  auto pt186 = cc->MakeCKKSPackedPlaintext(pt186_filled);
  const auto& ct377 = cc->EvalMult(ct20, pt186);
  std::vector<float> v1110(std::begin(v28) + 187 * 512, std::begin(v28) + 187 * 512 + 1024);
  std::vector<float> v1111(848);
  std::copy(v1110.begin() + 0, v1110.begin() + 0 + 848, v1111.begin());
  std::vector<float> v1112(176);
  std::copy(v1110.begin() + 848, v1110.begin() + 848 + 176, v1112.begin());
  std::copy(v1111.begin(), v1111.end(), v86.begin() + 176);
  std::copy(v1112.begin(), v1112.end(), v86.begin() + 0);
  std::vector<double> v1115(std::begin(v86), std::end(v86));
  auto pt187_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt187_filled = v1115;
  pt187_filled.clear();
  pt187_filled.reserve(pt187_filled_n);
  for (auto i = 0; i < pt187_filled_n; ++i) {
    pt187_filled.push_back(v1115[i % v1115.size()]);
  }
  auto pt187 = cc->MakeCKKSPackedPlaintext(pt187_filled);
  const auto& ct378 = cc->EvalMult(ct22, pt187);
  std::vector<float> v1116(std::begin(v28) + 188 * 512, std::begin(v28) + 188 * 512 + 1024);
  std::vector<float> v1117(848);
  std::copy(v1116.begin() + 0, v1116.begin() + 0 + 848, v1117.begin());
  std::vector<float> v1118(176);
  std::copy(v1116.begin() + 848, v1116.begin() + 848 + 176, v1118.begin());
  std::copy(v1117.begin(), v1117.end(), v86.begin() + 176);
  std::copy(v1118.begin(), v1118.end(), v86.begin() + 0);
  std::vector<double> v1121(std::begin(v86), std::end(v86));
  auto pt188_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt188_filled = v1121;
  pt188_filled.clear();
  pt188_filled.reserve(pt188_filled_n);
  for (auto i = 0; i < pt188_filled_n; ++i) {
    pt188_filled.push_back(v1121[i % v1121.size()]);
  }
  auto pt188 = cc->MakeCKKSPackedPlaintext(pt188_filled);
  const auto& ct379 = cc->EvalMult(ct24, pt188);
  std::vector<float> v1122(std::begin(v28) + 189 * 512, std::begin(v28) + 189 * 512 + 1024);
  std::vector<float> v1123(848);
  std::copy(v1122.begin() + 0, v1122.begin() + 0 + 848, v1123.begin());
  std::vector<float> v1124(176);
  std::copy(v1122.begin() + 848, v1122.begin() + 848 + 176, v1124.begin());
  std::copy(v1123.begin(), v1123.end(), v86.begin() + 176);
  std::copy(v1124.begin(), v1124.end(), v86.begin() + 0);
  std::vector<double> v1127(std::begin(v86), std::end(v86));
  auto pt189_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt189_filled = v1127;
  pt189_filled.clear();
  pt189_filled.reserve(pt189_filled_n);
  for (auto i = 0; i < pt189_filled_n; ++i) {
    pt189_filled.push_back(v1127[i % v1127.size()]);
  }
  auto pt189 = cc->MakeCKKSPackedPlaintext(pt189_filled);
  const auto& ct380 = cc->EvalMult(ct26, pt189);
  std::vector<float> v1128(std::begin(v28) + 190 * 512, std::begin(v28) + 190 * 512 + 1024);
  std::vector<float> v1129(848);
  std::copy(v1128.begin() + 0, v1128.begin() + 0 + 848, v1129.begin());
  std::vector<float> v1130(176);
  std::copy(v1128.begin() + 848, v1128.begin() + 848 + 176, v1130.begin());
  std::copy(v1129.begin(), v1129.end(), v86.begin() + 176);
  std::copy(v1130.begin(), v1130.end(), v86.begin() + 0);
  std::vector<double> v1133(std::begin(v86), std::end(v86));
  auto pt190_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt190_filled = v1133;
  pt190_filled.clear();
  pt190_filled.reserve(pt190_filled_n);
  for (auto i = 0; i < pt190_filled_n; ++i) {
    pt190_filled.push_back(v1133[i % v1133.size()]);
  }
  auto pt190 = cc->MakeCKKSPackedPlaintext(pt190_filled);
  const auto& ct381 = cc->EvalMult(ct28, pt190);
  std::vector<float> v1134(std::begin(v28) + 191 * 512, std::begin(v28) + 191 * 512 + 1024);
  std::vector<float> v1135(848);
  std::copy(v1134.begin() + 0, v1134.begin() + 0 + 848, v1135.begin());
  std::vector<float> v1136(176);
  std::copy(v1134.begin() + 848, v1134.begin() + 848 + 176, v1136.begin());
  std::copy(v1135.begin(), v1135.end(), v86.begin() + 176);
  std::copy(v1136.begin(), v1136.end(), v86.begin() + 0);
  std::vector<double> v1139(std::begin(v86), std::end(v86));
  auto pt191_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt191_filled = v1139;
  pt191_filled.clear();
  pt191_filled.reserve(pt191_filled_n);
  for (auto i = 0; i < pt191_filled_n; ++i) {
    pt191_filled.push_back(v1139[i % v1139.size()]);
  }
  auto pt191 = cc->MakeCKKSPackedPlaintext(pt191_filled);
  const auto& ct382 = cc->EvalMult(ct30, pt191);
  const auto& ct383 = cc->EvalAdd(ct367, ct368);
  const auto& ct384 = cc->EvalAdd(ct369, ct370);
  const auto& ct385 = cc->EvalAdd(ct383, ct384);
  const auto& ct386 = cc->EvalAdd(ct371, ct372);
  const auto& ct387 = cc->EvalAdd(ct373, ct374);
  const auto& ct388 = cc->EvalAdd(ct386, ct387);
  const auto& ct389 = cc->EvalAdd(ct385, ct388);
  const auto& ct390 = cc->EvalAdd(ct375, ct376);
  const auto& ct391 = cc->EvalAdd(ct377, ct378);
  const auto& ct392 = cc->EvalAdd(ct390, ct391);
  const auto& ct393 = cc->EvalAdd(ct379, ct380);
  const auto& ct394 = cc->EvalAdd(ct381, ct382);
  const auto& ct395 = cc->EvalAdd(ct393, ct394);
  const auto& ct396 = cc->EvalAdd(ct392, ct395);
  const auto& ct397 = cc->EvalAdd(ct389, ct396);
  const auto& ct398 = cc->EvalRotate(ct397, 176);
  std::vector<float> v1140(std::begin(v28) + 192 * 512, std::begin(v28) + 192 * 512 + 1024);
  std::vector<float> v1141(832);
  std::copy(v1140.begin() + 0, v1140.begin() + 0 + 832, v1141.begin());
  std::vector<float> v1142(192);
  std::copy(v1140.begin() + 832, v1140.begin() + 832 + 192, v1142.begin());
  std::copy(v1141.begin(), v1141.end(), v86.begin() + 192);
  std::copy(v1142.begin(), v1142.end(), v86.begin() + 0);
  std::vector<double> v1145(std::begin(v86), std::end(v86));
  auto pt192_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt192_filled = v1145;
  pt192_filled.clear();
  pt192_filled.reserve(pt192_filled_n);
  for (auto i = 0; i < pt192_filled_n; ++i) {
    pt192_filled.push_back(v1145[i % v1145.size()]);
  }
  auto pt192 = cc->MakeCKKSPackedPlaintext(pt192_filled);
  const auto& ct399 = cc->EvalMult(ct, pt192);
  std::vector<float> v1146(std::begin(v28) + 193 * 512, std::begin(v28) + 193 * 512 + 1024);
  std::vector<float> v1147(832);
  std::copy(v1146.begin() + 0, v1146.begin() + 0 + 832, v1147.begin());
  std::vector<float> v1148(192);
  std::copy(v1146.begin() + 832, v1146.begin() + 832 + 192, v1148.begin());
  std::copy(v1147.begin(), v1147.end(), v86.begin() + 192);
  std::copy(v1148.begin(), v1148.end(), v86.begin() + 0);
  std::vector<double> v1151(std::begin(v86), std::end(v86));
  auto pt193_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt193_filled = v1151;
  pt193_filled.clear();
  pt193_filled.reserve(pt193_filled_n);
  for (auto i = 0; i < pt193_filled_n; ++i) {
    pt193_filled.push_back(v1151[i % v1151.size()]);
  }
  auto pt193 = cc->MakeCKKSPackedPlaintext(pt193_filled);
  const auto& ct400 = cc->EvalMult(ct2, pt193);
  std::vector<float> v1152(std::begin(v28) + 194 * 512, std::begin(v28) + 194 * 512 + 1024);
  std::vector<float> v1153(832);
  std::copy(v1152.begin() + 0, v1152.begin() + 0 + 832, v1153.begin());
  std::vector<float> v1154(192);
  std::copy(v1152.begin() + 832, v1152.begin() + 832 + 192, v1154.begin());
  std::copy(v1153.begin(), v1153.end(), v86.begin() + 192);
  std::copy(v1154.begin(), v1154.end(), v86.begin() + 0);
  std::vector<double> v1157(std::begin(v86), std::end(v86));
  auto pt194_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt194_filled = v1157;
  pt194_filled.clear();
  pt194_filled.reserve(pt194_filled_n);
  for (auto i = 0; i < pt194_filled_n; ++i) {
    pt194_filled.push_back(v1157[i % v1157.size()]);
  }
  auto pt194 = cc->MakeCKKSPackedPlaintext(pt194_filled);
  const auto& ct401 = cc->EvalMult(ct4, pt194);
  std::vector<float> v1158(std::begin(v28) + 195 * 512, std::begin(v28) + 195 * 512 + 1024);
  std::vector<float> v1159(832);
  std::copy(v1158.begin() + 0, v1158.begin() + 0 + 832, v1159.begin());
  std::vector<float> v1160(192);
  std::copy(v1158.begin() + 832, v1158.begin() + 832 + 192, v1160.begin());
  std::copy(v1159.begin(), v1159.end(), v86.begin() + 192);
  std::copy(v1160.begin(), v1160.end(), v86.begin() + 0);
  std::vector<double> v1163(std::begin(v86), std::end(v86));
  auto pt195_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt195_filled = v1163;
  pt195_filled.clear();
  pt195_filled.reserve(pt195_filled_n);
  for (auto i = 0; i < pt195_filled_n; ++i) {
    pt195_filled.push_back(v1163[i % v1163.size()]);
  }
  auto pt195 = cc->MakeCKKSPackedPlaintext(pt195_filled);
  const auto& ct402 = cc->EvalMult(ct6, pt195);
  std::vector<float> v1164(std::begin(v28) + 196 * 512, std::begin(v28) + 196 * 512 + 1024);
  std::vector<float> v1165(832);
  std::copy(v1164.begin() + 0, v1164.begin() + 0 + 832, v1165.begin());
  std::vector<float> v1166(192);
  std::copy(v1164.begin() + 832, v1164.begin() + 832 + 192, v1166.begin());
  std::copy(v1165.begin(), v1165.end(), v86.begin() + 192);
  std::copy(v1166.begin(), v1166.end(), v86.begin() + 0);
  std::vector<double> v1169(std::begin(v86), std::end(v86));
  auto pt196_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt196_filled = v1169;
  pt196_filled.clear();
  pt196_filled.reserve(pt196_filled_n);
  for (auto i = 0; i < pt196_filled_n; ++i) {
    pt196_filled.push_back(v1169[i % v1169.size()]);
  }
  auto pt196 = cc->MakeCKKSPackedPlaintext(pt196_filled);
  const auto& ct403 = cc->EvalMult(ct8, pt196);
  std::vector<float> v1170(std::begin(v28) + 197 * 512, std::begin(v28) + 197 * 512 + 1024);
  std::vector<float> v1171(832);
  std::copy(v1170.begin() + 0, v1170.begin() + 0 + 832, v1171.begin());
  std::vector<float> v1172(192);
  std::copy(v1170.begin() + 832, v1170.begin() + 832 + 192, v1172.begin());
  std::copy(v1171.begin(), v1171.end(), v86.begin() + 192);
  std::copy(v1172.begin(), v1172.end(), v86.begin() + 0);
  std::vector<double> v1175(std::begin(v86), std::end(v86));
  auto pt197_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt197_filled = v1175;
  pt197_filled.clear();
  pt197_filled.reserve(pt197_filled_n);
  for (auto i = 0; i < pt197_filled_n; ++i) {
    pt197_filled.push_back(v1175[i % v1175.size()]);
  }
  auto pt197 = cc->MakeCKKSPackedPlaintext(pt197_filled);
  const auto& ct404 = cc->EvalMult(ct10, pt197);
  std::vector<float> v1176(std::begin(v28) + 198 * 512, std::begin(v28) + 198 * 512 + 1024);
  std::vector<float> v1177(832);
  std::copy(v1176.begin() + 0, v1176.begin() + 0 + 832, v1177.begin());
  std::vector<float> v1178(192);
  std::copy(v1176.begin() + 832, v1176.begin() + 832 + 192, v1178.begin());
  std::copy(v1177.begin(), v1177.end(), v86.begin() + 192);
  std::copy(v1178.begin(), v1178.end(), v86.begin() + 0);
  std::vector<double> v1181(std::begin(v86), std::end(v86));
  auto pt198_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt198_filled = v1181;
  pt198_filled.clear();
  pt198_filled.reserve(pt198_filled_n);
  for (auto i = 0; i < pt198_filled_n; ++i) {
    pt198_filled.push_back(v1181[i % v1181.size()]);
  }
  auto pt198 = cc->MakeCKKSPackedPlaintext(pt198_filled);
  const auto& ct405 = cc->EvalMult(ct12, pt198);
  std::vector<float> v1182(std::begin(v28) + 199 * 512, std::begin(v28) + 199 * 512 + 1024);
  std::vector<float> v1183(832);
  std::copy(v1182.begin() + 0, v1182.begin() + 0 + 832, v1183.begin());
  std::vector<float> v1184(192);
  std::copy(v1182.begin() + 832, v1182.begin() + 832 + 192, v1184.begin());
  std::copy(v1183.begin(), v1183.end(), v86.begin() + 192);
  std::copy(v1184.begin(), v1184.end(), v86.begin() + 0);
  std::vector<double> v1187(std::begin(v86), std::end(v86));
  auto pt199_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt199_filled = v1187;
  pt199_filled.clear();
  pt199_filled.reserve(pt199_filled_n);
  for (auto i = 0; i < pt199_filled_n; ++i) {
    pt199_filled.push_back(v1187[i % v1187.size()]);
  }
  auto pt199 = cc->MakeCKKSPackedPlaintext(pt199_filled);
  const auto& ct406 = cc->EvalMult(ct14, pt199);
  std::vector<float> v1188(std::begin(v28) + 200 * 512, std::begin(v28) + 200 * 512 + 1024);
  std::vector<float> v1189(832);
  std::copy(v1188.begin() + 0, v1188.begin() + 0 + 832, v1189.begin());
  std::vector<float> v1190(192);
  std::copy(v1188.begin() + 832, v1188.begin() + 832 + 192, v1190.begin());
  std::copy(v1189.begin(), v1189.end(), v86.begin() + 192);
  std::copy(v1190.begin(), v1190.end(), v86.begin() + 0);
  std::vector<double> v1193(std::begin(v86), std::end(v86));
  auto pt200_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt200_filled = v1193;
  pt200_filled.clear();
  pt200_filled.reserve(pt200_filled_n);
  for (auto i = 0; i < pt200_filled_n; ++i) {
    pt200_filled.push_back(v1193[i % v1193.size()]);
  }
  auto pt200 = cc->MakeCKKSPackedPlaintext(pt200_filled);
  const auto& ct407 = cc->EvalMult(ct16, pt200);
  std::vector<float> v1194(std::begin(v28) + 201 * 512, std::begin(v28) + 201 * 512 + 1024);
  std::vector<float> v1195(832);
  std::copy(v1194.begin() + 0, v1194.begin() + 0 + 832, v1195.begin());
  std::vector<float> v1196(192);
  std::copy(v1194.begin() + 832, v1194.begin() + 832 + 192, v1196.begin());
  std::copy(v1195.begin(), v1195.end(), v86.begin() + 192);
  std::copy(v1196.begin(), v1196.end(), v86.begin() + 0);
  std::vector<double> v1199(std::begin(v86), std::end(v86));
  auto pt201_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt201_filled = v1199;
  pt201_filled.clear();
  pt201_filled.reserve(pt201_filled_n);
  for (auto i = 0; i < pt201_filled_n; ++i) {
    pt201_filled.push_back(v1199[i % v1199.size()]);
  }
  auto pt201 = cc->MakeCKKSPackedPlaintext(pt201_filled);
  const auto& ct408 = cc->EvalMult(ct18, pt201);
  std::vector<float> v1200(std::begin(v28) + 202 * 512, std::begin(v28) + 202 * 512 + 1024);
  std::vector<float> v1201(832);
  std::copy(v1200.begin() + 0, v1200.begin() + 0 + 832, v1201.begin());
  std::vector<float> v1202(192);
  std::copy(v1200.begin() + 832, v1200.begin() + 832 + 192, v1202.begin());
  std::copy(v1201.begin(), v1201.end(), v86.begin() + 192);
  std::copy(v1202.begin(), v1202.end(), v86.begin() + 0);
  std::vector<double> v1205(std::begin(v86), std::end(v86));
  auto pt202_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt202_filled = v1205;
  pt202_filled.clear();
  pt202_filled.reserve(pt202_filled_n);
  for (auto i = 0; i < pt202_filled_n; ++i) {
    pt202_filled.push_back(v1205[i % v1205.size()]);
  }
  auto pt202 = cc->MakeCKKSPackedPlaintext(pt202_filled);
  const auto& ct409 = cc->EvalMult(ct20, pt202);
  std::vector<float> v1206(std::begin(v28) + 203 * 512, std::begin(v28) + 203 * 512 + 1024);
  std::vector<float> v1207(832);
  std::copy(v1206.begin() + 0, v1206.begin() + 0 + 832, v1207.begin());
  std::vector<float> v1208(192);
  std::copy(v1206.begin() + 832, v1206.begin() + 832 + 192, v1208.begin());
  std::copy(v1207.begin(), v1207.end(), v86.begin() + 192);
  std::copy(v1208.begin(), v1208.end(), v86.begin() + 0);
  std::vector<double> v1211(std::begin(v86), std::end(v86));
  auto pt203_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt203_filled = v1211;
  pt203_filled.clear();
  pt203_filled.reserve(pt203_filled_n);
  for (auto i = 0; i < pt203_filled_n; ++i) {
    pt203_filled.push_back(v1211[i % v1211.size()]);
  }
  auto pt203 = cc->MakeCKKSPackedPlaintext(pt203_filled);
  const auto& ct410 = cc->EvalMult(ct22, pt203);
  std::vector<float> v1212(std::begin(v28) + 204 * 512, std::begin(v28) + 204 * 512 + 1024);
  std::vector<float> v1213(832);
  std::copy(v1212.begin() + 0, v1212.begin() + 0 + 832, v1213.begin());
  std::vector<float> v1214(192);
  std::copy(v1212.begin() + 832, v1212.begin() + 832 + 192, v1214.begin());
  std::copy(v1213.begin(), v1213.end(), v86.begin() + 192);
  std::copy(v1214.begin(), v1214.end(), v86.begin() + 0);
  std::vector<double> v1217(std::begin(v86), std::end(v86));
  auto pt204_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt204_filled = v1217;
  pt204_filled.clear();
  pt204_filled.reserve(pt204_filled_n);
  for (auto i = 0; i < pt204_filled_n; ++i) {
    pt204_filled.push_back(v1217[i % v1217.size()]);
  }
  auto pt204 = cc->MakeCKKSPackedPlaintext(pt204_filled);
  const auto& ct411 = cc->EvalMult(ct24, pt204);
  std::vector<float> v1218(std::begin(v28) + 205 * 512, std::begin(v28) + 205 * 512 + 1024);
  std::vector<float> v1219(832);
  std::copy(v1218.begin() + 0, v1218.begin() + 0 + 832, v1219.begin());
  std::vector<float> v1220(192);
  std::copy(v1218.begin() + 832, v1218.begin() + 832 + 192, v1220.begin());
  std::copy(v1219.begin(), v1219.end(), v86.begin() + 192);
  std::copy(v1220.begin(), v1220.end(), v86.begin() + 0);
  std::vector<double> v1223(std::begin(v86), std::end(v86));
  auto pt205_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt205_filled = v1223;
  pt205_filled.clear();
  pt205_filled.reserve(pt205_filled_n);
  for (auto i = 0; i < pt205_filled_n; ++i) {
    pt205_filled.push_back(v1223[i % v1223.size()]);
  }
  auto pt205 = cc->MakeCKKSPackedPlaintext(pt205_filled);
  const auto& ct412 = cc->EvalMult(ct26, pt205);
  std::vector<float> v1224(std::begin(v28) + 206 * 512, std::begin(v28) + 206 * 512 + 1024);
  std::vector<float> v1225(832);
  std::copy(v1224.begin() + 0, v1224.begin() + 0 + 832, v1225.begin());
  std::vector<float> v1226(192);
  std::copy(v1224.begin() + 832, v1224.begin() + 832 + 192, v1226.begin());
  std::copy(v1225.begin(), v1225.end(), v86.begin() + 192);
  std::copy(v1226.begin(), v1226.end(), v86.begin() + 0);
  std::vector<double> v1229(std::begin(v86), std::end(v86));
  auto pt206_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt206_filled = v1229;
  pt206_filled.clear();
  pt206_filled.reserve(pt206_filled_n);
  for (auto i = 0; i < pt206_filled_n; ++i) {
    pt206_filled.push_back(v1229[i % v1229.size()]);
  }
  auto pt206 = cc->MakeCKKSPackedPlaintext(pt206_filled);
  const auto& ct413 = cc->EvalMult(ct28, pt206);
  std::vector<float> v1230(std::begin(v28) + 207 * 512, std::begin(v28) + 207 * 512 + 1024);
  std::vector<float> v1231(832);
  std::copy(v1230.begin() + 0, v1230.begin() + 0 + 832, v1231.begin());
  std::vector<float> v1232(192);
  std::copy(v1230.begin() + 832, v1230.begin() + 832 + 192, v1232.begin());
  std::copy(v1231.begin(), v1231.end(), v86.begin() + 192);
  std::copy(v1232.begin(), v1232.end(), v86.begin() + 0);
  std::vector<double> v1235(std::begin(v86), std::end(v86));
  auto pt207_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt207_filled = v1235;
  pt207_filled.clear();
  pt207_filled.reserve(pt207_filled_n);
  for (auto i = 0; i < pt207_filled_n; ++i) {
    pt207_filled.push_back(v1235[i % v1235.size()]);
  }
  auto pt207 = cc->MakeCKKSPackedPlaintext(pt207_filled);
  const auto& ct414 = cc->EvalMult(ct30, pt207);
  const auto& ct415 = cc->EvalAdd(ct399, ct400);
  const auto& ct416 = cc->EvalAdd(ct401, ct402);
  const auto& ct417 = cc->EvalAdd(ct415, ct416);
  const auto& ct418 = cc->EvalAdd(ct403, ct404);
  const auto& ct419 = cc->EvalAdd(ct405, ct406);
  const auto& ct420 = cc->EvalAdd(ct418, ct419);
  const auto& ct421 = cc->EvalAdd(ct417, ct420);
  const auto& ct422 = cc->EvalAdd(ct407, ct408);
  const auto& ct423 = cc->EvalAdd(ct409, ct410);
  const auto& ct424 = cc->EvalAdd(ct422, ct423);
  const auto& ct425 = cc->EvalAdd(ct411, ct412);
  const auto& ct426 = cc->EvalAdd(ct413, ct414);
  const auto& ct427 = cc->EvalAdd(ct425, ct426);
  const auto& ct428 = cc->EvalAdd(ct424, ct427);
  const auto& ct429 = cc->EvalAdd(ct421, ct428);
  const auto& ct430 = cc->EvalRotate(ct429, 192);
  std::vector<float> v1236(std::begin(v28) + 208 * 512, std::begin(v28) + 208 * 512 + 1024);
  std::vector<float> v1237(816);
  std::copy(v1236.begin() + 0, v1236.begin() + 0 + 816, v1237.begin());
  std::vector<float> v1238(208);
  std::copy(v1236.begin() + 816, v1236.begin() + 816 + 208, v1238.begin());
  std::copy(v1237.begin(), v1237.end(), v86.begin() + 208);
  std::copy(v1238.begin(), v1238.end(), v86.begin() + 0);
  std::vector<double> v1241(std::begin(v86), std::end(v86));
  auto pt208_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt208_filled = v1241;
  pt208_filled.clear();
  pt208_filled.reserve(pt208_filled_n);
  for (auto i = 0; i < pt208_filled_n; ++i) {
    pt208_filled.push_back(v1241[i % v1241.size()]);
  }
  auto pt208 = cc->MakeCKKSPackedPlaintext(pt208_filled);
  const auto& ct431 = cc->EvalMult(ct, pt208);
  std::vector<float> v1242(std::begin(v28) + 209 * 512, std::begin(v28) + 209 * 512 + 1024);
  std::vector<float> v1243(816);
  std::copy(v1242.begin() + 0, v1242.begin() + 0 + 816, v1243.begin());
  std::vector<float> v1244(208);
  std::copy(v1242.begin() + 816, v1242.begin() + 816 + 208, v1244.begin());
  std::copy(v1243.begin(), v1243.end(), v86.begin() + 208);
  std::copy(v1244.begin(), v1244.end(), v86.begin() + 0);
  std::vector<double> v1247(std::begin(v86), std::end(v86));
  auto pt209_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt209_filled = v1247;
  pt209_filled.clear();
  pt209_filled.reserve(pt209_filled_n);
  for (auto i = 0; i < pt209_filled_n; ++i) {
    pt209_filled.push_back(v1247[i % v1247.size()]);
  }
  auto pt209 = cc->MakeCKKSPackedPlaintext(pt209_filled);
  const auto& ct432 = cc->EvalMult(ct2, pt209);
  std::vector<float> v1248(std::begin(v28) + 210 * 512, std::begin(v28) + 210 * 512 + 1024);
  std::vector<float> v1249(816);
  std::copy(v1248.begin() + 0, v1248.begin() + 0 + 816, v1249.begin());
  std::vector<float> v1250(208);
  std::copy(v1248.begin() + 816, v1248.begin() + 816 + 208, v1250.begin());
  std::copy(v1249.begin(), v1249.end(), v86.begin() + 208);
  std::copy(v1250.begin(), v1250.end(), v86.begin() + 0);
  std::vector<double> v1253(std::begin(v86), std::end(v86));
  auto pt210_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt210_filled = v1253;
  pt210_filled.clear();
  pt210_filled.reserve(pt210_filled_n);
  for (auto i = 0; i < pt210_filled_n; ++i) {
    pt210_filled.push_back(v1253[i % v1253.size()]);
  }
  auto pt210 = cc->MakeCKKSPackedPlaintext(pt210_filled);
  const auto& ct433 = cc->EvalMult(ct4, pt210);
  std::vector<float> v1254(std::begin(v28) + 211 * 512, std::begin(v28) + 211 * 512 + 1024);
  std::vector<float> v1255(816);
  std::copy(v1254.begin() + 0, v1254.begin() + 0 + 816, v1255.begin());
  std::vector<float> v1256(208);
  std::copy(v1254.begin() + 816, v1254.begin() + 816 + 208, v1256.begin());
  std::copy(v1255.begin(), v1255.end(), v86.begin() + 208);
  std::copy(v1256.begin(), v1256.end(), v86.begin() + 0);
  std::vector<double> v1259(std::begin(v86), std::end(v86));
  auto pt211_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt211_filled = v1259;
  pt211_filled.clear();
  pt211_filled.reserve(pt211_filled_n);
  for (auto i = 0; i < pt211_filled_n; ++i) {
    pt211_filled.push_back(v1259[i % v1259.size()]);
  }
  auto pt211 = cc->MakeCKKSPackedPlaintext(pt211_filled);
  const auto& ct434 = cc->EvalMult(ct6, pt211);
  std::vector<float> v1260(std::begin(v28) + 212 * 512, std::begin(v28) + 212 * 512 + 1024);
  std::vector<float> v1261(816);
  std::copy(v1260.begin() + 0, v1260.begin() + 0 + 816, v1261.begin());
  std::vector<float> v1262(208);
  std::copy(v1260.begin() + 816, v1260.begin() + 816 + 208, v1262.begin());
  std::copy(v1261.begin(), v1261.end(), v86.begin() + 208);
  std::copy(v1262.begin(), v1262.end(), v86.begin() + 0);
  std::vector<double> v1265(std::begin(v86), std::end(v86));
  auto pt212_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt212_filled = v1265;
  pt212_filled.clear();
  pt212_filled.reserve(pt212_filled_n);
  for (auto i = 0; i < pt212_filled_n; ++i) {
    pt212_filled.push_back(v1265[i % v1265.size()]);
  }
  auto pt212 = cc->MakeCKKSPackedPlaintext(pt212_filled);
  const auto& ct435 = cc->EvalMult(ct8, pt212);
  std::vector<float> v1266(std::begin(v28) + 213 * 512, std::begin(v28) + 213 * 512 + 1024);
  std::vector<float> v1267(816);
  std::copy(v1266.begin() + 0, v1266.begin() + 0 + 816, v1267.begin());
  std::vector<float> v1268(208);
  std::copy(v1266.begin() + 816, v1266.begin() + 816 + 208, v1268.begin());
  std::copy(v1267.begin(), v1267.end(), v86.begin() + 208);
  std::copy(v1268.begin(), v1268.end(), v86.begin() + 0);
  std::vector<double> v1271(std::begin(v86), std::end(v86));
  auto pt213_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt213_filled = v1271;
  pt213_filled.clear();
  pt213_filled.reserve(pt213_filled_n);
  for (auto i = 0; i < pt213_filled_n; ++i) {
    pt213_filled.push_back(v1271[i % v1271.size()]);
  }
  auto pt213 = cc->MakeCKKSPackedPlaintext(pt213_filled);
  const auto& ct436 = cc->EvalMult(ct10, pt213);
  std::vector<float> v1272(std::begin(v28) + 214 * 512, std::begin(v28) + 214 * 512 + 1024);
  std::vector<float> v1273(816);
  std::copy(v1272.begin() + 0, v1272.begin() + 0 + 816, v1273.begin());
  std::vector<float> v1274(208);
  std::copy(v1272.begin() + 816, v1272.begin() + 816 + 208, v1274.begin());
  std::copy(v1273.begin(), v1273.end(), v86.begin() + 208);
  std::copy(v1274.begin(), v1274.end(), v86.begin() + 0);
  std::vector<double> v1277(std::begin(v86), std::end(v86));
  auto pt214_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt214_filled = v1277;
  pt214_filled.clear();
  pt214_filled.reserve(pt214_filled_n);
  for (auto i = 0; i < pt214_filled_n; ++i) {
    pt214_filled.push_back(v1277[i % v1277.size()]);
  }
  auto pt214 = cc->MakeCKKSPackedPlaintext(pt214_filled);
  const auto& ct437 = cc->EvalMult(ct12, pt214);
  std::vector<float> v1278(std::begin(v28) + 215 * 512, std::begin(v28) + 215 * 512 + 1024);
  std::vector<float> v1279(816);
  std::copy(v1278.begin() + 0, v1278.begin() + 0 + 816, v1279.begin());
  std::vector<float> v1280(208);
  std::copy(v1278.begin() + 816, v1278.begin() + 816 + 208, v1280.begin());
  std::copy(v1279.begin(), v1279.end(), v86.begin() + 208);
  std::copy(v1280.begin(), v1280.end(), v86.begin() + 0);
  std::vector<double> v1283(std::begin(v86), std::end(v86));
  auto pt215_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt215_filled = v1283;
  pt215_filled.clear();
  pt215_filled.reserve(pt215_filled_n);
  for (auto i = 0; i < pt215_filled_n; ++i) {
    pt215_filled.push_back(v1283[i % v1283.size()]);
  }
  auto pt215 = cc->MakeCKKSPackedPlaintext(pt215_filled);
  const auto& ct438 = cc->EvalMult(ct14, pt215);
  std::vector<float> v1284(std::begin(v28) + 216 * 512, std::begin(v28) + 216 * 512 + 1024);
  std::vector<float> v1285(816);
  std::copy(v1284.begin() + 0, v1284.begin() + 0 + 816, v1285.begin());
  std::vector<float> v1286(208);
  std::copy(v1284.begin() + 816, v1284.begin() + 816 + 208, v1286.begin());
  std::copy(v1285.begin(), v1285.end(), v86.begin() + 208);
  std::copy(v1286.begin(), v1286.end(), v86.begin() + 0);
  std::vector<double> v1289(std::begin(v86), std::end(v86));
  auto pt216_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt216_filled = v1289;
  pt216_filled.clear();
  pt216_filled.reserve(pt216_filled_n);
  for (auto i = 0; i < pt216_filled_n; ++i) {
    pt216_filled.push_back(v1289[i % v1289.size()]);
  }
  auto pt216 = cc->MakeCKKSPackedPlaintext(pt216_filled);
  const auto& ct439 = cc->EvalMult(ct16, pt216);
  std::vector<float> v1290(std::begin(v28) + 217 * 512, std::begin(v28) + 217 * 512 + 1024);
  std::vector<float> v1291(816);
  std::copy(v1290.begin() + 0, v1290.begin() + 0 + 816, v1291.begin());
  std::vector<float> v1292(208);
  std::copy(v1290.begin() + 816, v1290.begin() + 816 + 208, v1292.begin());
  std::copy(v1291.begin(), v1291.end(), v86.begin() + 208);
  std::copy(v1292.begin(), v1292.end(), v86.begin() + 0);
  std::vector<double> v1295(std::begin(v86), std::end(v86));
  auto pt217_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt217_filled = v1295;
  pt217_filled.clear();
  pt217_filled.reserve(pt217_filled_n);
  for (auto i = 0; i < pt217_filled_n; ++i) {
    pt217_filled.push_back(v1295[i % v1295.size()]);
  }
  auto pt217 = cc->MakeCKKSPackedPlaintext(pt217_filled);
  const auto& ct440 = cc->EvalMult(ct18, pt217);
  std::vector<float> v1296(std::begin(v28) + 218 * 512, std::begin(v28) + 218 * 512 + 1024);
  std::vector<float> v1297(816);
  std::copy(v1296.begin() + 0, v1296.begin() + 0 + 816, v1297.begin());
  std::vector<float> v1298(208);
  std::copy(v1296.begin() + 816, v1296.begin() + 816 + 208, v1298.begin());
  std::copy(v1297.begin(), v1297.end(), v86.begin() + 208);
  std::copy(v1298.begin(), v1298.end(), v86.begin() + 0);
  std::vector<double> v1301(std::begin(v86), std::end(v86));
  auto pt218_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt218_filled = v1301;
  pt218_filled.clear();
  pt218_filled.reserve(pt218_filled_n);
  for (auto i = 0; i < pt218_filled_n; ++i) {
    pt218_filled.push_back(v1301[i % v1301.size()]);
  }
  auto pt218 = cc->MakeCKKSPackedPlaintext(pt218_filled);
  const auto& ct441 = cc->EvalMult(ct20, pt218);
  std::vector<float> v1302(std::begin(v28) + 219 * 512, std::begin(v28) + 219 * 512 + 1024);
  std::vector<float> v1303(816);
  std::copy(v1302.begin() + 0, v1302.begin() + 0 + 816, v1303.begin());
  std::vector<float> v1304(208);
  std::copy(v1302.begin() + 816, v1302.begin() + 816 + 208, v1304.begin());
  std::copy(v1303.begin(), v1303.end(), v86.begin() + 208);
  std::copy(v1304.begin(), v1304.end(), v86.begin() + 0);
  std::vector<double> v1307(std::begin(v86), std::end(v86));
  auto pt219_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt219_filled = v1307;
  pt219_filled.clear();
  pt219_filled.reserve(pt219_filled_n);
  for (auto i = 0; i < pt219_filled_n; ++i) {
    pt219_filled.push_back(v1307[i % v1307.size()]);
  }
  auto pt219 = cc->MakeCKKSPackedPlaintext(pt219_filled);
  const auto& ct442 = cc->EvalMult(ct22, pt219);
  std::vector<float> v1308(std::begin(v28) + 220 * 512, std::begin(v28) + 220 * 512 + 1024);
  std::vector<float> v1309(816);
  std::copy(v1308.begin() + 0, v1308.begin() + 0 + 816, v1309.begin());
  std::vector<float> v1310(208);
  std::copy(v1308.begin() + 816, v1308.begin() + 816 + 208, v1310.begin());
  std::copy(v1309.begin(), v1309.end(), v86.begin() + 208);
  std::copy(v1310.begin(), v1310.end(), v86.begin() + 0);
  std::vector<double> v1313(std::begin(v86), std::end(v86));
  auto pt220_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt220_filled = v1313;
  pt220_filled.clear();
  pt220_filled.reserve(pt220_filled_n);
  for (auto i = 0; i < pt220_filled_n; ++i) {
    pt220_filled.push_back(v1313[i % v1313.size()]);
  }
  auto pt220 = cc->MakeCKKSPackedPlaintext(pt220_filled);
  const auto& ct443 = cc->EvalMult(ct24, pt220);
  std::vector<float> v1314(std::begin(v28) + 221 * 512, std::begin(v28) + 221 * 512 + 1024);
  std::vector<float> v1315(816);
  std::copy(v1314.begin() + 0, v1314.begin() + 0 + 816, v1315.begin());
  std::vector<float> v1316(208);
  std::copy(v1314.begin() + 816, v1314.begin() + 816 + 208, v1316.begin());
  std::copy(v1315.begin(), v1315.end(), v86.begin() + 208);
  std::copy(v1316.begin(), v1316.end(), v86.begin() + 0);
  std::vector<double> v1319(std::begin(v86), std::end(v86));
  auto pt221_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt221_filled = v1319;
  pt221_filled.clear();
  pt221_filled.reserve(pt221_filled_n);
  for (auto i = 0; i < pt221_filled_n; ++i) {
    pt221_filled.push_back(v1319[i % v1319.size()]);
  }
  auto pt221 = cc->MakeCKKSPackedPlaintext(pt221_filled);
  const auto& ct444 = cc->EvalMult(ct26, pt221);
  std::vector<float> v1320(std::begin(v28) + 222 * 512, std::begin(v28) + 222 * 512 + 1024);
  std::vector<float> v1321(816);
  std::copy(v1320.begin() + 0, v1320.begin() + 0 + 816, v1321.begin());
  std::vector<float> v1322(208);
  std::copy(v1320.begin() + 816, v1320.begin() + 816 + 208, v1322.begin());
  std::copy(v1321.begin(), v1321.end(), v86.begin() + 208);
  std::copy(v1322.begin(), v1322.end(), v86.begin() + 0);
  std::vector<double> v1325(std::begin(v86), std::end(v86));
  auto pt222_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt222_filled = v1325;
  pt222_filled.clear();
  pt222_filled.reserve(pt222_filled_n);
  for (auto i = 0; i < pt222_filled_n; ++i) {
    pt222_filled.push_back(v1325[i % v1325.size()]);
  }
  auto pt222 = cc->MakeCKKSPackedPlaintext(pt222_filled);
  const auto& ct445 = cc->EvalMult(ct28, pt222);
  std::vector<float> v1326(std::begin(v28) + 223 * 512, std::begin(v28) + 223 * 512 + 1024);
  std::vector<float> v1327(816);
  std::copy(v1326.begin() + 0, v1326.begin() + 0 + 816, v1327.begin());
  std::vector<float> v1328(208);
  std::copy(v1326.begin() + 816, v1326.begin() + 816 + 208, v1328.begin());
  std::copy(v1327.begin(), v1327.end(), v86.begin() + 208);
  std::copy(v1328.begin(), v1328.end(), v86.begin() + 0);
  std::vector<double> v1331(std::begin(v86), std::end(v86));
  auto pt223_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt223_filled = v1331;
  pt223_filled.clear();
  pt223_filled.reserve(pt223_filled_n);
  for (auto i = 0; i < pt223_filled_n; ++i) {
    pt223_filled.push_back(v1331[i % v1331.size()]);
  }
  auto pt223 = cc->MakeCKKSPackedPlaintext(pt223_filled);
  const auto& ct446 = cc->EvalMult(ct30, pt223);
  const auto& ct447 = cc->EvalAdd(ct431, ct432);
  const auto& ct448 = cc->EvalAdd(ct433, ct434);
  const auto& ct449 = cc->EvalAdd(ct447, ct448);
  const auto& ct450 = cc->EvalAdd(ct435, ct436);
  const auto& ct451 = cc->EvalAdd(ct437, ct438);
  const auto& ct452 = cc->EvalAdd(ct450, ct451);
  const auto& ct453 = cc->EvalAdd(ct449, ct452);
  const auto& ct454 = cc->EvalAdd(ct439, ct440);
  const auto& ct455 = cc->EvalAdd(ct441, ct442);
  const auto& ct456 = cc->EvalAdd(ct454, ct455);
  const auto& ct457 = cc->EvalAdd(ct443, ct444);
  const auto& ct458 = cc->EvalAdd(ct445, ct446);
  const auto& ct459 = cc->EvalAdd(ct457, ct458);
  const auto& ct460 = cc->EvalAdd(ct456, ct459);
  const auto& ct461 = cc->EvalAdd(ct453, ct460);
  const auto& ct462 = cc->EvalRotate(ct461, 208);
  std::vector<float> v1332(std::begin(v28) + 224 * 512, std::begin(v28) + 224 * 512 + 1024);
  std::vector<float> v1333(800);
  std::copy(v1332.begin() + 0, v1332.begin() + 0 + 800, v1333.begin());
  std::vector<float> v1334(224);
  std::copy(v1332.begin() + 800, v1332.begin() + 800 + 224, v1334.begin());
  std::copy(v1333.begin(), v1333.end(), v86.begin() + 224);
  std::copy(v1334.begin(), v1334.end(), v86.begin() + 0);
  std::vector<double> v1337(std::begin(v86), std::end(v86));
  auto pt224_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt224_filled = v1337;
  pt224_filled.clear();
  pt224_filled.reserve(pt224_filled_n);
  for (auto i = 0; i < pt224_filled_n; ++i) {
    pt224_filled.push_back(v1337[i % v1337.size()]);
  }
  auto pt224 = cc->MakeCKKSPackedPlaintext(pt224_filled);
  const auto& ct463 = cc->EvalMult(ct, pt224);
  std::vector<float> v1338(std::begin(v28) + 225 * 512, std::begin(v28) + 225 * 512 + 1024);
  std::vector<float> v1339(800);
  std::copy(v1338.begin() + 0, v1338.begin() + 0 + 800, v1339.begin());
  std::vector<float> v1340(224);
  std::copy(v1338.begin() + 800, v1338.begin() + 800 + 224, v1340.begin());
  std::copy(v1339.begin(), v1339.end(), v86.begin() + 224);
  std::copy(v1340.begin(), v1340.end(), v86.begin() + 0);
  std::vector<double> v1343(std::begin(v86), std::end(v86));
  auto pt225_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt225_filled = v1343;
  pt225_filled.clear();
  pt225_filled.reserve(pt225_filled_n);
  for (auto i = 0; i < pt225_filled_n; ++i) {
    pt225_filled.push_back(v1343[i % v1343.size()]);
  }
  auto pt225 = cc->MakeCKKSPackedPlaintext(pt225_filled);
  const auto& ct464 = cc->EvalMult(ct2, pt225);
  std::vector<float> v1344(std::begin(v28) + 226 * 512, std::begin(v28) + 226 * 512 + 1024);
  std::vector<float> v1345(800);
  std::copy(v1344.begin() + 0, v1344.begin() + 0 + 800, v1345.begin());
  std::vector<float> v1346(224);
  std::copy(v1344.begin() + 800, v1344.begin() + 800 + 224, v1346.begin());
  std::copy(v1345.begin(), v1345.end(), v86.begin() + 224);
  std::copy(v1346.begin(), v1346.end(), v86.begin() + 0);
  std::vector<double> v1349(std::begin(v86), std::end(v86));
  auto pt226_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt226_filled = v1349;
  pt226_filled.clear();
  pt226_filled.reserve(pt226_filled_n);
  for (auto i = 0; i < pt226_filled_n; ++i) {
    pt226_filled.push_back(v1349[i % v1349.size()]);
  }
  auto pt226 = cc->MakeCKKSPackedPlaintext(pt226_filled);
  const auto& ct465 = cc->EvalMult(ct4, pt226);
  std::vector<float> v1350(std::begin(v28) + 227 * 512, std::begin(v28) + 227 * 512 + 1024);
  std::vector<float> v1351(800);
  std::copy(v1350.begin() + 0, v1350.begin() + 0 + 800, v1351.begin());
  std::vector<float> v1352(224);
  std::copy(v1350.begin() + 800, v1350.begin() + 800 + 224, v1352.begin());
  std::copy(v1351.begin(), v1351.end(), v86.begin() + 224);
  std::copy(v1352.begin(), v1352.end(), v86.begin() + 0);
  std::vector<double> v1355(std::begin(v86), std::end(v86));
  auto pt227_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt227_filled = v1355;
  pt227_filled.clear();
  pt227_filled.reserve(pt227_filled_n);
  for (auto i = 0; i < pt227_filled_n; ++i) {
    pt227_filled.push_back(v1355[i % v1355.size()]);
  }
  auto pt227 = cc->MakeCKKSPackedPlaintext(pt227_filled);
  const auto& ct466 = cc->EvalMult(ct6, pt227);
  std::vector<float> v1356(std::begin(v28) + 228 * 512, std::begin(v28) + 228 * 512 + 1024);
  std::vector<float> v1357(800);
  std::copy(v1356.begin() + 0, v1356.begin() + 0 + 800, v1357.begin());
  std::vector<float> v1358(224);
  std::copy(v1356.begin() + 800, v1356.begin() + 800 + 224, v1358.begin());
  std::copy(v1357.begin(), v1357.end(), v86.begin() + 224);
  std::copy(v1358.begin(), v1358.end(), v86.begin() + 0);
  std::vector<double> v1361(std::begin(v86), std::end(v86));
  auto pt228_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt228_filled = v1361;
  pt228_filled.clear();
  pt228_filled.reserve(pt228_filled_n);
  for (auto i = 0; i < pt228_filled_n; ++i) {
    pt228_filled.push_back(v1361[i % v1361.size()]);
  }
  auto pt228 = cc->MakeCKKSPackedPlaintext(pt228_filled);
  const auto& ct467 = cc->EvalMult(ct8, pt228);
  std::vector<float> v1362(std::begin(v28) + 229 * 512, std::begin(v28) + 229 * 512 + 1024);
  std::vector<float> v1363(800);
  std::copy(v1362.begin() + 0, v1362.begin() + 0 + 800, v1363.begin());
  std::vector<float> v1364(224);
  std::copy(v1362.begin() + 800, v1362.begin() + 800 + 224, v1364.begin());
  std::copy(v1363.begin(), v1363.end(), v86.begin() + 224);
  std::copy(v1364.begin(), v1364.end(), v86.begin() + 0);
  std::vector<double> v1367(std::begin(v86), std::end(v86));
  auto pt229_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt229_filled = v1367;
  pt229_filled.clear();
  pt229_filled.reserve(pt229_filled_n);
  for (auto i = 0; i < pt229_filled_n; ++i) {
    pt229_filled.push_back(v1367[i % v1367.size()]);
  }
  auto pt229 = cc->MakeCKKSPackedPlaintext(pt229_filled);
  const auto& ct468 = cc->EvalMult(ct10, pt229);
  std::vector<float> v1368(std::begin(v28) + 230 * 512, std::begin(v28) + 230 * 512 + 1024);
  std::vector<float> v1369(800);
  std::copy(v1368.begin() + 0, v1368.begin() + 0 + 800, v1369.begin());
  std::vector<float> v1370(224);
  std::copy(v1368.begin() + 800, v1368.begin() + 800 + 224, v1370.begin());
  std::copy(v1369.begin(), v1369.end(), v86.begin() + 224);
  std::copy(v1370.begin(), v1370.end(), v86.begin() + 0);
  std::vector<double> v1373(std::begin(v86), std::end(v86));
  auto pt230_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt230_filled = v1373;
  pt230_filled.clear();
  pt230_filled.reserve(pt230_filled_n);
  for (auto i = 0; i < pt230_filled_n; ++i) {
    pt230_filled.push_back(v1373[i % v1373.size()]);
  }
  auto pt230 = cc->MakeCKKSPackedPlaintext(pt230_filled);
  const auto& ct469 = cc->EvalMult(ct12, pt230);
  std::vector<float> v1374(std::begin(v28) + 231 * 512, std::begin(v28) + 231 * 512 + 1024);
  std::vector<float> v1375(800);
  std::copy(v1374.begin() + 0, v1374.begin() + 0 + 800, v1375.begin());
  std::vector<float> v1376(224);
  std::copy(v1374.begin() + 800, v1374.begin() + 800 + 224, v1376.begin());
  std::copy(v1375.begin(), v1375.end(), v86.begin() + 224);
  std::copy(v1376.begin(), v1376.end(), v86.begin() + 0);
  std::vector<double> v1379(std::begin(v86), std::end(v86));
  auto pt231_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt231_filled = v1379;
  pt231_filled.clear();
  pt231_filled.reserve(pt231_filled_n);
  for (auto i = 0; i < pt231_filled_n; ++i) {
    pt231_filled.push_back(v1379[i % v1379.size()]);
  }
  auto pt231 = cc->MakeCKKSPackedPlaintext(pt231_filled);
  const auto& ct470 = cc->EvalMult(ct14, pt231);
  std::vector<float> v1380(std::begin(v28) + 232 * 512, std::begin(v28) + 232 * 512 + 1024);
  std::vector<float> v1381(800);
  std::copy(v1380.begin() + 0, v1380.begin() + 0 + 800, v1381.begin());
  std::vector<float> v1382(224);
  std::copy(v1380.begin() + 800, v1380.begin() + 800 + 224, v1382.begin());
  std::copy(v1381.begin(), v1381.end(), v86.begin() + 224);
  std::copy(v1382.begin(), v1382.end(), v86.begin() + 0);
  std::vector<double> v1385(std::begin(v86), std::end(v86));
  auto pt232_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt232_filled = v1385;
  pt232_filled.clear();
  pt232_filled.reserve(pt232_filled_n);
  for (auto i = 0; i < pt232_filled_n; ++i) {
    pt232_filled.push_back(v1385[i % v1385.size()]);
  }
  auto pt232 = cc->MakeCKKSPackedPlaintext(pt232_filled);
  const auto& ct471 = cc->EvalMult(ct16, pt232);
  std::vector<float> v1386(std::begin(v28) + 233 * 512, std::begin(v28) + 233 * 512 + 1024);
  std::vector<float> v1387(800);
  std::copy(v1386.begin() + 0, v1386.begin() + 0 + 800, v1387.begin());
  std::vector<float> v1388(224);
  std::copy(v1386.begin() + 800, v1386.begin() + 800 + 224, v1388.begin());
  std::copy(v1387.begin(), v1387.end(), v86.begin() + 224);
  std::copy(v1388.begin(), v1388.end(), v86.begin() + 0);
  std::vector<double> v1391(std::begin(v86), std::end(v86));
  auto pt233_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt233_filled = v1391;
  pt233_filled.clear();
  pt233_filled.reserve(pt233_filled_n);
  for (auto i = 0; i < pt233_filled_n; ++i) {
    pt233_filled.push_back(v1391[i % v1391.size()]);
  }
  auto pt233 = cc->MakeCKKSPackedPlaintext(pt233_filled);
  const auto& ct472 = cc->EvalMult(ct18, pt233);
  std::vector<float> v1392(std::begin(v28) + 234 * 512, std::begin(v28) + 234 * 512 + 1024);
  std::vector<float> v1393(800);
  std::copy(v1392.begin() + 0, v1392.begin() + 0 + 800, v1393.begin());
  std::vector<float> v1394(224);
  std::copy(v1392.begin() + 800, v1392.begin() + 800 + 224, v1394.begin());
  std::copy(v1393.begin(), v1393.end(), v86.begin() + 224);
  std::copy(v1394.begin(), v1394.end(), v86.begin() + 0);
  std::vector<double> v1397(std::begin(v86), std::end(v86));
  auto pt234_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt234_filled = v1397;
  pt234_filled.clear();
  pt234_filled.reserve(pt234_filled_n);
  for (auto i = 0; i < pt234_filled_n; ++i) {
    pt234_filled.push_back(v1397[i % v1397.size()]);
  }
  auto pt234 = cc->MakeCKKSPackedPlaintext(pt234_filled);
  const auto& ct473 = cc->EvalMult(ct20, pt234);
  std::vector<float> v1398(std::begin(v28) + 235 * 512, std::begin(v28) + 235 * 512 + 1024);
  std::vector<float> v1399(800);
  std::copy(v1398.begin() + 0, v1398.begin() + 0 + 800, v1399.begin());
  std::vector<float> v1400(224);
  std::copy(v1398.begin() + 800, v1398.begin() + 800 + 224, v1400.begin());
  std::copy(v1399.begin(), v1399.end(), v86.begin() + 224);
  std::copy(v1400.begin(), v1400.end(), v86.begin() + 0);
  std::vector<double> v1403(std::begin(v86), std::end(v86));
  auto pt235_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt235_filled = v1403;
  pt235_filled.clear();
  pt235_filled.reserve(pt235_filled_n);
  for (auto i = 0; i < pt235_filled_n; ++i) {
    pt235_filled.push_back(v1403[i % v1403.size()]);
  }
  auto pt235 = cc->MakeCKKSPackedPlaintext(pt235_filled);
  const auto& ct474 = cc->EvalMult(ct22, pt235);
  std::vector<float> v1404(std::begin(v28) + 236 * 512, std::begin(v28) + 236 * 512 + 1024);
  std::vector<float> v1405(800);
  std::copy(v1404.begin() + 0, v1404.begin() + 0 + 800, v1405.begin());
  std::vector<float> v1406(224);
  std::copy(v1404.begin() + 800, v1404.begin() + 800 + 224, v1406.begin());
  std::copy(v1405.begin(), v1405.end(), v86.begin() + 224);
  std::copy(v1406.begin(), v1406.end(), v86.begin() + 0);
  std::vector<double> v1409(std::begin(v86), std::end(v86));
  auto pt236_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt236_filled = v1409;
  pt236_filled.clear();
  pt236_filled.reserve(pt236_filled_n);
  for (auto i = 0; i < pt236_filled_n; ++i) {
    pt236_filled.push_back(v1409[i % v1409.size()]);
  }
  auto pt236 = cc->MakeCKKSPackedPlaintext(pt236_filled);
  const auto& ct475 = cc->EvalMult(ct24, pt236);
  std::vector<float> v1410(std::begin(v28) + 237 * 512, std::begin(v28) + 237 * 512 + 1024);
  std::vector<float> v1411(800);
  std::copy(v1410.begin() + 0, v1410.begin() + 0 + 800, v1411.begin());
  std::vector<float> v1412(224);
  std::copy(v1410.begin() + 800, v1410.begin() + 800 + 224, v1412.begin());
  std::copy(v1411.begin(), v1411.end(), v86.begin() + 224);
  std::copy(v1412.begin(), v1412.end(), v86.begin() + 0);
  std::vector<double> v1415(std::begin(v86), std::end(v86));
  auto pt237_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt237_filled = v1415;
  pt237_filled.clear();
  pt237_filled.reserve(pt237_filled_n);
  for (auto i = 0; i < pt237_filled_n; ++i) {
    pt237_filled.push_back(v1415[i % v1415.size()]);
  }
  auto pt237 = cc->MakeCKKSPackedPlaintext(pt237_filled);
  const auto& ct476 = cc->EvalMult(ct26, pt237);
  std::vector<float> v1416(std::begin(v28) + 238 * 512, std::begin(v28) + 238 * 512 + 1024);
  std::vector<float> v1417(800);
  std::copy(v1416.begin() + 0, v1416.begin() + 0 + 800, v1417.begin());
  std::vector<float> v1418(224);
  std::copy(v1416.begin() + 800, v1416.begin() + 800 + 224, v1418.begin());
  std::copy(v1417.begin(), v1417.end(), v86.begin() + 224);
  std::copy(v1418.begin(), v1418.end(), v86.begin() + 0);
  std::vector<double> v1421(std::begin(v86), std::end(v86));
  auto pt238_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt238_filled = v1421;
  pt238_filled.clear();
  pt238_filled.reserve(pt238_filled_n);
  for (auto i = 0; i < pt238_filled_n; ++i) {
    pt238_filled.push_back(v1421[i % v1421.size()]);
  }
  auto pt238 = cc->MakeCKKSPackedPlaintext(pt238_filled);
  const auto& ct477 = cc->EvalMult(ct28, pt238);
  std::vector<float> v1422(std::begin(v28) + 239 * 512, std::begin(v28) + 239 * 512 + 1024);
  std::vector<float> v1423(800);
  std::copy(v1422.begin() + 0, v1422.begin() + 0 + 800, v1423.begin());
  std::vector<float> v1424(224);
  std::copy(v1422.begin() + 800, v1422.begin() + 800 + 224, v1424.begin());
  std::copy(v1423.begin(), v1423.end(), v86.begin() + 224);
  std::copy(v1424.begin(), v1424.end(), v86.begin() + 0);
  std::vector<double> v1427(std::begin(v86), std::end(v86));
  auto pt239_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt239_filled = v1427;
  pt239_filled.clear();
  pt239_filled.reserve(pt239_filled_n);
  for (auto i = 0; i < pt239_filled_n; ++i) {
    pt239_filled.push_back(v1427[i % v1427.size()]);
  }
  auto pt239 = cc->MakeCKKSPackedPlaintext(pt239_filled);
  const auto& ct478 = cc->EvalMult(ct30, pt239);
  const auto& ct479 = cc->EvalAdd(ct463, ct464);
  const auto& ct480 = cc->EvalAdd(ct465, ct466);
  const auto& ct481 = cc->EvalAdd(ct479, ct480);
  const auto& ct482 = cc->EvalAdd(ct467, ct468);
  const auto& ct483 = cc->EvalAdd(ct469, ct470);
  const auto& ct484 = cc->EvalAdd(ct482, ct483);
  const auto& ct485 = cc->EvalAdd(ct481, ct484);
  const auto& ct486 = cc->EvalAdd(ct471, ct472);
  const auto& ct487 = cc->EvalAdd(ct473, ct474);
  const auto& ct488 = cc->EvalAdd(ct486, ct487);
  const auto& ct489 = cc->EvalAdd(ct475, ct476);
  const auto& ct490 = cc->EvalAdd(ct477, ct478);
  const auto& ct491 = cc->EvalAdd(ct489, ct490);
  const auto& ct492 = cc->EvalAdd(ct488, ct491);
  const auto& ct493 = cc->EvalAdd(ct485, ct492);
  const auto& ct494 = cc->EvalRotate(ct493, 224);
  std::vector<float> v1428(std::begin(v28) + 240 * 512, std::begin(v28) + 240 * 512 + 1024);
  std::vector<float> v1429(784);
  std::copy(v1428.begin() + 0, v1428.begin() + 0 + 784, v1429.begin());
  std::vector<float> v1430(240);
  std::copy(v1428.begin() + 784, v1428.begin() + 784 + 240, v1430.begin());
  std::copy(v1429.begin(), v1429.end(), v86.begin() + 240);
  std::copy(v1430.begin(), v1430.end(), v86.begin() + 0);
  std::vector<double> v1433(std::begin(v86), std::end(v86));
  auto pt240_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt240_filled = v1433;
  pt240_filled.clear();
  pt240_filled.reserve(pt240_filled_n);
  for (auto i = 0; i < pt240_filled_n; ++i) {
    pt240_filled.push_back(v1433[i % v1433.size()]);
  }
  auto pt240 = cc->MakeCKKSPackedPlaintext(pt240_filled);
  const auto& ct495 = cc->EvalMult(ct, pt240);
  std::vector<float> v1434(std::begin(v28) + 241 * 512, std::begin(v28) + 241 * 512 + 1024);
  std::vector<float> v1435(784);
  std::copy(v1434.begin() + 0, v1434.begin() + 0 + 784, v1435.begin());
  std::vector<float> v1436(240);
  std::copy(v1434.begin() + 784, v1434.begin() + 784 + 240, v1436.begin());
  std::copy(v1435.begin(), v1435.end(), v86.begin() + 240);
  std::copy(v1436.begin(), v1436.end(), v86.begin() + 0);
  std::vector<double> v1439(std::begin(v86), std::end(v86));
  auto pt241_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt241_filled = v1439;
  pt241_filled.clear();
  pt241_filled.reserve(pt241_filled_n);
  for (auto i = 0; i < pt241_filled_n; ++i) {
    pt241_filled.push_back(v1439[i % v1439.size()]);
  }
  auto pt241 = cc->MakeCKKSPackedPlaintext(pt241_filled);
  const auto& ct496 = cc->EvalMult(ct2, pt241);
  std::vector<float> v1440(std::begin(v28) + 242 * 512, std::begin(v28) + 242 * 512 + 1024);
  std::vector<float> v1441(784);
  std::copy(v1440.begin() + 0, v1440.begin() + 0 + 784, v1441.begin());
  std::vector<float> v1442(240);
  std::copy(v1440.begin() + 784, v1440.begin() + 784 + 240, v1442.begin());
  std::copy(v1441.begin(), v1441.end(), v86.begin() + 240);
  std::copy(v1442.begin(), v1442.end(), v86.begin() + 0);
  std::vector<double> v1445(std::begin(v86), std::end(v86));
  auto pt242_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt242_filled = v1445;
  pt242_filled.clear();
  pt242_filled.reserve(pt242_filled_n);
  for (auto i = 0; i < pt242_filled_n; ++i) {
    pt242_filled.push_back(v1445[i % v1445.size()]);
  }
  auto pt242 = cc->MakeCKKSPackedPlaintext(pt242_filled);
  const auto& ct497 = cc->EvalMult(ct4, pt242);
  std::vector<float> v1446(std::begin(v28) + 243 * 512, std::begin(v28) + 243 * 512 + 1024);
  std::vector<float> v1447(784);
  std::copy(v1446.begin() + 0, v1446.begin() + 0 + 784, v1447.begin());
  std::vector<float> v1448(240);
  std::copy(v1446.begin() + 784, v1446.begin() + 784 + 240, v1448.begin());
  std::copy(v1447.begin(), v1447.end(), v86.begin() + 240);
  std::copy(v1448.begin(), v1448.end(), v86.begin() + 0);
  std::vector<double> v1451(std::begin(v86), std::end(v86));
  auto pt243_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt243_filled = v1451;
  pt243_filled.clear();
  pt243_filled.reserve(pt243_filled_n);
  for (auto i = 0; i < pt243_filled_n; ++i) {
    pt243_filled.push_back(v1451[i % v1451.size()]);
  }
  auto pt243 = cc->MakeCKKSPackedPlaintext(pt243_filled);
  const auto& ct498 = cc->EvalMult(ct6, pt243);
  std::vector<float> v1452(std::begin(v28) + 244 * 512, std::begin(v28) + 244 * 512 + 1024);
  std::vector<float> v1453(784);
  std::copy(v1452.begin() + 0, v1452.begin() + 0 + 784, v1453.begin());
  std::vector<float> v1454(240);
  std::copy(v1452.begin() + 784, v1452.begin() + 784 + 240, v1454.begin());
  std::copy(v1453.begin(), v1453.end(), v86.begin() + 240);
  std::copy(v1454.begin(), v1454.end(), v86.begin() + 0);
  std::vector<double> v1457(std::begin(v86), std::end(v86));
  auto pt244_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt244_filled = v1457;
  pt244_filled.clear();
  pt244_filled.reserve(pt244_filled_n);
  for (auto i = 0; i < pt244_filled_n; ++i) {
    pt244_filled.push_back(v1457[i % v1457.size()]);
  }
  auto pt244 = cc->MakeCKKSPackedPlaintext(pt244_filled);
  const auto& ct499 = cc->EvalMult(ct8, pt244);
  std::vector<float> v1458(std::begin(v28) + 245 * 512, std::begin(v28) + 245 * 512 + 1024);
  std::vector<float> v1459(784);
  std::copy(v1458.begin() + 0, v1458.begin() + 0 + 784, v1459.begin());
  std::vector<float> v1460(240);
  std::copy(v1458.begin() + 784, v1458.begin() + 784 + 240, v1460.begin());
  std::copy(v1459.begin(), v1459.end(), v86.begin() + 240);
  std::copy(v1460.begin(), v1460.end(), v86.begin() + 0);
  std::vector<double> v1463(std::begin(v86), std::end(v86));
  auto pt245_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt245_filled = v1463;
  pt245_filled.clear();
  pt245_filled.reserve(pt245_filled_n);
  for (auto i = 0; i < pt245_filled_n; ++i) {
    pt245_filled.push_back(v1463[i % v1463.size()]);
  }
  auto pt245 = cc->MakeCKKSPackedPlaintext(pt245_filled);
  const auto& ct500 = cc->EvalMult(ct10, pt245);
  std::vector<float> v1464(std::begin(v28) + 246 * 512, std::begin(v28) + 246 * 512 + 1024);
  std::vector<float> v1465(784);
  std::copy(v1464.begin() + 0, v1464.begin() + 0 + 784, v1465.begin());
  std::vector<float> v1466(240);
  std::copy(v1464.begin() + 784, v1464.begin() + 784 + 240, v1466.begin());
  std::copy(v1465.begin(), v1465.end(), v86.begin() + 240);
  std::copy(v1466.begin(), v1466.end(), v86.begin() + 0);
  std::vector<double> v1469(std::begin(v86), std::end(v86));
  auto pt246_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt246_filled = v1469;
  pt246_filled.clear();
  pt246_filled.reserve(pt246_filled_n);
  for (auto i = 0; i < pt246_filled_n; ++i) {
    pt246_filled.push_back(v1469[i % v1469.size()]);
  }
  auto pt246 = cc->MakeCKKSPackedPlaintext(pt246_filled);
  const auto& ct501 = cc->EvalMult(ct12, pt246);
  std::vector<float> v1470(std::begin(v28) + 247 * 512, std::begin(v28) + 247 * 512 + 1024);
  std::vector<float> v1471(784);
  std::copy(v1470.begin() + 0, v1470.begin() + 0 + 784, v1471.begin());
  std::vector<float> v1472(240);
  std::copy(v1470.begin() + 784, v1470.begin() + 784 + 240, v1472.begin());
  std::copy(v1471.begin(), v1471.end(), v86.begin() + 240);
  std::copy(v1472.begin(), v1472.end(), v86.begin() + 0);
  std::vector<double> v1475(std::begin(v86), std::end(v86));
  auto pt247_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt247_filled = v1475;
  pt247_filled.clear();
  pt247_filled.reserve(pt247_filled_n);
  for (auto i = 0; i < pt247_filled_n; ++i) {
    pt247_filled.push_back(v1475[i % v1475.size()]);
  }
  auto pt247 = cc->MakeCKKSPackedPlaintext(pt247_filled);
  const auto& ct502 = cc->EvalMult(ct14, pt247);
  std::vector<float> v1476(std::begin(v28) + 248 * 512, std::begin(v28) + 248 * 512 + 1024);
  std::vector<float> v1477(784);
  std::copy(v1476.begin() + 0, v1476.begin() + 0 + 784, v1477.begin());
  std::vector<float> v1478(240);
  std::copy(v1476.begin() + 784, v1476.begin() + 784 + 240, v1478.begin());
  std::copy(v1477.begin(), v1477.end(), v86.begin() + 240);
  std::copy(v1478.begin(), v1478.end(), v86.begin() + 0);
  std::vector<double> v1481(std::begin(v86), std::end(v86));
  auto pt248_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt248_filled = v1481;
  pt248_filled.clear();
  pt248_filled.reserve(pt248_filled_n);
  for (auto i = 0; i < pt248_filled_n; ++i) {
    pt248_filled.push_back(v1481[i % v1481.size()]);
  }
  auto pt248 = cc->MakeCKKSPackedPlaintext(pt248_filled);
  const auto& ct503 = cc->EvalMult(ct16, pt248);
  std::vector<float> v1482(std::begin(v28) + 249 * 512, std::begin(v28) + 249 * 512 + 1024);
  std::vector<float> v1483(784);
  std::copy(v1482.begin() + 0, v1482.begin() + 0 + 784, v1483.begin());
  std::vector<float> v1484(240);
  std::copy(v1482.begin() + 784, v1482.begin() + 784 + 240, v1484.begin());
  std::copy(v1483.begin(), v1483.end(), v86.begin() + 240);
  std::copy(v1484.begin(), v1484.end(), v86.begin() + 0);
  std::vector<double> v1487(std::begin(v86), std::end(v86));
  auto pt249_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt249_filled = v1487;
  pt249_filled.clear();
  pt249_filled.reserve(pt249_filled_n);
  for (auto i = 0; i < pt249_filled_n; ++i) {
    pt249_filled.push_back(v1487[i % v1487.size()]);
  }
  auto pt249 = cc->MakeCKKSPackedPlaintext(pt249_filled);
  const auto& ct504 = cc->EvalMult(ct18, pt249);
  std::vector<float> v1488(std::begin(v28) + 250 * 512, std::begin(v28) + 250 * 512 + 1024);
  std::vector<float> v1489(784);
  std::copy(v1488.begin() + 0, v1488.begin() + 0 + 784, v1489.begin());
  std::vector<float> v1490(240);
  std::copy(v1488.begin() + 784, v1488.begin() + 784 + 240, v1490.begin());
  std::copy(v1489.begin(), v1489.end(), v86.begin() + 240);
  std::copy(v1490.begin(), v1490.end(), v86.begin() + 0);
  std::vector<double> v1493(std::begin(v86), std::end(v86));
  auto pt250_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt250_filled = v1493;
  pt250_filled.clear();
  pt250_filled.reserve(pt250_filled_n);
  for (auto i = 0; i < pt250_filled_n; ++i) {
    pt250_filled.push_back(v1493[i % v1493.size()]);
  }
  auto pt250 = cc->MakeCKKSPackedPlaintext(pt250_filled);
  const auto& ct505 = cc->EvalMult(ct20, pt250);
  std::vector<float> v1494(std::begin(v28) + 251 * 512, std::begin(v28) + 251 * 512 + 1024);
  std::vector<float> v1495(784);
  std::copy(v1494.begin() + 0, v1494.begin() + 0 + 784, v1495.begin());
  std::vector<float> v1496(240);
  std::copy(v1494.begin() + 784, v1494.begin() + 784 + 240, v1496.begin());
  std::copy(v1495.begin(), v1495.end(), v86.begin() + 240);
  std::copy(v1496.begin(), v1496.end(), v86.begin() + 0);
  std::vector<double> v1499(std::begin(v86), std::end(v86));
  auto pt251_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt251_filled = v1499;
  pt251_filled.clear();
  pt251_filled.reserve(pt251_filled_n);
  for (auto i = 0; i < pt251_filled_n; ++i) {
    pt251_filled.push_back(v1499[i % v1499.size()]);
  }
  auto pt251 = cc->MakeCKKSPackedPlaintext(pt251_filled);
  const auto& ct506 = cc->EvalMult(ct22, pt251);
  std::vector<float> v1500(std::begin(v28) + 252 * 512, std::begin(v28) + 252 * 512 + 1024);
  std::vector<float> v1501(784);
  std::copy(v1500.begin() + 0, v1500.begin() + 0 + 784, v1501.begin());
  std::vector<float> v1502(240);
  std::copy(v1500.begin() + 784, v1500.begin() + 784 + 240, v1502.begin());
  std::copy(v1501.begin(), v1501.end(), v86.begin() + 240);
  std::copy(v1502.begin(), v1502.end(), v86.begin() + 0);
  std::vector<double> v1505(std::begin(v86), std::end(v86));
  auto pt252_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt252_filled = v1505;
  pt252_filled.clear();
  pt252_filled.reserve(pt252_filled_n);
  for (auto i = 0; i < pt252_filled_n; ++i) {
    pt252_filled.push_back(v1505[i % v1505.size()]);
  }
  auto pt252 = cc->MakeCKKSPackedPlaintext(pt252_filled);
  const auto& ct507 = cc->EvalMult(ct24, pt252);
  std::vector<float> v1506(std::begin(v28) + 253 * 512, std::begin(v28) + 253 * 512 + 1024);
  std::vector<float> v1507(784);
  std::copy(v1506.begin() + 0, v1506.begin() + 0 + 784, v1507.begin());
  std::vector<float> v1508(240);
  std::copy(v1506.begin() + 784, v1506.begin() + 784 + 240, v1508.begin());
  std::copy(v1507.begin(), v1507.end(), v86.begin() + 240);
  std::copy(v1508.begin(), v1508.end(), v86.begin() + 0);
  std::vector<double> v1511(std::begin(v86), std::end(v86));
  auto pt253_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt253_filled = v1511;
  pt253_filled.clear();
  pt253_filled.reserve(pt253_filled_n);
  for (auto i = 0; i < pt253_filled_n; ++i) {
    pt253_filled.push_back(v1511[i % v1511.size()]);
  }
  auto pt253 = cc->MakeCKKSPackedPlaintext(pt253_filled);
  const auto& ct508 = cc->EvalMult(ct26, pt253);
  std::vector<float> v1512(std::begin(v28) + 254 * 512, std::begin(v28) + 254 * 512 + 1024);
  std::vector<float> v1513(784);
  std::copy(v1512.begin() + 0, v1512.begin() + 0 + 784, v1513.begin());
  std::vector<float> v1514(240);
  std::copy(v1512.begin() + 784, v1512.begin() + 784 + 240, v1514.begin());
  std::copy(v1513.begin(), v1513.end(), v86.begin() + 240);
  std::copy(v1514.begin(), v1514.end(), v86.begin() + 0);
  std::vector<double> v1517(std::begin(v86), std::end(v86));
  auto pt254_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt254_filled = v1517;
  pt254_filled.clear();
  pt254_filled.reserve(pt254_filled_n);
  for (auto i = 0; i < pt254_filled_n; ++i) {
    pt254_filled.push_back(v1517[i % v1517.size()]);
  }
  auto pt254 = cc->MakeCKKSPackedPlaintext(pt254_filled);
  const auto& ct509 = cc->EvalMult(ct28, pt254);
  std::vector<float> v1518(std::begin(v28) + 255 * 512, std::begin(v28) + 255 * 512 + 1024);
  std::vector<float> v1519(784);
  std::copy(v1518.begin() + 0, v1518.begin() + 0 + 784, v1519.begin());
  std::vector<float> v1520(240);
  std::copy(v1518.begin() + 784, v1518.begin() + 784 + 240, v1520.begin());
  std::copy(v1519.begin(), v1519.end(), v86.begin() + 240);
  std::copy(v1520.begin(), v1520.end(), v86.begin() + 0);
  std::vector<double> v1523(std::begin(v86), std::end(v86));
  auto pt255_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt255_filled = v1523;
  pt255_filled.clear();
  pt255_filled.reserve(pt255_filled_n);
  for (auto i = 0; i < pt255_filled_n; ++i) {
    pt255_filled.push_back(v1523[i % v1523.size()]);
  }
  auto pt255 = cc->MakeCKKSPackedPlaintext(pt255_filled);
  const auto& ct510 = cc->EvalMult(ct30, pt255);
  const auto& ct511 = cc->EvalAdd(ct495, ct496);
  const auto& ct512 = cc->EvalAdd(ct497, ct498);
  const auto& ct513 = cc->EvalAdd(ct511, ct512);
  const auto& ct514 = cc->EvalAdd(ct499, ct500);
  const auto& ct515 = cc->EvalAdd(ct501, ct502);
  const auto& ct516 = cc->EvalAdd(ct514, ct515);
  const auto& ct517 = cc->EvalAdd(ct513, ct516);
  const auto& ct518 = cc->EvalAdd(ct503, ct504);
  const auto& ct519 = cc->EvalAdd(ct505, ct506);
  const auto& ct520 = cc->EvalAdd(ct518, ct519);
  const auto& ct521 = cc->EvalAdd(ct507, ct508);
  const auto& ct522 = cc->EvalAdd(ct509, ct510);
  const auto& ct523 = cc->EvalAdd(ct521, ct522);
  const auto& ct524 = cc->EvalAdd(ct520, ct523);
  const auto& ct525 = cc->EvalAdd(ct517, ct524);
  const auto& ct526 = cc->EvalRotate(ct525, 240);
  std::vector<float> v1524(std::begin(v28) + 256 * 512, std::begin(v28) + 256 * 512 + 1024);
  std::vector<float> v1525(768);
  std::copy(v1524.begin() + 0, v1524.begin() + 0 + 768, v1525.begin());
  std::vector<float> v1526(256);
  std::copy(v1524.begin() + 768, v1524.begin() + 768 + 256, v1526.begin());
  std::copy(v1525.begin(), v1525.end(), v86.begin() + 256);
  std::copy(v1526.begin(), v1526.end(), v86.begin() + 0);
  std::vector<double> v1529(std::begin(v86), std::end(v86));
  auto pt256_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt256_filled = v1529;
  pt256_filled.clear();
  pt256_filled.reserve(pt256_filled_n);
  for (auto i = 0; i < pt256_filled_n; ++i) {
    pt256_filled.push_back(v1529[i % v1529.size()]);
  }
  auto pt256 = cc->MakeCKKSPackedPlaintext(pt256_filled);
  const auto& ct527 = cc->EvalMult(ct, pt256);
  std::vector<float> v1530(std::begin(v28) + 257 * 512, std::begin(v28) + 257 * 512 + 1024);
  std::vector<float> v1531(768);
  std::copy(v1530.begin() + 0, v1530.begin() + 0 + 768, v1531.begin());
  std::vector<float> v1532(256);
  std::copy(v1530.begin() + 768, v1530.begin() + 768 + 256, v1532.begin());
  std::copy(v1531.begin(), v1531.end(), v86.begin() + 256);
  std::copy(v1532.begin(), v1532.end(), v86.begin() + 0);
  std::vector<double> v1535(std::begin(v86), std::end(v86));
  auto pt257_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt257_filled = v1535;
  pt257_filled.clear();
  pt257_filled.reserve(pt257_filled_n);
  for (auto i = 0; i < pt257_filled_n; ++i) {
    pt257_filled.push_back(v1535[i % v1535.size()]);
  }
  auto pt257 = cc->MakeCKKSPackedPlaintext(pt257_filled);
  const auto& ct528 = cc->EvalMult(ct2, pt257);
  std::vector<float> v1536(std::begin(v28) + 258 * 512, std::begin(v28) + 258 * 512 + 1024);
  std::vector<float> v1537(768);
  std::copy(v1536.begin() + 0, v1536.begin() + 0 + 768, v1537.begin());
  std::vector<float> v1538(256);
  std::copy(v1536.begin() + 768, v1536.begin() + 768 + 256, v1538.begin());
  std::copy(v1537.begin(), v1537.end(), v86.begin() + 256);
  std::copy(v1538.begin(), v1538.end(), v86.begin() + 0);
  std::vector<double> v1541(std::begin(v86), std::end(v86));
  auto pt258_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt258_filled = v1541;
  pt258_filled.clear();
  pt258_filled.reserve(pt258_filled_n);
  for (auto i = 0; i < pt258_filled_n; ++i) {
    pt258_filled.push_back(v1541[i % v1541.size()]);
  }
  auto pt258 = cc->MakeCKKSPackedPlaintext(pt258_filled);
  const auto& ct529 = cc->EvalMult(ct4, pt258);
  std::vector<float> v1542(std::begin(v28) + 259 * 512, std::begin(v28) + 259 * 512 + 1024);
  std::vector<float> v1543(768);
  std::copy(v1542.begin() + 0, v1542.begin() + 0 + 768, v1543.begin());
  std::vector<float> v1544(256);
  std::copy(v1542.begin() + 768, v1542.begin() + 768 + 256, v1544.begin());
  std::copy(v1543.begin(), v1543.end(), v86.begin() + 256);
  std::copy(v1544.begin(), v1544.end(), v86.begin() + 0);
  std::vector<double> v1547(std::begin(v86), std::end(v86));
  auto pt259_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt259_filled = v1547;
  pt259_filled.clear();
  pt259_filled.reserve(pt259_filled_n);
  for (auto i = 0; i < pt259_filled_n; ++i) {
    pt259_filled.push_back(v1547[i % v1547.size()]);
  }
  auto pt259 = cc->MakeCKKSPackedPlaintext(pt259_filled);
  const auto& ct530 = cc->EvalMult(ct6, pt259);
  std::vector<float> v1548(std::begin(v28) + 260 * 512, std::begin(v28) + 260 * 512 + 1024);
  std::vector<float> v1549(768);
  std::copy(v1548.begin() + 0, v1548.begin() + 0 + 768, v1549.begin());
  std::vector<float> v1550(256);
  std::copy(v1548.begin() + 768, v1548.begin() + 768 + 256, v1550.begin());
  std::copy(v1549.begin(), v1549.end(), v86.begin() + 256);
  std::copy(v1550.begin(), v1550.end(), v86.begin() + 0);
  std::vector<double> v1553(std::begin(v86), std::end(v86));
  auto pt260_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt260_filled = v1553;
  pt260_filled.clear();
  pt260_filled.reserve(pt260_filled_n);
  for (auto i = 0; i < pt260_filled_n; ++i) {
    pt260_filled.push_back(v1553[i % v1553.size()]);
  }
  auto pt260 = cc->MakeCKKSPackedPlaintext(pt260_filled);
  const auto& ct531 = cc->EvalMult(ct8, pt260);
  std::vector<float> v1554(std::begin(v28) + 261 * 512, std::begin(v28) + 261 * 512 + 1024);
  std::vector<float> v1555(768);
  std::copy(v1554.begin() + 0, v1554.begin() + 0 + 768, v1555.begin());
  std::vector<float> v1556(256);
  std::copy(v1554.begin() + 768, v1554.begin() + 768 + 256, v1556.begin());
  std::copy(v1555.begin(), v1555.end(), v86.begin() + 256);
  std::copy(v1556.begin(), v1556.end(), v86.begin() + 0);
  std::vector<double> v1559(std::begin(v86), std::end(v86));
  auto pt261_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt261_filled = v1559;
  pt261_filled.clear();
  pt261_filled.reserve(pt261_filled_n);
  for (auto i = 0; i < pt261_filled_n; ++i) {
    pt261_filled.push_back(v1559[i % v1559.size()]);
  }
  auto pt261 = cc->MakeCKKSPackedPlaintext(pt261_filled);
  const auto& ct532 = cc->EvalMult(ct10, pt261);
  std::vector<float> v1560(std::begin(v28) + 262 * 512, std::begin(v28) + 262 * 512 + 1024);
  std::vector<float> v1561(768);
  std::copy(v1560.begin() + 0, v1560.begin() + 0 + 768, v1561.begin());
  std::vector<float> v1562(256);
  std::copy(v1560.begin() + 768, v1560.begin() + 768 + 256, v1562.begin());
  std::copy(v1561.begin(), v1561.end(), v86.begin() + 256);
  std::copy(v1562.begin(), v1562.end(), v86.begin() + 0);
  std::vector<double> v1565(std::begin(v86), std::end(v86));
  auto pt262_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt262_filled = v1565;
  pt262_filled.clear();
  pt262_filled.reserve(pt262_filled_n);
  for (auto i = 0; i < pt262_filled_n; ++i) {
    pt262_filled.push_back(v1565[i % v1565.size()]);
  }
  auto pt262 = cc->MakeCKKSPackedPlaintext(pt262_filled);
  const auto& ct533 = cc->EvalMult(ct12, pt262);
  std::vector<float> v1566(std::begin(v28) + 263 * 512, std::begin(v28) + 263 * 512 + 1024);
  std::vector<float> v1567(768);
  std::copy(v1566.begin() + 0, v1566.begin() + 0 + 768, v1567.begin());
  std::vector<float> v1568(256);
  std::copy(v1566.begin() + 768, v1566.begin() + 768 + 256, v1568.begin());
  std::copy(v1567.begin(), v1567.end(), v86.begin() + 256);
  std::copy(v1568.begin(), v1568.end(), v86.begin() + 0);
  std::vector<double> v1571(std::begin(v86), std::end(v86));
  auto pt263_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt263_filled = v1571;
  pt263_filled.clear();
  pt263_filled.reserve(pt263_filled_n);
  for (auto i = 0; i < pt263_filled_n; ++i) {
    pt263_filled.push_back(v1571[i % v1571.size()]);
  }
  auto pt263 = cc->MakeCKKSPackedPlaintext(pt263_filled);
  const auto& ct534 = cc->EvalMult(ct14, pt263);
  std::vector<float> v1572(std::begin(v28) + 264 * 512, std::begin(v28) + 264 * 512 + 1024);
  std::vector<float> v1573(768);
  std::copy(v1572.begin() + 0, v1572.begin() + 0 + 768, v1573.begin());
  std::vector<float> v1574(256);
  std::copy(v1572.begin() + 768, v1572.begin() + 768 + 256, v1574.begin());
  std::copy(v1573.begin(), v1573.end(), v86.begin() + 256);
  std::copy(v1574.begin(), v1574.end(), v86.begin() + 0);
  std::vector<double> v1577(std::begin(v86), std::end(v86));
  auto pt264_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt264_filled = v1577;
  pt264_filled.clear();
  pt264_filled.reserve(pt264_filled_n);
  for (auto i = 0; i < pt264_filled_n; ++i) {
    pt264_filled.push_back(v1577[i % v1577.size()]);
  }
  auto pt264 = cc->MakeCKKSPackedPlaintext(pt264_filled);
  const auto& ct535 = cc->EvalMult(ct16, pt264);
  std::vector<float> v1578(std::begin(v28) + 265 * 512, std::begin(v28) + 265 * 512 + 1024);
  std::vector<float> v1579(768);
  std::copy(v1578.begin() + 0, v1578.begin() + 0 + 768, v1579.begin());
  std::vector<float> v1580(256);
  std::copy(v1578.begin() + 768, v1578.begin() + 768 + 256, v1580.begin());
  std::copy(v1579.begin(), v1579.end(), v86.begin() + 256);
  std::copy(v1580.begin(), v1580.end(), v86.begin() + 0);
  std::vector<double> v1583(std::begin(v86), std::end(v86));
  auto pt265_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt265_filled = v1583;
  pt265_filled.clear();
  pt265_filled.reserve(pt265_filled_n);
  for (auto i = 0; i < pt265_filled_n; ++i) {
    pt265_filled.push_back(v1583[i % v1583.size()]);
  }
  auto pt265 = cc->MakeCKKSPackedPlaintext(pt265_filled);
  const auto& ct536 = cc->EvalMult(ct18, pt265);
  std::vector<float> v1584(std::begin(v28) + 266 * 512, std::begin(v28) + 266 * 512 + 1024);
  std::vector<float> v1585(768);
  std::copy(v1584.begin() + 0, v1584.begin() + 0 + 768, v1585.begin());
  std::vector<float> v1586(256);
  std::copy(v1584.begin() + 768, v1584.begin() + 768 + 256, v1586.begin());
  std::copy(v1585.begin(), v1585.end(), v86.begin() + 256);
  std::copy(v1586.begin(), v1586.end(), v86.begin() + 0);
  std::vector<double> v1589(std::begin(v86), std::end(v86));
  auto pt266_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt266_filled = v1589;
  pt266_filled.clear();
  pt266_filled.reserve(pt266_filled_n);
  for (auto i = 0; i < pt266_filled_n; ++i) {
    pt266_filled.push_back(v1589[i % v1589.size()]);
  }
  auto pt266 = cc->MakeCKKSPackedPlaintext(pt266_filled);
  const auto& ct537 = cc->EvalMult(ct20, pt266);
  std::vector<float> v1590(std::begin(v28) + 267 * 512, std::begin(v28) + 267 * 512 + 1024);
  std::vector<float> v1591(768);
  std::copy(v1590.begin() + 0, v1590.begin() + 0 + 768, v1591.begin());
  std::vector<float> v1592(256);
  std::copy(v1590.begin() + 768, v1590.begin() + 768 + 256, v1592.begin());
  std::copy(v1591.begin(), v1591.end(), v86.begin() + 256);
  std::copy(v1592.begin(), v1592.end(), v86.begin() + 0);
  std::vector<double> v1595(std::begin(v86), std::end(v86));
  auto pt267_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt267_filled = v1595;
  pt267_filled.clear();
  pt267_filled.reserve(pt267_filled_n);
  for (auto i = 0; i < pt267_filled_n; ++i) {
    pt267_filled.push_back(v1595[i % v1595.size()]);
  }
  auto pt267 = cc->MakeCKKSPackedPlaintext(pt267_filled);
  const auto& ct538 = cc->EvalMult(ct22, pt267);
  std::vector<float> v1596(std::begin(v28) + 268 * 512, std::begin(v28) + 268 * 512 + 1024);
  std::vector<float> v1597(768);
  std::copy(v1596.begin() + 0, v1596.begin() + 0 + 768, v1597.begin());
  std::vector<float> v1598(256);
  std::copy(v1596.begin() + 768, v1596.begin() + 768 + 256, v1598.begin());
  std::copy(v1597.begin(), v1597.end(), v86.begin() + 256);
  std::copy(v1598.begin(), v1598.end(), v86.begin() + 0);
  std::vector<double> v1601(std::begin(v86), std::end(v86));
  auto pt268_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt268_filled = v1601;
  pt268_filled.clear();
  pt268_filled.reserve(pt268_filled_n);
  for (auto i = 0; i < pt268_filled_n; ++i) {
    pt268_filled.push_back(v1601[i % v1601.size()]);
  }
  auto pt268 = cc->MakeCKKSPackedPlaintext(pt268_filled);
  const auto& ct539 = cc->EvalMult(ct24, pt268);
  std::vector<float> v1602(std::begin(v28) + 269 * 512, std::begin(v28) + 269 * 512 + 1024);
  std::vector<float> v1603(768);
  std::copy(v1602.begin() + 0, v1602.begin() + 0 + 768, v1603.begin());
  std::vector<float> v1604(256);
  std::copy(v1602.begin() + 768, v1602.begin() + 768 + 256, v1604.begin());
  std::copy(v1603.begin(), v1603.end(), v86.begin() + 256);
  std::copy(v1604.begin(), v1604.end(), v86.begin() + 0);
  std::vector<double> v1607(std::begin(v86), std::end(v86));
  auto pt269_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt269_filled = v1607;
  pt269_filled.clear();
  pt269_filled.reserve(pt269_filled_n);
  for (auto i = 0; i < pt269_filled_n; ++i) {
    pt269_filled.push_back(v1607[i % v1607.size()]);
  }
  auto pt269 = cc->MakeCKKSPackedPlaintext(pt269_filled);
  const auto& ct540 = cc->EvalMult(ct26, pt269);
  std::vector<float> v1608(std::begin(v28) + 270 * 512, std::begin(v28) + 270 * 512 + 1024);
  std::vector<float> v1609(768);
  std::copy(v1608.begin() + 0, v1608.begin() + 0 + 768, v1609.begin());
  std::vector<float> v1610(256);
  std::copy(v1608.begin() + 768, v1608.begin() + 768 + 256, v1610.begin());
  std::copy(v1609.begin(), v1609.end(), v86.begin() + 256);
  std::copy(v1610.begin(), v1610.end(), v86.begin() + 0);
  std::vector<double> v1613(std::begin(v86), std::end(v86));
  auto pt270_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt270_filled = v1613;
  pt270_filled.clear();
  pt270_filled.reserve(pt270_filled_n);
  for (auto i = 0; i < pt270_filled_n; ++i) {
    pt270_filled.push_back(v1613[i % v1613.size()]);
  }
  auto pt270 = cc->MakeCKKSPackedPlaintext(pt270_filled);
  const auto& ct541 = cc->EvalMult(ct28, pt270);
  std::vector<float> v1614(std::begin(v28) + 271 * 512, std::begin(v28) + 271 * 512 + 1024);
  std::vector<float> v1615(768);
  std::copy(v1614.begin() + 0, v1614.begin() + 0 + 768, v1615.begin());
  std::vector<float> v1616(256);
  std::copy(v1614.begin() + 768, v1614.begin() + 768 + 256, v1616.begin());
  std::copy(v1615.begin(), v1615.end(), v86.begin() + 256);
  std::copy(v1616.begin(), v1616.end(), v86.begin() + 0);
  std::vector<double> v1619(std::begin(v86), std::end(v86));
  auto pt271_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt271_filled = v1619;
  pt271_filled.clear();
  pt271_filled.reserve(pt271_filled_n);
  for (auto i = 0; i < pt271_filled_n; ++i) {
    pt271_filled.push_back(v1619[i % v1619.size()]);
  }
  auto pt271 = cc->MakeCKKSPackedPlaintext(pt271_filled);
  const auto& ct542 = cc->EvalMult(ct30, pt271);
  const auto& ct543 = cc->EvalAdd(ct527, ct528);
  const auto& ct544 = cc->EvalAdd(ct529, ct530);
  const auto& ct545 = cc->EvalAdd(ct543, ct544);
  const auto& ct546 = cc->EvalAdd(ct531, ct532);
  const auto& ct547 = cc->EvalAdd(ct533, ct534);
  const auto& ct548 = cc->EvalAdd(ct546, ct547);
  const auto& ct549 = cc->EvalAdd(ct545, ct548);
  const auto& ct550 = cc->EvalAdd(ct535, ct536);
  const auto& ct551 = cc->EvalAdd(ct537, ct538);
  const auto& ct552 = cc->EvalAdd(ct550, ct551);
  const auto& ct553 = cc->EvalAdd(ct539, ct540);
  const auto& ct554 = cc->EvalAdd(ct541, ct542);
  const auto& ct555 = cc->EvalAdd(ct553, ct554);
  const auto& ct556 = cc->EvalAdd(ct552, ct555);
  const auto& ct557 = cc->EvalAdd(ct549, ct556);
  const auto& ct558 = cc->EvalRotate(ct557, 256);
  std::vector<float> v1620(std::begin(v28) + 272 * 512, std::begin(v28) + 272 * 512 + 1024);
  std::vector<float> v1621(752);
  std::copy(v1620.begin() + 0, v1620.begin() + 0 + 752, v1621.begin());
  std::vector<float> v1622(272);
  std::copy(v1620.begin() + 752, v1620.begin() + 752 + 272, v1622.begin());
  std::copy(v1621.begin(), v1621.end(), v86.begin() + 272);
  std::copy(v1622.begin(), v1622.end(), v86.begin() + 0);
  std::vector<double> v1625(std::begin(v86), std::end(v86));
  auto pt272_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt272_filled = v1625;
  pt272_filled.clear();
  pt272_filled.reserve(pt272_filled_n);
  for (auto i = 0; i < pt272_filled_n; ++i) {
    pt272_filled.push_back(v1625[i % v1625.size()]);
  }
  auto pt272 = cc->MakeCKKSPackedPlaintext(pt272_filled);
  const auto& ct559 = cc->EvalMult(ct, pt272);
  std::vector<float> v1626(std::begin(v28) + 273 * 512, std::begin(v28) + 273 * 512 + 1024);
  std::vector<float> v1627(752);
  std::copy(v1626.begin() + 0, v1626.begin() + 0 + 752, v1627.begin());
  std::vector<float> v1628(272);
  std::copy(v1626.begin() + 752, v1626.begin() + 752 + 272, v1628.begin());
  std::copy(v1627.begin(), v1627.end(), v86.begin() + 272);
  std::copy(v1628.begin(), v1628.end(), v86.begin() + 0);
  std::vector<double> v1631(std::begin(v86), std::end(v86));
  auto pt273_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt273_filled = v1631;
  pt273_filled.clear();
  pt273_filled.reserve(pt273_filled_n);
  for (auto i = 0; i < pt273_filled_n; ++i) {
    pt273_filled.push_back(v1631[i % v1631.size()]);
  }
  auto pt273 = cc->MakeCKKSPackedPlaintext(pt273_filled);
  const auto& ct560 = cc->EvalMult(ct2, pt273);
  std::vector<float> v1632(std::begin(v28) + 274 * 512, std::begin(v28) + 274 * 512 + 1024);
  std::vector<float> v1633(752);
  std::copy(v1632.begin() + 0, v1632.begin() + 0 + 752, v1633.begin());
  std::vector<float> v1634(272);
  std::copy(v1632.begin() + 752, v1632.begin() + 752 + 272, v1634.begin());
  std::copy(v1633.begin(), v1633.end(), v86.begin() + 272);
  std::copy(v1634.begin(), v1634.end(), v86.begin() + 0);
  std::vector<double> v1637(std::begin(v86), std::end(v86));
  auto pt274_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt274_filled = v1637;
  pt274_filled.clear();
  pt274_filled.reserve(pt274_filled_n);
  for (auto i = 0; i < pt274_filled_n; ++i) {
    pt274_filled.push_back(v1637[i % v1637.size()]);
  }
  auto pt274 = cc->MakeCKKSPackedPlaintext(pt274_filled);
  const auto& ct561 = cc->EvalMult(ct4, pt274);
  std::vector<float> v1638(std::begin(v28) + 275 * 512, std::begin(v28) + 275 * 512 + 1024);
  std::vector<float> v1639(752);
  std::copy(v1638.begin() + 0, v1638.begin() + 0 + 752, v1639.begin());
  std::vector<float> v1640(272);
  std::copy(v1638.begin() + 752, v1638.begin() + 752 + 272, v1640.begin());
  std::copy(v1639.begin(), v1639.end(), v86.begin() + 272);
  std::copy(v1640.begin(), v1640.end(), v86.begin() + 0);
  std::vector<double> v1643(std::begin(v86), std::end(v86));
  auto pt275_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt275_filled = v1643;
  pt275_filled.clear();
  pt275_filled.reserve(pt275_filled_n);
  for (auto i = 0; i < pt275_filled_n; ++i) {
    pt275_filled.push_back(v1643[i % v1643.size()]);
  }
  auto pt275 = cc->MakeCKKSPackedPlaintext(pt275_filled);
  const auto& ct562 = cc->EvalMult(ct6, pt275);
  std::vector<float> v1644(std::begin(v28) + 276 * 512, std::begin(v28) + 276 * 512 + 1024);
  std::vector<float> v1645(752);
  std::copy(v1644.begin() + 0, v1644.begin() + 0 + 752, v1645.begin());
  std::vector<float> v1646(272);
  std::copy(v1644.begin() + 752, v1644.begin() + 752 + 272, v1646.begin());
  std::copy(v1645.begin(), v1645.end(), v86.begin() + 272);
  std::copy(v1646.begin(), v1646.end(), v86.begin() + 0);
  std::vector<double> v1649(std::begin(v86), std::end(v86));
  auto pt276_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt276_filled = v1649;
  pt276_filled.clear();
  pt276_filled.reserve(pt276_filled_n);
  for (auto i = 0; i < pt276_filled_n; ++i) {
    pt276_filled.push_back(v1649[i % v1649.size()]);
  }
  auto pt276 = cc->MakeCKKSPackedPlaintext(pt276_filled);
  const auto& ct563 = cc->EvalMult(ct8, pt276);
  std::vector<float> v1650(std::begin(v28) + 277 * 512, std::begin(v28) + 277 * 512 + 1024);
  std::vector<float> v1651(752);
  std::copy(v1650.begin() + 0, v1650.begin() + 0 + 752, v1651.begin());
  std::vector<float> v1652(272);
  std::copy(v1650.begin() + 752, v1650.begin() + 752 + 272, v1652.begin());
  std::copy(v1651.begin(), v1651.end(), v86.begin() + 272);
  std::copy(v1652.begin(), v1652.end(), v86.begin() + 0);
  std::vector<double> v1655(std::begin(v86), std::end(v86));
  auto pt277_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt277_filled = v1655;
  pt277_filled.clear();
  pt277_filled.reserve(pt277_filled_n);
  for (auto i = 0; i < pt277_filled_n; ++i) {
    pt277_filled.push_back(v1655[i % v1655.size()]);
  }
  auto pt277 = cc->MakeCKKSPackedPlaintext(pt277_filled);
  const auto& ct564 = cc->EvalMult(ct10, pt277);
  std::vector<float> v1656(std::begin(v28) + 278 * 512, std::begin(v28) + 278 * 512 + 1024);
  std::vector<float> v1657(752);
  std::copy(v1656.begin() + 0, v1656.begin() + 0 + 752, v1657.begin());
  std::vector<float> v1658(272);
  std::copy(v1656.begin() + 752, v1656.begin() + 752 + 272, v1658.begin());
  std::copy(v1657.begin(), v1657.end(), v86.begin() + 272);
  std::copy(v1658.begin(), v1658.end(), v86.begin() + 0);
  std::vector<double> v1661(std::begin(v86), std::end(v86));
  auto pt278_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt278_filled = v1661;
  pt278_filled.clear();
  pt278_filled.reserve(pt278_filled_n);
  for (auto i = 0; i < pt278_filled_n; ++i) {
    pt278_filled.push_back(v1661[i % v1661.size()]);
  }
  auto pt278 = cc->MakeCKKSPackedPlaintext(pt278_filled);
  const auto& ct565 = cc->EvalMult(ct12, pt278);
  std::vector<float> v1662(std::begin(v28) + 279 * 512, std::begin(v28) + 279 * 512 + 1024);
  std::vector<float> v1663(752);
  std::copy(v1662.begin() + 0, v1662.begin() + 0 + 752, v1663.begin());
  std::vector<float> v1664(272);
  std::copy(v1662.begin() + 752, v1662.begin() + 752 + 272, v1664.begin());
  std::copy(v1663.begin(), v1663.end(), v86.begin() + 272);
  std::copy(v1664.begin(), v1664.end(), v86.begin() + 0);
  std::vector<double> v1667(std::begin(v86), std::end(v86));
  auto pt279_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt279_filled = v1667;
  pt279_filled.clear();
  pt279_filled.reserve(pt279_filled_n);
  for (auto i = 0; i < pt279_filled_n; ++i) {
    pt279_filled.push_back(v1667[i % v1667.size()]);
  }
  auto pt279 = cc->MakeCKKSPackedPlaintext(pt279_filled);
  const auto& ct566 = cc->EvalMult(ct14, pt279);
  std::vector<float> v1668(std::begin(v28) + 280 * 512, std::begin(v28) + 280 * 512 + 1024);
  std::vector<float> v1669(752);
  std::copy(v1668.begin() + 0, v1668.begin() + 0 + 752, v1669.begin());
  std::vector<float> v1670(272);
  std::copy(v1668.begin() + 752, v1668.begin() + 752 + 272, v1670.begin());
  std::copy(v1669.begin(), v1669.end(), v86.begin() + 272);
  std::copy(v1670.begin(), v1670.end(), v86.begin() + 0);
  std::vector<double> v1673(std::begin(v86), std::end(v86));
  auto pt280_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt280_filled = v1673;
  pt280_filled.clear();
  pt280_filled.reserve(pt280_filled_n);
  for (auto i = 0; i < pt280_filled_n; ++i) {
    pt280_filled.push_back(v1673[i % v1673.size()]);
  }
  auto pt280 = cc->MakeCKKSPackedPlaintext(pt280_filled);
  const auto& ct567 = cc->EvalMult(ct16, pt280);
  std::vector<float> v1674(std::begin(v28) + 281 * 512, std::begin(v28) + 281 * 512 + 1024);
  std::vector<float> v1675(752);
  std::copy(v1674.begin() + 0, v1674.begin() + 0 + 752, v1675.begin());
  std::vector<float> v1676(272);
  std::copy(v1674.begin() + 752, v1674.begin() + 752 + 272, v1676.begin());
  std::copy(v1675.begin(), v1675.end(), v86.begin() + 272);
  std::copy(v1676.begin(), v1676.end(), v86.begin() + 0);
  std::vector<double> v1679(std::begin(v86), std::end(v86));
  auto pt281_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt281_filled = v1679;
  pt281_filled.clear();
  pt281_filled.reserve(pt281_filled_n);
  for (auto i = 0; i < pt281_filled_n; ++i) {
    pt281_filled.push_back(v1679[i % v1679.size()]);
  }
  auto pt281 = cc->MakeCKKSPackedPlaintext(pt281_filled);
  const auto& ct568 = cc->EvalMult(ct18, pt281);
  std::vector<float> v1680(std::begin(v28) + 282 * 512, std::begin(v28) + 282 * 512 + 1024);
  std::vector<float> v1681(752);
  std::copy(v1680.begin() + 0, v1680.begin() + 0 + 752, v1681.begin());
  std::vector<float> v1682(272);
  std::copy(v1680.begin() + 752, v1680.begin() + 752 + 272, v1682.begin());
  std::copy(v1681.begin(), v1681.end(), v86.begin() + 272);
  std::copy(v1682.begin(), v1682.end(), v86.begin() + 0);
  std::vector<double> v1685(std::begin(v86), std::end(v86));
  auto pt282_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt282_filled = v1685;
  pt282_filled.clear();
  pt282_filled.reserve(pt282_filled_n);
  for (auto i = 0; i < pt282_filled_n; ++i) {
    pt282_filled.push_back(v1685[i % v1685.size()]);
  }
  auto pt282 = cc->MakeCKKSPackedPlaintext(pt282_filled);
  const auto& ct569 = cc->EvalMult(ct20, pt282);
  std::vector<float> v1686(std::begin(v28) + 283 * 512, std::begin(v28) + 283 * 512 + 1024);
  std::vector<float> v1687(752);
  std::copy(v1686.begin() + 0, v1686.begin() + 0 + 752, v1687.begin());
  std::vector<float> v1688(272);
  std::copy(v1686.begin() + 752, v1686.begin() + 752 + 272, v1688.begin());
  std::copy(v1687.begin(), v1687.end(), v86.begin() + 272);
  std::copy(v1688.begin(), v1688.end(), v86.begin() + 0);
  std::vector<double> v1691(std::begin(v86), std::end(v86));
  auto pt283_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt283_filled = v1691;
  pt283_filled.clear();
  pt283_filled.reserve(pt283_filled_n);
  for (auto i = 0; i < pt283_filled_n; ++i) {
    pt283_filled.push_back(v1691[i % v1691.size()]);
  }
  auto pt283 = cc->MakeCKKSPackedPlaintext(pt283_filled);
  const auto& ct570 = cc->EvalMult(ct22, pt283);
  std::vector<float> v1692(std::begin(v28) + 284 * 512, std::begin(v28) + 284 * 512 + 1024);
  std::vector<float> v1693(752);
  std::copy(v1692.begin() + 0, v1692.begin() + 0 + 752, v1693.begin());
  std::vector<float> v1694(272);
  std::copy(v1692.begin() + 752, v1692.begin() + 752 + 272, v1694.begin());
  std::copy(v1693.begin(), v1693.end(), v86.begin() + 272);
  std::copy(v1694.begin(), v1694.end(), v86.begin() + 0);
  std::vector<double> v1697(std::begin(v86), std::end(v86));
  auto pt284_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt284_filled = v1697;
  pt284_filled.clear();
  pt284_filled.reserve(pt284_filled_n);
  for (auto i = 0; i < pt284_filled_n; ++i) {
    pt284_filled.push_back(v1697[i % v1697.size()]);
  }
  auto pt284 = cc->MakeCKKSPackedPlaintext(pt284_filled);
  const auto& ct571 = cc->EvalMult(ct24, pt284);
  std::vector<float> v1698(std::begin(v28) + 285 * 512, std::begin(v28) + 285 * 512 + 1024);
  std::vector<float> v1699(752);
  std::copy(v1698.begin() + 0, v1698.begin() + 0 + 752, v1699.begin());
  std::vector<float> v1700(272);
  std::copy(v1698.begin() + 752, v1698.begin() + 752 + 272, v1700.begin());
  std::copy(v1699.begin(), v1699.end(), v86.begin() + 272);
  std::copy(v1700.begin(), v1700.end(), v86.begin() + 0);
  std::vector<double> v1703(std::begin(v86), std::end(v86));
  auto pt285_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt285_filled = v1703;
  pt285_filled.clear();
  pt285_filled.reserve(pt285_filled_n);
  for (auto i = 0; i < pt285_filled_n; ++i) {
    pt285_filled.push_back(v1703[i % v1703.size()]);
  }
  auto pt285 = cc->MakeCKKSPackedPlaintext(pt285_filled);
  const auto& ct572 = cc->EvalMult(ct26, pt285);
  std::vector<float> v1704(std::begin(v28) + 286 * 512, std::begin(v28) + 286 * 512 + 1024);
  std::vector<float> v1705(752);
  std::copy(v1704.begin() + 0, v1704.begin() + 0 + 752, v1705.begin());
  std::vector<float> v1706(272);
  std::copy(v1704.begin() + 752, v1704.begin() + 752 + 272, v1706.begin());
  std::copy(v1705.begin(), v1705.end(), v86.begin() + 272);
  std::copy(v1706.begin(), v1706.end(), v86.begin() + 0);
  std::vector<double> v1709(std::begin(v86), std::end(v86));
  auto pt286_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt286_filled = v1709;
  pt286_filled.clear();
  pt286_filled.reserve(pt286_filled_n);
  for (auto i = 0; i < pt286_filled_n; ++i) {
    pt286_filled.push_back(v1709[i % v1709.size()]);
  }
  auto pt286 = cc->MakeCKKSPackedPlaintext(pt286_filled);
  const auto& ct573 = cc->EvalMult(ct28, pt286);
  std::vector<float> v1710(std::begin(v28) + 287 * 512, std::begin(v28) + 287 * 512 + 1024);
  std::vector<float> v1711(752);
  std::copy(v1710.begin() + 0, v1710.begin() + 0 + 752, v1711.begin());
  std::vector<float> v1712(272);
  std::copy(v1710.begin() + 752, v1710.begin() + 752 + 272, v1712.begin());
  std::copy(v1711.begin(), v1711.end(), v86.begin() + 272);
  std::copy(v1712.begin(), v1712.end(), v86.begin() + 0);
  std::vector<double> v1715(std::begin(v86), std::end(v86));
  auto pt287_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt287_filled = v1715;
  pt287_filled.clear();
  pt287_filled.reserve(pt287_filled_n);
  for (auto i = 0; i < pt287_filled_n; ++i) {
    pt287_filled.push_back(v1715[i % v1715.size()]);
  }
  auto pt287 = cc->MakeCKKSPackedPlaintext(pt287_filled);
  const auto& ct574 = cc->EvalMult(ct30, pt287);
  const auto& ct575 = cc->EvalAdd(ct559, ct560);
  const auto& ct576 = cc->EvalAdd(ct561, ct562);
  const auto& ct577 = cc->EvalAdd(ct575, ct576);
  const auto& ct578 = cc->EvalAdd(ct563, ct564);
  const auto& ct579 = cc->EvalAdd(ct565, ct566);
  const auto& ct580 = cc->EvalAdd(ct578, ct579);
  const auto& ct581 = cc->EvalAdd(ct577, ct580);
  const auto& ct582 = cc->EvalAdd(ct567, ct568);
  const auto& ct583 = cc->EvalAdd(ct569, ct570);
  const auto& ct584 = cc->EvalAdd(ct582, ct583);
  const auto& ct585 = cc->EvalAdd(ct571, ct572);
  const auto& ct586 = cc->EvalAdd(ct573, ct574);
  const auto& ct587 = cc->EvalAdd(ct585, ct586);
  const auto& ct588 = cc->EvalAdd(ct584, ct587);
  const auto& ct589 = cc->EvalAdd(ct581, ct588);
  const auto& ct590 = cc->EvalRotate(ct589, 272);
  std::vector<float> v1716(std::begin(v28) + 288 * 512, std::begin(v28) + 288 * 512 + 1024);
  std::vector<float> v1717(736);
  std::copy(v1716.begin() + 0, v1716.begin() + 0 + 736, v1717.begin());
  std::vector<float> v1718(288);
  std::copy(v1716.begin() + 736, v1716.begin() + 736 + 288, v1718.begin());
  std::copy(v1717.begin(), v1717.end(), v86.begin() + 288);
  std::copy(v1718.begin(), v1718.end(), v86.begin() + 0);
  std::vector<double> v1721(std::begin(v86), std::end(v86));
  auto pt288_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt288_filled = v1721;
  pt288_filled.clear();
  pt288_filled.reserve(pt288_filled_n);
  for (auto i = 0; i < pt288_filled_n; ++i) {
    pt288_filled.push_back(v1721[i % v1721.size()]);
  }
  auto pt288 = cc->MakeCKKSPackedPlaintext(pt288_filled);
  const auto& ct591 = cc->EvalMult(ct, pt288);
  std::vector<float> v1722(std::begin(v28) + 289 * 512, std::begin(v28) + 289 * 512 + 1024);
  std::vector<float> v1723(736);
  std::copy(v1722.begin() + 0, v1722.begin() + 0 + 736, v1723.begin());
  std::vector<float> v1724(288);
  std::copy(v1722.begin() + 736, v1722.begin() + 736 + 288, v1724.begin());
  std::copy(v1723.begin(), v1723.end(), v86.begin() + 288);
  std::copy(v1724.begin(), v1724.end(), v86.begin() + 0);
  std::vector<double> v1727(std::begin(v86), std::end(v86));
  auto pt289_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt289_filled = v1727;
  pt289_filled.clear();
  pt289_filled.reserve(pt289_filled_n);
  for (auto i = 0; i < pt289_filled_n; ++i) {
    pt289_filled.push_back(v1727[i % v1727.size()]);
  }
  auto pt289 = cc->MakeCKKSPackedPlaintext(pt289_filled);
  const auto& ct592 = cc->EvalMult(ct2, pt289);
  std::vector<float> v1728(std::begin(v28) + 290 * 512, std::begin(v28) + 290 * 512 + 1024);
  std::vector<float> v1729(736);
  std::copy(v1728.begin() + 0, v1728.begin() + 0 + 736, v1729.begin());
  std::vector<float> v1730(288);
  std::copy(v1728.begin() + 736, v1728.begin() + 736 + 288, v1730.begin());
  std::copy(v1729.begin(), v1729.end(), v86.begin() + 288);
  std::copy(v1730.begin(), v1730.end(), v86.begin() + 0);
  std::vector<double> v1733(std::begin(v86), std::end(v86));
  auto pt290_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt290_filled = v1733;
  pt290_filled.clear();
  pt290_filled.reserve(pt290_filled_n);
  for (auto i = 0; i < pt290_filled_n; ++i) {
    pt290_filled.push_back(v1733[i % v1733.size()]);
  }
  auto pt290 = cc->MakeCKKSPackedPlaintext(pt290_filled);
  const auto& ct593 = cc->EvalMult(ct4, pt290);
  std::vector<float> v1734(std::begin(v28) + 291 * 512, std::begin(v28) + 291 * 512 + 1024);
  std::vector<float> v1735(736);
  std::copy(v1734.begin() + 0, v1734.begin() + 0 + 736, v1735.begin());
  std::vector<float> v1736(288);
  std::copy(v1734.begin() + 736, v1734.begin() + 736 + 288, v1736.begin());
  std::copy(v1735.begin(), v1735.end(), v86.begin() + 288);
  std::copy(v1736.begin(), v1736.end(), v86.begin() + 0);
  std::vector<double> v1739(std::begin(v86), std::end(v86));
  auto pt291_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt291_filled = v1739;
  pt291_filled.clear();
  pt291_filled.reserve(pt291_filled_n);
  for (auto i = 0; i < pt291_filled_n; ++i) {
    pt291_filled.push_back(v1739[i % v1739.size()]);
  }
  auto pt291 = cc->MakeCKKSPackedPlaintext(pt291_filled);
  const auto& ct594 = cc->EvalMult(ct6, pt291);
  std::vector<float> v1740(std::begin(v28) + 292 * 512, std::begin(v28) + 292 * 512 + 1024);
  std::vector<float> v1741(736);
  std::copy(v1740.begin() + 0, v1740.begin() + 0 + 736, v1741.begin());
  std::vector<float> v1742(288);
  std::copy(v1740.begin() + 736, v1740.begin() + 736 + 288, v1742.begin());
  std::copy(v1741.begin(), v1741.end(), v86.begin() + 288);
  std::copy(v1742.begin(), v1742.end(), v86.begin() + 0);
  std::vector<double> v1745(std::begin(v86), std::end(v86));
  auto pt292_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt292_filled = v1745;
  pt292_filled.clear();
  pt292_filled.reserve(pt292_filled_n);
  for (auto i = 0; i < pt292_filled_n; ++i) {
    pt292_filled.push_back(v1745[i % v1745.size()]);
  }
  auto pt292 = cc->MakeCKKSPackedPlaintext(pt292_filled);
  const auto& ct595 = cc->EvalMult(ct8, pt292);
  std::vector<float> v1746(std::begin(v28) + 293 * 512, std::begin(v28) + 293 * 512 + 1024);
  std::vector<float> v1747(736);
  std::copy(v1746.begin() + 0, v1746.begin() + 0 + 736, v1747.begin());
  std::vector<float> v1748(288);
  std::copy(v1746.begin() + 736, v1746.begin() + 736 + 288, v1748.begin());
  std::copy(v1747.begin(), v1747.end(), v86.begin() + 288);
  std::copy(v1748.begin(), v1748.end(), v86.begin() + 0);
  std::vector<double> v1751(std::begin(v86), std::end(v86));
  auto pt293_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt293_filled = v1751;
  pt293_filled.clear();
  pt293_filled.reserve(pt293_filled_n);
  for (auto i = 0; i < pt293_filled_n; ++i) {
    pt293_filled.push_back(v1751[i % v1751.size()]);
  }
  auto pt293 = cc->MakeCKKSPackedPlaintext(pt293_filled);
  const auto& ct596 = cc->EvalMult(ct10, pt293);
  std::vector<float> v1752(std::begin(v28) + 294 * 512, std::begin(v28) + 294 * 512 + 1024);
  std::vector<float> v1753(736);
  std::copy(v1752.begin() + 0, v1752.begin() + 0 + 736, v1753.begin());
  std::vector<float> v1754(288);
  std::copy(v1752.begin() + 736, v1752.begin() + 736 + 288, v1754.begin());
  std::copy(v1753.begin(), v1753.end(), v86.begin() + 288);
  std::copy(v1754.begin(), v1754.end(), v86.begin() + 0);
  std::vector<double> v1757(std::begin(v86), std::end(v86));
  auto pt294_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt294_filled = v1757;
  pt294_filled.clear();
  pt294_filled.reserve(pt294_filled_n);
  for (auto i = 0; i < pt294_filled_n; ++i) {
    pt294_filled.push_back(v1757[i % v1757.size()]);
  }
  auto pt294 = cc->MakeCKKSPackedPlaintext(pt294_filled);
  const auto& ct597 = cc->EvalMult(ct12, pt294);
  std::vector<float> v1758(std::begin(v28) + 295 * 512, std::begin(v28) + 295 * 512 + 1024);
  std::vector<float> v1759(736);
  std::copy(v1758.begin() + 0, v1758.begin() + 0 + 736, v1759.begin());
  std::vector<float> v1760(288);
  std::copy(v1758.begin() + 736, v1758.begin() + 736 + 288, v1760.begin());
  std::copy(v1759.begin(), v1759.end(), v86.begin() + 288);
  std::copy(v1760.begin(), v1760.end(), v86.begin() + 0);
  std::vector<double> v1763(std::begin(v86), std::end(v86));
  auto pt295_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt295_filled = v1763;
  pt295_filled.clear();
  pt295_filled.reserve(pt295_filled_n);
  for (auto i = 0; i < pt295_filled_n; ++i) {
    pt295_filled.push_back(v1763[i % v1763.size()]);
  }
  auto pt295 = cc->MakeCKKSPackedPlaintext(pt295_filled);
  const auto& ct598 = cc->EvalMult(ct14, pt295);
  std::vector<float> v1764(std::begin(v28) + 296 * 512, std::begin(v28) + 296 * 512 + 1024);
  std::vector<float> v1765(736);
  std::copy(v1764.begin() + 0, v1764.begin() + 0 + 736, v1765.begin());
  std::vector<float> v1766(288);
  std::copy(v1764.begin() + 736, v1764.begin() + 736 + 288, v1766.begin());
  std::copy(v1765.begin(), v1765.end(), v86.begin() + 288);
  std::copy(v1766.begin(), v1766.end(), v86.begin() + 0);
  std::vector<double> v1769(std::begin(v86), std::end(v86));
  auto pt296_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt296_filled = v1769;
  pt296_filled.clear();
  pt296_filled.reserve(pt296_filled_n);
  for (auto i = 0; i < pt296_filled_n; ++i) {
    pt296_filled.push_back(v1769[i % v1769.size()]);
  }
  auto pt296 = cc->MakeCKKSPackedPlaintext(pt296_filled);
  const auto& ct599 = cc->EvalMult(ct16, pt296);
  std::vector<float> v1770(std::begin(v28) + 297 * 512, std::begin(v28) + 297 * 512 + 1024);
  std::vector<float> v1771(736);
  std::copy(v1770.begin() + 0, v1770.begin() + 0 + 736, v1771.begin());
  std::vector<float> v1772(288);
  std::copy(v1770.begin() + 736, v1770.begin() + 736 + 288, v1772.begin());
  std::copy(v1771.begin(), v1771.end(), v86.begin() + 288);
  std::copy(v1772.begin(), v1772.end(), v86.begin() + 0);
  std::vector<double> v1775(std::begin(v86), std::end(v86));
  auto pt297_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt297_filled = v1775;
  pt297_filled.clear();
  pt297_filled.reserve(pt297_filled_n);
  for (auto i = 0; i < pt297_filled_n; ++i) {
    pt297_filled.push_back(v1775[i % v1775.size()]);
  }
  auto pt297 = cc->MakeCKKSPackedPlaintext(pt297_filled);
  const auto& ct600 = cc->EvalMult(ct18, pt297);
  std::vector<float> v1776(std::begin(v28) + 298 * 512, std::begin(v28) + 298 * 512 + 1024);
  std::vector<float> v1777(736);
  std::copy(v1776.begin() + 0, v1776.begin() + 0 + 736, v1777.begin());
  std::vector<float> v1778(288);
  std::copy(v1776.begin() + 736, v1776.begin() + 736 + 288, v1778.begin());
  std::copy(v1777.begin(), v1777.end(), v86.begin() + 288);
  std::copy(v1778.begin(), v1778.end(), v86.begin() + 0);
  std::vector<double> v1781(std::begin(v86), std::end(v86));
  auto pt298_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt298_filled = v1781;
  pt298_filled.clear();
  pt298_filled.reserve(pt298_filled_n);
  for (auto i = 0; i < pt298_filled_n; ++i) {
    pt298_filled.push_back(v1781[i % v1781.size()]);
  }
  auto pt298 = cc->MakeCKKSPackedPlaintext(pt298_filled);
  const auto& ct601 = cc->EvalMult(ct20, pt298);
  std::vector<float> v1782(std::begin(v28) + 299 * 512, std::begin(v28) + 299 * 512 + 1024);
  std::vector<float> v1783(736);
  std::copy(v1782.begin() + 0, v1782.begin() + 0 + 736, v1783.begin());
  std::vector<float> v1784(288);
  std::copy(v1782.begin() + 736, v1782.begin() + 736 + 288, v1784.begin());
  std::copy(v1783.begin(), v1783.end(), v86.begin() + 288);
  std::copy(v1784.begin(), v1784.end(), v86.begin() + 0);
  std::vector<double> v1787(std::begin(v86), std::end(v86));
  auto pt299_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt299_filled = v1787;
  pt299_filled.clear();
  pt299_filled.reserve(pt299_filled_n);
  for (auto i = 0; i < pt299_filled_n; ++i) {
    pt299_filled.push_back(v1787[i % v1787.size()]);
  }
  auto pt299 = cc->MakeCKKSPackedPlaintext(pt299_filled);
  const auto& ct602 = cc->EvalMult(ct22, pt299);
  std::vector<float> v1788(std::begin(v28) + 300 * 512, std::begin(v28) + 300 * 512 + 1024);
  std::vector<float> v1789(736);
  std::copy(v1788.begin() + 0, v1788.begin() + 0 + 736, v1789.begin());
  std::vector<float> v1790(288);
  std::copy(v1788.begin() + 736, v1788.begin() + 736 + 288, v1790.begin());
  std::copy(v1789.begin(), v1789.end(), v86.begin() + 288);
  std::copy(v1790.begin(), v1790.end(), v86.begin() + 0);
  std::vector<double> v1793(std::begin(v86), std::end(v86));
  auto pt300_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt300_filled = v1793;
  pt300_filled.clear();
  pt300_filled.reserve(pt300_filled_n);
  for (auto i = 0; i < pt300_filled_n; ++i) {
    pt300_filled.push_back(v1793[i % v1793.size()]);
  }
  auto pt300 = cc->MakeCKKSPackedPlaintext(pt300_filled);
  const auto& ct603 = cc->EvalMult(ct24, pt300);
  std::vector<float> v1794(std::begin(v28) + 301 * 512, std::begin(v28) + 301 * 512 + 1024);
  std::vector<float> v1795(736);
  std::copy(v1794.begin() + 0, v1794.begin() + 0 + 736, v1795.begin());
  std::vector<float> v1796(288);
  std::copy(v1794.begin() + 736, v1794.begin() + 736 + 288, v1796.begin());
  std::copy(v1795.begin(), v1795.end(), v86.begin() + 288);
  std::copy(v1796.begin(), v1796.end(), v86.begin() + 0);
  std::vector<double> v1799(std::begin(v86), std::end(v86));
  auto pt301_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt301_filled = v1799;
  pt301_filled.clear();
  pt301_filled.reserve(pt301_filled_n);
  for (auto i = 0; i < pt301_filled_n; ++i) {
    pt301_filled.push_back(v1799[i % v1799.size()]);
  }
  auto pt301 = cc->MakeCKKSPackedPlaintext(pt301_filled);
  const auto& ct604 = cc->EvalMult(ct26, pt301);
  std::vector<float> v1800(std::begin(v28) + 302 * 512, std::begin(v28) + 302 * 512 + 1024);
  std::vector<float> v1801(736);
  std::copy(v1800.begin() + 0, v1800.begin() + 0 + 736, v1801.begin());
  std::vector<float> v1802(288);
  std::copy(v1800.begin() + 736, v1800.begin() + 736 + 288, v1802.begin());
  std::copy(v1801.begin(), v1801.end(), v86.begin() + 288);
  std::copy(v1802.begin(), v1802.end(), v86.begin() + 0);
  std::vector<double> v1805(std::begin(v86), std::end(v86));
  auto pt302_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt302_filled = v1805;
  pt302_filled.clear();
  pt302_filled.reserve(pt302_filled_n);
  for (auto i = 0; i < pt302_filled_n; ++i) {
    pt302_filled.push_back(v1805[i % v1805.size()]);
  }
  auto pt302 = cc->MakeCKKSPackedPlaintext(pt302_filled);
  const auto& ct605 = cc->EvalMult(ct28, pt302);
  std::vector<float> v1806(std::begin(v28) + 303 * 512, std::begin(v28) + 303 * 512 + 1024);
  std::vector<float> v1807(736);
  std::copy(v1806.begin() + 0, v1806.begin() + 0 + 736, v1807.begin());
  std::vector<float> v1808(288);
  std::copy(v1806.begin() + 736, v1806.begin() + 736 + 288, v1808.begin());
  std::copy(v1807.begin(), v1807.end(), v86.begin() + 288);
  std::copy(v1808.begin(), v1808.end(), v86.begin() + 0);
  std::vector<double> v1811(std::begin(v86), std::end(v86));
  auto pt303_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt303_filled = v1811;
  pt303_filled.clear();
  pt303_filled.reserve(pt303_filled_n);
  for (auto i = 0; i < pt303_filled_n; ++i) {
    pt303_filled.push_back(v1811[i % v1811.size()]);
  }
  auto pt303 = cc->MakeCKKSPackedPlaintext(pt303_filled);
  const auto& ct606 = cc->EvalMult(ct30, pt303);
  const auto& ct607 = cc->EvalAdd(ct591, ct592);
  const auto& ct608 = cc->EvalAdd(ct593, ct594);
  const auto& ct609 = cc->EvalAdd(ct607, ct608);
  const auto& ct610 = cc->EvalAdd(ct595, ct596);
  const auto& ct611 = cc->EvalAdd(ct597, ct598);
  const auto& ct612 = cc->EvalAdd(ct610, ct611);
  const auto& ct613 = cc->EvalAdd(ct609, ct612);
  const auto& ct614 = cc->EvalAdd(ct599, ct600);
  const auto& ct615 = cc->EvalAdd(ct601, ct602);
  const auto& ct616 = cc->EvalAdd(ct614, ct615);
  const auto& ct617 = cc->EvalAdd(ct603, ct604);
  const auto& ct618 = cc->EvalAdd(ct605, ct606);
  const auto& ct619 = cc->EvalAdd(ct617, ct618);
  const auto& ct620 = cc->EvalAdd(ct616, ct619);
  const auto& ct621 = cc->EvalAdd(ct613, ct620);
  const auto& ct622 = cc->EvalRotate(ct621, 288);
  std::vector<float> v1812(std::begin(v28) + 304 * 512, std::begin(v28) + 304 * 512 + 1024);
  std::vector<float> v1813(720);
  std::copy(v1812.begin() + 0, v1812.begin() + 0 + 720, v1813.begin());
  std::vector<float> v1814(304);
  std::copy(v1812.begin() + 720, v1812.begin() + 720 + 304, v1814.begin());
  std::copy(v1813.begin(), v1813.end(), v86.begin() + 304);
  std::copy(v1814.begin(), v1814.end(), v86.begin() + 0);
  std::vector<double> v1817(std::begin(v86), std::end(v86));
  auto pt304_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt304_filled = v1817;
  pt304_filled.clear();
  pt304_filled.reserve(pt304_filled_n);
  for (auto i = 0; i < pt304_filled_n; ++i) {
    pt304_filled.push_back(v1817[i % v1817.size()]);
  }
  auto pt304 = cc->MakeCKKSPackedPlaintext(pt304_filled);
  const auto& ct623 = cc->EvalMult(ct, pt304);
  std::vector<float> v1818(std::begin(v28) + 305 * 512, std::begin(v28) + 305 * 512 + 1024);
  std::vector<float> v1819(720);
  std::copy(v1818.begin() + 0, v1818.begin() + 0 + 720, v1819.begin());
  std::vector<float> v1820(304);
  std::copy(v1818.begin() + 720, v1818.begin() + 720 + 304, v1820.begin());
  std::copy(v1819.begin(), v1819.end(), v86.begin() + 304);
  std::copy(v1820.begin(), v1820.end(), v86.begin() + 0);
  std::vector<double> v1823(std::begin(v86), std::end(v86));
  auto pt305_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt305_filled = v1823;
  pt305_filled.clear();
  pt305_filled.reserve(pt305_filled_n);
  for (auto i = 0; i < pt305_filled_n; ++i) {
    pt305_filled.push_back(v1823[i % v1823.size()]);
  }
  auto pt305 = cc->MakeCKKSPackedPlaintext(pt305_filled);
  const auto& ct624 = cc->EvalMult(ct2, pt305);
  std::vector<float> v1824(std::begin(v28) + 306 * 512, std::begin(v28) + 306 * 512 + 1024);
  std::vector<float> v1825(720);
  std::copy(v1824.begin() + 0, v1824.begin() + 0 + 720, v1825.begin());
  std::vector<float> v1826(304);
  std::copy(v1824.begin() + 720, v1824.begin() + 720 + 304, v1826.begin());
  std::copy(v1825.begin(), v1825.end(), v86.begin() + 304);
  std::copy(v1826.begin(), v1826.end(), v86.begin() + 0);
  std::vector<double> v1829(std::begin(v86), std::end(v86));
  auto pt306_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt306_filled = v1829;
  pt306_filled.clear();
  pt306_filled.reserve(pt306_filled_n);
  for (auto i = 0; i < pt306_filled_n; ++i) {
    pt306_filled.push_back(v1829[i % v1829.size()]);
  }
  auto pt306 = cc->MakeCKKSPackedPlaintext(pt306_filled);
  const auto& ct625 = cc->EvalMult(ct4, pt306);
  std::vector<float> v1830(std::begin(v28) + 307 * 512, std::begin(v28) + 307 * 512 + 1024);
  std::vector<float> v1831(720);
  std::copy(v1830.begin() + 0, v1830.begin() + 0 + 720, v1831.begin());
  std::vector<float> v1832(304);
  std::copy(v1830.begin() + 720, v1830.begin() + 720 + 304, v1832.begin());
  std::copy(v1831.begin(), v1831.end(), v86.begin() + 304);
  std::copy(v1832.begin(), v1832.end(), v86.begin() + 0);
  std::vector<double> v1835(std::begin(v86), std::end(v86));
  auto pt307_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt307_filled = v1835;
  pt307_filled.clear();
  pt307_filled.reserve(pt307_filled_n);
  for (auto i = 0; i < pt307_filled_n; ++i) {
    pt307_filled.push_back(v1835[i % v1835.size()]);
  }
  auto pt307 = cc->MakeCKKSPackedPlaintext(pt307_filled);
  const auto& ct626 = cc->EvalMult(ct6, pt307);
  std::vector<float> v1836(std::begin(v28) + 308 * 512, std::begin(v28) + 308 * 512 + 1024);
  std::vector<float> v1837(720);
  std::copy(v1836.begin() + 0, v1836.begin() + 0 + 720, v1837.begin());
  std::vector<float> v1838(304);
  std::copy(v1836.begin() + 720, v1836.begin() + 720 + 304, v1838.begin());
  std::copy(v1837.begin(), v1837.end(), v86.begin() + 304);
  std::copy(v1838.begin(), v1838.end(), v86.begin() + 0);
  std::vector<double> v1841(std::begin(v86), std::end(v86));
  auto pt308_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt308_filled = v1841;
  pt308_filled.clear();
  pt308_filled.reserve(pt308_filled_n);
  for (auto i = 0; i < pt308_filled_n; ++i) {
    pt308_filled.push_back(v1841[i % v1841.size()]);
  }
  auto pt308 = cc->MakeCKKSPackedPlaintext(pt308_filled);
  const auto& ct627 = cc->EvalMult(ct8, pt308);
  std::vector<float> v1842(std::begin(v28) + 309 * 512, std::begin(v28) + 309 * 512 + 1024);
  std::vector<float> v1843(720);
  std::copy(v1842.begin() + 0, v1842.begin() + 0 + 720, v1843.begin());
  std::vector<float> v1844(304);
  std::copy(v1842.begin() + 720, v1842.begin() + 720 + 304, v1844.begin());
  std::copy(v1843.begin(), v1843.end(), v86.begin() + 304);
  std::copy(v1844.begin(), v1844.end(), v86.begin() + 0);
  std::vector<double> v1847(std::begin(v86), std::end(v86));
  auto pt309_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt309_filled = v1847;
  pt309_filled.clear();
  pt309_filled.reserve(pt309_filled_n);
  for (auto i = 0; i < pt309_filled_n; ++i) {
    pt309_filled.push_back(v1847[i % v1847.size()]);
  }
  auto pt309 = cc->MakeCKKSPackedPlaintext(pt309_filled);
  const auto& ct628 = cc->EvalMult(ct10, pt309);
  std::vector<float> v1848(std::begin(v28) + 310 * 512, std::begin(v28) + 310 * 512 + 1024);
  std::vector<float> v1849(720);
  std::copy(v1848.begin() + 0, v1848.begin() + 0 + 720, v1849.begin());
  std::vector<float> v1850(304);
  std::copy(v1848.begin() + 720, v1848.begin() + 720 + 304, v1850.begin());
  std::copy(v1849.begin(), v1849.end(), v86.begin() + 304);
  std::copy(v1850.begin(), v1850.end(), v86.begin() + 0);
  std::vector<double> v1853(std::begin(v86), std::end(v86));
  auto pt310_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt310_filled = v1853;
  pt310_filled.clear();
  pt310_filled.reserve(pt310_filled_n);
  for (auto i = 0; i < pt310_filled_n; ++i) {
    pt310_filled.push_back(v1853[i % v1853.size()]);
  }
  auto pt310 = cc->MakeCKKSPackedPlaintext(pt310_filled);
  const auto& ct629 = cc->EvalMult(ct12, pt310);
  std::vector<float> v1854(std::begin(v28) + 311 * 512, std::begin(v28) + 311 * 512 + 1024);
  std::vector<float> v1855(720);
  std::copy(v1854.begin() + 0, v1854.begin() + 0 + 720, v1855.begin());
  std::vector<float> v1856(304);
  std::copy(v1854.begin() + 720, v1854.begin() + 720 + 304, v1856.begin());
  std::copy(v1855.begin(), v1855.end(), v86.begin() + 304);
  std::copy(v1856.begin(), v1856.end(), v86.begin() + 0);
  std::vector<double> v1859(std::begin(v86), std::end(v86));
  auto pt311_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt311_filled = v1859;
  pt311_filled.clear();
  pt311_filled.reserve(pt311_filled_n);
  for (auto i = 0; i < pt311_filled_n; ++i) {
    pt311_filled.push_back(v1859[i % v1859.size()]);
  }
  auto pt311 = cc->MakeCKKSPackedPlaintext(pt311_filled);
  const auto& ct630 = cc->EvalMult(ct14, pt311);
  std::vector<float> v1860(std::begin(v28) + 312 * 512, std::begin(v28) + 312 * 512 + 1024);
  std::vector<float> v1861(720);
  std::copy(v1860.begin() + 0, v1860.begin() + 0 + 720, v1861.begin());
  std::vector<float> v1862(304);
  std::copy(v1860.begin() + 720, v1860.begin() + 720 + 304, v1862.begin());
  std::copy(v1861.begin(), v1861.end(), v86.begin() + 304);
  std::copy(v1862.begin(), v1862.end(), v86.begin() + 0);
  std::vector<double> v1865(std::begin(v86), std::end(v86));
  auto pt312_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt312_filled = v1865;
  pt312_filled.clear();
  pt312_filled.reserve(pt312_filled_n);
  for (auto i = 0; i < pt312_filled_n; ++i) {
    pt312_filled.push_back(v1865[i % v1865.size()]);
  }
  auto pt312 = cc->MakeCKKSPackedPlaintext(pt312_filled);
  const auto& ct631 = cc->EvalMult(ct16, pt312);
  std::vector<float> v1866(std::begin(v28) + 313 * 512, std::begin(v28) + 313 * 512 + 1024);
  std::vector<float> v1867(720);
  std::copy(v1866.begin() + 0, v1866.begin() + 0 + 720, v1867.begin());
  std::vector<float> v1868(304);
  std::copy(v1866.begin() + 720, v1866.begin() + 720 + 304, v1868.begin());
  std::copy(v1867.begin(), v1867.end(), v86.begin() + 304);
  std::copy(v1868.begin(), v1868.end(), v86.begin() + 0);
  std::vector<double> v1871(std::begin(v86), std::end(v86));
  auto pt313_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt313_filled = v1871;
  pt313_filled.clear();
  pt313_filled.reserve(pt313_filled_n);
  for (auto i = 0; i < pt313_filled_n; ++i) {
    pt313_filled.push_back(v1871[i % v1871.size()]);
  }
  auto pt313 = cc->MakeCKKSPackedPlaintext(pt313_filled);
  const auto& ct632 = cc->EvalMult(ct18, pt313);
  std::vector<float> v1872(std::begin(v28) + 314 * 512, std::begin(v28) + 314 * 512 + 1024);
  std::vector<float> v1873(720);
  std::copy(v1872.begin() + 0, v1872.begin() + 0 + 720, v1873.begin());
  std::vector<float> v1874(304);
  std::copy(v1872.begin() + 720, v1872.begin() + 720 + 304, v1874.begin());
  std::copy(v1873.begin(), v1873.end(), v86.begin() + 304);
  std::copy(v1874.begin(), v1874.end(), v86.begin() + 0);
  std::vector<double> v1877(std::begin(v86), std::end(v86));
  auto pt314_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt314_filled = v1877;
  pt314_filled.clear();
  pt314_filled.reserve(pt314_filled_n);
  for (auto i = 0; i < pt314_filled_n; ++i) {
    pt314_filled.push_back(v1877[i % v1877.size()]);
  }
  auto pt314 = cc->MakeCKKSPackedPlaintext(pt314_filled);
  const auto& ct633 = cc->EvalMult(ct20, pt314);
  std::vector<float> v1878(std::begin(v28) + 315 * 512, std::begin(v28) + 315 * 512 + 1024);
  std::vector<float> v1879(720);
  std::copy(v1878.begin() + 0, v1878.begin() + 0 + 720, v1879.begin());
  std::vector<float> v1880(304);
  std::copy(v1878.begin() + 720, v1878.begin() + 720 + 304, v1880.begin());
  std::copy(v1879.begin(), v1879.end(), v86.begin() + 304);
  std::copy(v1880.begin(), v1880.end(), v86.begin() + 0);
  std::vector<double> v1883(std::begin(v86), std::end(v86));
  auto pt315_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt315_filled = v1883;
  pt315_filled.clear();
  pt315_filled.reserve(pt315_filled_n);
  for (auto i = 0; i < pt315_filled_n; ++i) {
    pt315_filled.push_back(v1883[i % v1883.size()]);
  }
  auto pt315 = cc->MakeCKKSPackedPlaintext(pt315_filled);
  const auto& ct634 = cc->EvalMult(ct22, pt315);
  std::vector<float> v1884(std::begin(v28) + 316 * 512, std::begin(v28) + 316 * 512 + 1024);
  std::vector<float> v1885(720);
  std::copy(v1884.begin() + 0, v1884.begin() + 0 + 720, v1885.begin());
  std::vector<float> v1886(304);
  std::copy(v1884.begin() + 720, v1884.begin() + 720 + 304, v1886.begin());
  std::copy(v1885.begin(), v1885.end(), v86.begin() + 304);
  std::copy(v1886.begin(), v1886.end(), v86.begin() + 0);
  std::vector<double> v1889(std::begin(v86), std::end(v86));
  auto pt316_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt316_filled = v1889;
  pt316_filled.clear();
  pt316_filled.reserve(pt316_filled_n);
  for (auto i = 0; i < pt316_filled_n; ++i) {
    pt316_filled.push_back(v1889[i % v1889.size()]);
  }
  auto pt316 = cc->MakeCKKSPackedPlaintext(pt316_filled);
  const auto& ct635 = cc->EvalMult(ct24, pt316);
  std::vector<float> v1890(std::begin(v28) + 317 * 512, std::begin(v28) + 317 * 512 + 1024);
  std::vector<float> v1891(720);
  std::copy(v1890.begin() + 0, v1890.begin() + 0 + 720, v1891.begin());
  std::vector<float> v1892(304);
  std::copy(v1890.begin() + 720, v1890.begin() + 720 + 304, v1892.begin());
  std::copy(v1891.begin(), v1891.end(), v86.begin() + 304);
  std::copy(v1892.begin(), v1892.end(), v86.begin() + 0);
  std::vector<double> v1895(std::begin(v86), std::end(v86));
  auto pt317_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt317_filled = v1895;
  pt317_filled.clear();
  pt317_filled.reserve(pt317_filled_n);
  for (auto i = 0; i < pt317_filled_n; ++i) {
    pt317_filled.push_back(v1895[i % v1895.size()]);
  }
  auto pt317 = cc->MakeCKKSPackedPlaintext(pt317_filled);
  const auto& ct636 = cc->EvalMult(ct26, pt317);
  std::vector<float> v1896(std::begin(v28) + 318 * 512, std::begin(v28) + 318 * 512 + 1024);
  std::vector<float> v1897(720);
  std::copy(v1896.begin() + 0, v1896.begin() + 0 + 720, v1897.begin());
  std::vector<float> v1898(304);
  std::copy(v1896.begin() + 720, v1896.begin() + 720 + 304, v1898.begin());
  std::copy(v1897.begin(), v1897.end(), v86.begin() + 304);
  std::copy(v1898.begin(), v1898.end(), v86.begin() + 0);
  std::vector<double> v1901(std::begin(v86), std::end(v86));
  auto pt318_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt318_filled = v1901;
  pt318_filled.clear();
  pt318_filled.reserve(pt318_filled_n);
  for (auto i = 0; i < pt318_filled_n; ++i) {
    pt318_filled.push_back(v1901[i % v1901.size()]);
  }
  auto pt318 = cc->MakeCKKSPackedPlaintext(pt318_filled);
  const auto& ct637 = cc->EvalMult(ct28, pt318);
  std::vector<float> v1902(std::begin(v28) + 319 * 512, std::begin(v28) + 319 * 512 + 1024);
  std::vector<float> v1903(720);
  std::copy(v1902.begin() + 0, v1902.begin() + 0 + 720, v1903.begin());
  std::vector<float> v1904(304);
  std::copy(v1902.begin() + 720, v1902.begin() + 720 + 304, v1904.begin());
  std::copy(v1903.begin(), v1903.end(), v86.begin() + 304);
  std::copy(v1904.begin(), v1904.end(), v86.begin() + 0);
  std::vector<double> v1907(std::begin(v86), std::end(v86));
  auto pt319_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt319_filled = v1907;
  pt319_filled.clear();
  pt319_filled.reserve(pt319_filled_n);
  for (auto i = 0; i < pt319_filled_n; ++i) {
    pt319_filled.push_back(v1907[i % v1907.size()]);
  }
  auto pt319 = cc->MakeCKKSPackedPlaintext(pt319_filled);
  const auto& ct638 = cc->EvalMult(ct30, pt319);
  const auto& ct639 = cc->EvalAdd(ct623, ct624);
  const auto& ct640 = cc->EvalAdd(ct625, ct626);
  const auto& ct641 = cc->EvalAdd(ct639, ct640);
  const auto& ct642 = cc->EvalAdd(ct627, ct628);
  const auto& ct643 = cc->EvalAdd(ct629, ct630);
  const auto& ct644 = cc->EvalAdd(ct642, ct643);
  const auto& ct645 = cc->EvalAdd(ct641, ct644);
  const auto& ct646 = cc->EvalAdd(ct631, ct632);
  const auto& ct647 = cc->EvalAdd(ct633, ct634);
  const auto& ct648 = cc->EvalAdd(ct646, ct647);
  const auto& ct649 = cc->EvalAdd(ct635, ct636);
  const auto& ct650 = cc->EvalAdd(ct637, ct638);
  const auto& ct651 = cc->EvalAdd(ct649, ct650);
  const auto& ct652 = cc->EvalAdd(ct648, ct651);
  const auto& ct653 = cc->EvalAdd(ct645, ct652);
  const auto& ct654 = cc->EvalRotate(ct653, 304);
  std::vector<float> v1908(std::begin(v28) + 320 * 512, std::begin(v28) + 320 * 512 + 1024);
  std::vector<float> v1909(704);
  std::copy(v1908.begin() + 0, v1908.begin() + 0 + 704, v1909.begin());
  std::vector<float> v1910(320);
  std::copy(v1908.begin() + 704, v1908.begin() + 704 + 320, v1910.begin());
  std::copy(v1909.begin(), v1909.end(), v86.begin() + 320);
  std::copy(v1910.begin(), v1910.end(), v86.begin() + 0);
  std::vector<double> v1913(std::begin(v86), std::end(v86));
  auto pt320_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt320_filled = v1913;
  pt320_filled.clear();
  pt320_filled.reserve(pt320_filled_n);
  for (auto i = 0; i < pt320_filled_n; ++i) {
    pt320_filled.push_back(v1913[i % v1913.size()]);
  }
  auto pt320 = cc->MakeCKKSPackedPlaintext(pt320_filled);
  const auto& ct655 = cc->EvalMult(ct, pt320);
  std::vector<float> v1914(std::begin(v28) + 321 * 512, std::begin(v28) + 321 * 512 + 1024);
  std::vector<float> v1915(704);
  std::copy(v1914.begin() + 0, v1914.begin() + 0 + 704, v1915.begin());
  std::vector<float> v1916(320);
  std::copy(v1914.begin() + 704, v1914.begin() + 704 + 320, v1916.begin());
  std::copy(v1915.begin(), v1915.end(), v86.begin() + 320);
  std::copy(v1916.begin(), v1916.end(), v86.begin() + 0);
  std::vector<double> v1919(std::begin(v86), std::end(v86));
  auto pt321_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt321_filled = v1919;
  pt321_filled.clear();
  pt321_filled.reserve(pt321_filled_n);
  for (auto i = 0; i < pt321_filled_n; ++i) {
    pt321_filled.push_back(v1919[i % v1919.size()]);
  }
  auto pt321 = cc->MakeCKKSPackedPlaintext(pt321_filled);
  const auto& ct656 = cc->EvalMult(ct2, pt321);
  std::vector<float> v1920(std::begin(v28) + 322 * 512, std::begin(v28) + 322 * 512 + 1024);
  std::vector<float> v1921(704);
  std::copy(v1920.begin() + 0, v1920.begin() + 0 + 704, v1921.begin());
  std::vector<float> v1922(320);
  std::copy(v1920.begin() + 704, v1920.begin() + 704 + 320, v1922.begin());
  std::copy(v1921.begin(), v1921.end(), v86.begin() + 320);
  std::copy(v1922.begin(), v1922.end(), v86.begin() + 0);
  std::vector<double> v1925(std::begin(v86), std::end(v86));
  auto pt322_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt322_filled = v1925;
  pt322_filled.clear();
  pt322_filled.reserve(pt322_filled_n);
  for (auto i = 0; i < pt322_filled_n; ++i) {
    pt322_filled.push_back(v1925[i % v1925.size()]);
  }
  auto pt322 = cc->MakeCKKSPackedPlaintext(pt322_filled);
  const auto& ct657 = cc->EvalMult(ct4, pt322);
  std::vector<float> v1926(std::begin(v28) + 323 * 512, std::begin(v28) + 323 * 512 + 1024);
  std::vector<float> v1927(704);
  std::copy(v1926.begin() + 0, v1926.begin() + 0 + 704, v1927.begin());
  std::vector<float> v1928(320);
  std::copy(v1926.begin() + 704, v1926.begin() + 704 + 320, v1928.begin());
  std::copy(v1927.begin(), v1927.end(), v86.begin() + 320);
  std::copy(v1928.begin(), v1928.end(), v86.begin() + 0);
  std::vector<double> v1931(std::begin(v86), std::end(v86));
  auto pt323_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt323_filled = v1931;
  pt323_filled.clear();
  pt323_filled.reserve(pt323_filled_n);
  for (auto i = 0; i < pt323_filled_n; ++i) {
    pt323_filled.push_back(v1931[i % v1931.size()]);
  }
  auto pt323 = cc->MakeCKKSPackedPlaintext(pt323_filled);
  const auto& ct658 = cc->EvalMult(ct6, pt323);
  std::vector<float> v1932(std::begin(v28) + 324 * 512, std::begin(v28) + 324 * 512 + 1024);
  std::vector<float> v1933(704);
  std::copy(v1932.begin() + 0, v1932.begin() + 0 + 704, v1933.begin());
  std::vector<float> v1934(320);
  std::copy(v1932.begin() + 704, v1932.begin() + 704 + 320, v1934.begin());
  std::copy(v1933.begin(), v1933.end(), v86.begin() + 320);
  std::copy(v1934.begin(), v1934.end(), v86.begin() + 0);
  std::vector<double> v1937(std::begin(v86), std::end(v86));
  auto pt324_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt324_filled = v1937;
  pt324_filled.clear();
  pt324_filled.reserve(pt324_filled_n);
  for (auto i = 0; i < pt324_filled_n; ++i) {
    pt324_filled.push_back(v1937[i % v1937.size()]);
  }
  auto pt324 = cc->MakeCKKSPackedPlaintext(pt324_filled);
  const auto& ct659 = cc->EvalMult(ct8, pt324);
  std::vector<float> v1938(std::begin(v28) + 325 * 512, std::begin(v28) + 325 * 512 + 1024);
  std::vector<float> v1939(704);
  std::copy(v1938.begin() + 0, v1938.begin() + 0 + 704, v1939.begin());
  std::vector<float> v1940(320);
  std::copy(v1938.begin() + 704, v1938.begin() + 704 + 320, v1940.begin());
  std::copy(v1939.begin(), v1939.end(), v86.begin() + 320);
  std::copy(v1940.begin(), v1940.end(), v86.begin() + 0);
  std::vector<double> v1943(std::begin(v86), std::end(v86));
  auto pt325_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt325_filled = v1943;
  pt325_filled.clear();
  pt325_filled.reserve(pt325_filled_n);
  for (auto i = 0; i < pt325_filled_n; ++i) {
    pt325_filled.push_back(v1943[i % v1943.size()]);
  }
  auto pt325 = cc->MakeCKKSPackedPlaintext(pt325_filled);
  const auto& ct660 = cc->EvalMult(ct10, pt325);
  std::vector<float> v1944(std::begin(v28) + 326 * 512, std::begin(v28) + 326 * 512 + 1024);
  std::vector<float> v1945(704);
  std::copy(v1944.begin() + 0, v1944.begin() + 0 + 704, v1945.begin());
  std::vector<float> v1946(320);
  std::copy(v1944.begin() + 704, v1944.begin() + 704 + 320, v1946.begin());
  std::copy(v1945.begin(), v1945.end(), v86.begin() + 320);
  std::copy(v1946.begin(), v1946.end(), v86.begin() + 0);
  std::vector<double> v1949(std::begin(v86), std::end(v86));
  auto pt326_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt326_filled = v1949;
  pt326_filled.clear();
  pt326_filled.reserve(pt326_filled_n);
  for (auto i = 0; i < pt326_filled_n; ++i) {
    pt326_filled.push_back(v1949[i % v1949.size()]);
  }
  auto pt326 = cc->MakeCKKSPackedPlaintext(pt326_filled);
  const auto& ct661 = cc->EvalMult(ct12, pt326);
  std::vector<float> v1950(std::begin(v28) + 327 * 512, std::begin(v28) + 327 * 512 + 1024);
  std::vector<float> v1951(704);
  std::copy(v1950.begin() + 0, v1950.begin() + 0 + 704, v1951.begin());
  std::vector<float> v1952(320);
  std::copy(v1950.begin() + 704, v1950.begin() + 704 + 320, v1952.begin());
  std::copy(v1951.begin(), v1951.end(), v86.begin() + 320);
  std::copy(v1952.begin(), v1952.end(), v86.begin() + 0);
  std::vector<double> v1955(std::begin(v86), std::end(v86));
  auto pt327_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt327_filled = v1955;
  pt327_filled.clear();
  pt327_filled.reserve(pt327_filled_n);
  for (auto i = 0; i < pt327_filled_n; ++i) {
    pt327_filled.push_back(v1955[i % v1955.size()]);
  }
  auto pt327 = cc->MakeCKKSPackedPlaintext(pt327_filled);
  const auto& ct662 = cc->EvalMult(ct14, pt327);
  std::vector<float> v1956(std::begin(v28) + 328 * 512, std::begin(v28) + 328 * 512 + 1024);
  std::vector<float> v1957(704);
  std::copy(v1956.begin() + 0, v1956.begin() + 0 + 704, v1957.begin());
  std::vector<float> v1958(320);
  std::copy(v1956.begin() + 704, v1956.begin() + 704 + 320, v1958.begin());
  std::copy(v1957.begin(), v1957.end(), v86.begin() + 320);
  std::copy(v1958.begin(), v1958.end(), v86.begin() + 0);
  std::vector<double> v1961(std::begin(v86), std::end(v86));
  auto pt328_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt328_filled = v1961;
  pt328_filled.clear();
  pt328_filled.reserve(pt328_filled_n);
  for (auto i = 0; i < pt328_filled_n; ++i) {
    pt328_filled.push_back(v1961[i % v1961.size()]);
  }
  auto pt328 = cc->MakeCKKSPackedPlaintext(pt328_filled);
  const auto& ct663 = cc->EvalMult(ct16, pt328);
  std::vector<float> v1962(std::begin(v28) + 329 * 512, std::begin(v28) + 329 * 512 + 1024);
  std::vector<float> v1963(704);
  std::copy(v1962.begin() + 0, v1962.begin() + 0 + 704, v1963.begin());
  std::vector<float> v1964(320);
  std::copy(v1962.begin() + 704, v1962.begin() + 704 + 320, v1964.begin());
  std::copy(v1963.begin(), v1963.end(), v86.begin() + 320);
  std::copy(v1964.begin(), v1964.end(), v86.begin() + 0);
  std::vector<double> v1967(std::begin(v86), std::end(v86));
  auto pt329_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt329_filled = v1967;
  pt329_filled.clear();
  pt329_filled.reserve(pt329_filled_n);
  for (auto i = 0; i < pt329_filled_n; ++i) {
    pt329_filled.push_back(v1967[i % v1967.size()]);
  }
  auto pt329 = cc->MakeCKKSPackedPlaintext(pt329_filled);
  const auto& ct664 = cc->EvalMult(ct18, pt329);
  std::vector<float> v1968(std::begin(v28) + 330 * 512, std::begin(v28) + 330 * 512 + 1024);
  std::vector<float> v1969(704);
  std::copy(v1968.begin() + 0, v1968.begin() + 0 + 704, v1969.begin());
  std::vector<float> v1970(320);
  std::copy(v1968.begin() + 704, v1968.begin() + 704 + 320, v1970.begin());
  std::copy(v1969.begin(), v1969.end(), v86.begin() + 320);
  std::copy(v1970.begin(), v1970.end(), v86.begin() + 0);
  std::vector<double> v1973(std::begin(v86), std::end(v86));
  auto pt330_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt330_filled = v1973;
  pt330_filled.clear();
  pt330_filled.reserve(pt330_filled_n);
  for (auto i = 0; i < pt330_filled_n; ++i) {
    pt330_filled.push_back(v1973[i % v1973.size()]);
  }
  auto pt330 = cc->MakeCKKSPackedPlaintext(pt330_filled);
  const auto& ct665 = cc->EvalMult(ct20, pt330);
  std::vector<float> v1974(std::begin(v28) + 331 * 512, std::begin(v28) + 331 * 512 + 1024);
  std::vector<float> v1975(704);
  std::copy(v1974.begin() + 0, v1974.begin() + 0 + 704, v1975.begin());
  std::vector<float> v1976(320);
  std::copy(v1974.begin() + 704, v1974.begin() + 704 + 320, v1976.begin());
  std::copy(v1975.begin(), v1975.end(), v86.begin() + 320);
  std::copy(v1976.begin(), v1976.end(), v86.begin() + 0);
  std::vector<double> v1979(std::begin(v86), std::end(v86));
  auto pt331_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt331_filled = v1979;
  pt331_filled.clear();
  pt331_filled.reserve(pt331_filled_n);
  for (auto i = 0; i < pt331_filled_n; ++i) {
    pt331_filled.push_back(v1979[i % v1979.size()]);
  }
  auto pt331 = cc->MakeCKKSPackedPlaintext(pt331_filled);
  const auto& ct666 = cc->EvalMult(ct22, pt331);
  std::vector<float> v1980(std::begin(v28) + 332 * 512, std::begin(v28) + 332 * 512 + 1024);
  std::vector<float> v1981(704);
  std::copy(v1980.begin() + 0, v1980.begin() + 0 + 704, v1981.begin());
  std::vector<float> v1982(320);
  std::copy(v1980.begin() + 704, v1980.begin() + 704 + 320, v1982.begin());
  std::copy(v1981.begin(), v1981.end(), v86.begin() + 320);
  std::copy(v1982.begin(), v1982.end(), v86.begin() + 0);
  std::vector<double> v1985(std::begin(v86), std::end(v86));
  auto pt332_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt332_filled = v1985;
  pt332_filled.clear();
  pt332_filled.reserve(pt332_filled_n);
  for (auto i = 0; i < pt332_filled_n; ++i) {
    pt332_filled.push_back(v1985[i % v1985.size()]);
  }
  auto pt332 = cc->MakeCKKSPackedPlaintext(pt332_filled);
  const auto& ct667 = cc->EvalMult(ct24, pt332);
  std::vector<float> v1986(std::begin(v28) + 333 * 512, std::begin(v28) + 333 * 512 + 1024);
  std::vector<float> v1987(704);
  std::copy(v1986.begin() + 0, v1986.begin() + 0 + 704, v1987.begin());
  std::vector<float> v1988(320);
  std::copy(v1986.begin() + 704, v1986.begin() + 704 + 320, v1988.begin());
  std::copy(v1987.begin(), v1987.end(), v86.begin() + 320);
  std::copy(v1988.begin(), v1988.end(), v86.begin() + 0);
  std::vector<double> v1991(std::begin(v86), std::end(v86));
  auto pt333_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt333_filled = v1991;
  pt333_filled.clear();
  pt333_filled.reserve(pt333_filled_n);
  for (auto i = 0; i < pt333_filled_n; ++i) {
    pt333_filled.push_back(v1991[i % v1991.size()]);
  }
  auto pt333 = cc->MakeCKKSPackedPlaintext(pt333_filled);
  const auto& ct668 = cc->EvalMult(ct26, pt333);
  std::vector<float> v1992(std::begin(v28) + 334 * 512, std::begin(v28) + 334 * 512 + 1024);
  std::vector<float> v1993(704);
  std::copy(v1992.begin() + 0, v1992.begin() + 0 + 704, v1993.begin());
  std::vector<float> v1994(320);
  std::copy(v1992.begin() + 704, v1992.begin() + 704 + 320, v1994.begin());
  std::copy(v1993.begin(), v1993.end(), v86.begin() + 320);
  std::copy(v1994.begin(), v1994.end(), v86.begin() + 0);
  std::vector<double> v1997(std::begin(v86), std::end(v86));
  auto pt334_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt334_filled = v1997;
  pt334_filled.clear();
  pt334_filled.reserve(pt334_filled_n);
  for (auto i = 0; i < pt334_filled_n; ++i) {
    pt334_filled.push_back(v1997[i % v1997.size()]);
  }
  auto pt334 = cc->MakeCKKSPackedPlaintext(pt334_filled);
  const auto& ct669 = cc->EvalMult(ct28, pt334);
  std::vector<float> v1998(std::begin(v28) + 335 * 512, std::begin(v28) + 335 * 512 + 1024);
  std::vector<float> v1999(704);
  std::copy(v1998.begin() + 0, v1998.begin() + 0 + 704, v1999.begin());
  std::vector<float> v2000(320);
  std::copy(v1998.begin() + 704, v1998.begin() + 704 + 320, v2000.begin());
  std::copy(v1999.begin(), v1999.end(), v86.begin() + 320);
  std::copy(v2000.begin(), v2000.end(), v86.begin() + 0);
  std::vector<double> v2003(std::begin(v86), std::end(v86));
  auto pt335_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt335_filled = v2003;
  pt335_filled.clear();
  pt335_filled.reserve(pt335_filled_n);
  for (auto i = 0; i < pt335_filled_n; ++i) {
    pt335_filled.push_back(v2003[i % v2003.size()]);
  }
  auto pt335 = cc->MakeCKKSPackedPlaintext(pt335_filled);
  const auto& ct670 = cc->EvalMult(ct30, pt335);
  const auto& ct671 = cc->EvalAdd(ct655, ct656);
  const auto& ct672 = cc->EvalAdd(ct657, ct658);
  const auto& ct673 = cc->EvalAdd(ct671, ct672);
  const auto& ct674 = cc->EvalAdd(ct659, ct660);
  const auto& ct675 = cc->EvalAdd(ct661, ct662);
  const auto& ct676 = cc->EvalAdd(ct674, ct675);
  const auto& ct677 = cc->EvalAdd(ct673, ct676);
  const auto& ct678 = cc->EvalAdd(ct663, ct664);
  const auto& ct679 = cc->EvalAdd(ct665, ct666);
  const auto& ct680 = cc->EvalAdd(ct678, ct679);
  const auto& ct681 = cc->EvalAdd(ct667, ct668);
  const auto& ct682 = cc->EvalAdd(ct669, ct670);
  const auto& ct683 = cc->EvalAdd(ct681, ct682);
  const auto& ct684 = cc->EvalAdd(ct680, ct683);
  const auto& ct685 = cc->EvalAdd(ct677, ct684);
  const auto& ct686 = cc->EvalRotate(ct685, 320);
  std::vector<float> v2004(std::begin(v28) + 336 * 512, std::begin(v28) + 336 * 512 + 1024);
  std::vector<float> v2005(688);
  std::copy(v2004.begin() + 0, v2004.begin() + 0 + 688, v2005.begin());
  std::vector<float> v2006(336);
  std::copy(v2004.begin() + 688, v2004.begin() + 688 + 336, v2006.begin());
  std::copy(v2005.begin(), v2005.end(), v86.begin() + 336);
  std::copy(v2006.begin(), v2006.end(), v86.begin() + 0);
  std::vector<double> v2009(std::begin(v86), std::end(v86));
  auto pt336_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt336_filled = v2009;
  pt336_filled.clear();
  pt336_filled.reserve(pt336_filled_n);
  for (auto i = 0; i < pt336_filled_n; ++i) {
    pt336_filled.push_back(v2009[i % v2009.size()]);
  }
  auto pt336 = cc->MakeCKKSPackedPlaintext(pt336_filled);
  const auto& ct687 = cc->EvalMult(ct, pt336);
  std::vector<float> v2010(std::begin(v28) + 337 * 512, std::begin(v28) + 337 * 512 + 1024);
  std::vector<float> v2011(688);
  std::copy(v2010.begin() + 0, v2010.begin() + 0 + 688, v2011.begin());
  std::vector<float> v2012(336);
  std::copy(v2010.begin() + 688, v2010.begin() + 688 + 336, v2012.begin());
  std::copy(v2011.begin(), v2011.end(), v86.begin() + 336);
  std::copy(v2012.begin(), v2012.end(), v86.begin() + 0);
  std::vector<double> v2015(std::begin(v86), std::end(v86));
  auto pt337_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt337_filled = v2015;
  pt337_filled.clear();
  pt337_filled.reserve(pt337_filled_n);
  for (auto i = 0; i < pt337_filled_n; ++i) {
    pt337_filled.push_back(v2015[i % v2015.size()]);
  }
  auto pt337 = cc->MakeCKKSPackedPlaintext(pt337_filled);
  const auto& ct688 = cc->EvalMult(ct2, pt337);
  std::vector<float> v2016(std::begin(v28) + 338 * 512, std::begin(v28) + 338 * 512 + 1024);
  std::vector<float> v2017(688);
  std::copy(v2016.begin() + 0, v2016.begin() + 0 + 688, v2017.begin());
  std::vector<float> v2018(336);
  std::copy(v2016.begin() + 688, v2016.begin() + 688 + 336, v2018.begin());
  std::copy(v2017.begin(), v2017.end(), v86.begin() + 336);
  std::copy(v2018.begin(), v2018.end(), v86.begin() + 0);
  std::vector<double> v2021(std::begin(v86), std::end(v86));
  auto pt338_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt338_filled = v2021;
  pt338_filled.clear();
  pt338_filled.reserve(pt338_filled_n);
  for (auto i = 0; i < pt338_filled_n; ++i) {
    pt338_filled.push_back(v2021[i % v2021.size()]);
  }
  auto pt338 = cc->MakeCKKSPackedPlaintext(pt338_filled);
  const auto& ct689 = cc->EvalMult(ct4, pt338);
  std::vector<float> v2022(std::begin(v28) + 339 * 512, std::begin(v28) + 339 * 512 + 1024);
  std::vector<float> v2023(688);
  std::copy(v2022.begin() + 0, v2022.begin() + 0 + 688, v2023.begin());
  std::vector<float> v2024(336);
  std::copy(v2022.begin() + 688, v2022.begin() + 688 + 336, v2024.begin());
  std::copy(v2023.begin(), v2023.end(), v86.begin() + 336);
  std::copy(v2024.begin(), v2024.end(), v86.begin() + 0);
  std::vector<double> v2027(std::begin(v86), std::end(v86));
  auto pt339_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt339_filled = v2027;
  pt339_filled.clear();
  pt339_filled.reserve(pt339_filled_n);
  for (auto i = 0; i < pt339_filled_n; ++i) {
    pt339_filled.push_back(v2027[i % v2027.size()]);
  }
  auto pt339 = cc->MakeCKKSPackedPlaintext(pt339_filled);
  const auto& ct690 = cc->EvalMult(ct6, pt339);
  std::vector<float> v2028(std::begin(v28) + 340 * 512, std::begin(v28) + 340 * 512 + 1024);
  std::vector<float> v2029(688);
  std::copy(v2028.begin() + 0, v2028.begin() + 0 + 688, v2029.begin());
  std::vector<float> v2030(336);
  std::copy(v2028.begin() + 688, v2028.begin() + 688 + 336, v2030.begin());
  std::copy(v2029.begin(), v2029.end(), v86.begin() + 336);
  std::copy(v2030.begin(), v2030.end(), v86.begin() + 0);
  std::vector<double> v2033(std::begin(v86), std::end(v86));
  auto pt340_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt340_filled = v2033;
  pt340_filled.clear();
  pt340_filled.reserve(pt340_filled_n);
  for (auto i = 0; i < pt340_filled_n; ++i) {
    pt340_filled.push_back(v2033[i % v2033.size()]);
  }
  auto pt340 = cc->MakeCKKSPackedPlaintext(pt340_filled);
  const auto& ct691 = cc->EvalMult(ct8, pt340);
  std::vector<float> v2034(std::begin(v28) + 341 * 512, std::begin(v28) + 341 * 512 + 1024);
  std::vector<float> v2035(688);
  std::copy(v2034.begin() + 0, v2034.begin() + 0 + 688, v2035.begin());
  std::vector<float> v2036(336);
  std::copy(v2034.begin() + 688, v2034.begin() + 688 + 336, v2036.begin());
  std::copy(v2035.begin(), v2035.end(), v86.begin() + 336);
  std::copy(v2036.begin(), v2036.end(), v86.begin() + 0);
  std::vector<double> v2039(std::begin(v86), std::end(v86));
  auto pt341_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt341_filled = v2039;
  pt341_filled.clear();
  pt341_filled.reserve(pt341_filled_n);
  for (auto i = 0; i < pt341_filled_n; ++i) {
    pt341_filled.push_back(v2039[i % v2039.size()]);
  }
  auto pt341 = cc->MakeCKKSPackedPlaintext(pt341_filled);
  const auto& ct692 = cc->EvalMult(ct10, pt341);
  std::vector<float> v2040(std::begin(v28) + 342 * 512, std::begin(v28) + 342 * 512 + 1024);
  std::vector<float> v2041(688);
  std::copy(v2040.begin() + 0, v2040.begin() + 0 + 688, v2041.begin());
  std::vector<float> v2042(336);
  std::copy(v2040.begin() + 688, v2040.begin() + 688 + 336, v2042.begin());
  std::copy(v2041.begin(), v2041.end(), v86.begin() + 336);
  std::copy(v2042.begin(), v2042.end(), v86.begin() + 0);
  std::vector<double> v2045(std::begin(v86), std::end(v86));
  auto pt342_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt342_filled = v2045;
  pt342_filled.clear();
  pt342_filled.reserve(pt342_filled_n);
  for (auto i = 0; i < pt342_filled_n; ++i) {
    pt342_filled.push_back(v2045[i % v2045.size()]);
  }
  auto pt342 = cc->MakeCKKSPackedPlaintext(pt342_filled);
  const auto& ct693 = cc->EvalMult(ct12, pt342);
  std::vector<float> v2046(std::begin(v28) + 343 * 512, std::begin(v28) + 343 * 512 + 1024);
  std::vector<float> v2047(688);
  std::copy(v2046.begin() + 0, v2046.begin() + 0 + 688, v2047.begin());
  std::vector<float> v2048(336);
  std::copy(v2046.begin() + 688, v2046.begin() + 688 + 336, v2048.begin());
  std::copy(v2047.begin(), v2047.end(), v86.begin() + 336);
  std::copy(v2048.begin(), v2048.end(), v86.begin() + 0);
  std::vector<double> v2051(std::begin(v86), std::end(v86));
  auto pt343_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt343_filled = v2051;
  pt343_filled.clear();
  pt343_filled.reserve(pt343_filled_n);
  for (auto i = 0; i < pt343_filled_n; ++i) {
    pt343_filled.push_back(v2051[i % v2051.size()]);
  }
  auto pt343 = cc->MakeCKKSPackedPlaintext(pt343_filled);
  const auto& ct694 = cc->EvalMult(ct14, pt343);
  std::vector<float> v2052(std::begin(v28) + 344 * 512, std::begin(v28) + 344 * 512 + 1024);
  std::vector<float> v2053(688);
  std::copy(v2052.begin() + 0, v2052.begin() + 0 + 688, v2053.begin());
  std::vector<float> v2054(336);
  std::copy(v2052.begin() + 688, v2052.begin() + 688 + 336, v2054.begin());
  std::copy(v2053.begin(), v2053.end(), v86.begin() + 336);
  std::copy(v2054.begin(), v2054.end(), v86.begin() + 0);
  std::vector<double> v2057(std::begin(v86), std::end(v86));
  auto pt344_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt344_filled = v2057;
  pt344_filled.clear();
  pt344_filled.reserve(pt344_filled_n);
  for (auto i = 0; i < pt344_filled_n; ++i) {
    pt344_filled.push_back(v2057[i % v2057.size()]);
  }
  auto pt344 = cc->MakeCKKSPackedPlaintext(pt344_filled);
  const auto& ct695 = cc->EvalMult(ct16, pt344);
  std::vector<float> v2058(std::begin(v28) + 345 * 512, std::begin(v28) + 345 * 512 + 1024);
  std::vector<float> v2059(688);
  std::copy(v2058.begin() + 0, v2058.begin() + 0 + 688, v2059.begin());
  std::vector<float> v2060(336);
  std::copy(v2058.begin() + 688, v2058.begin() + 688 + 336, v2060.begin());
  std::copy(v2059.begin(), v2059.end(), v86.begin() + 336);
  std::copy(v2060.begin(), v2060.end(), v86.begin() + 0);
  std::vector<double> v2063(std::begin(v86), std::end(v86));
  auto pt345_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt345_filled = v2063;
  pt345_filled.clear();
  pt345_filled.reserve(pt345_filled_n);
  for (auto i = 0; i < pt345_filled_n; ++i) {
    pt345_filled.push_back(v2063[i % v2063.size()]);
  }
  auto pt345 = cc->MakeCKKSPackedPlaintext(pt345_filled);
  const auto& ct696 = cc->EvalMult(ct18, pt345);
  std::vector<float> v2064(std::begin(v28) + 346 * 512, std::begin(v28) + 346 * 512 + 1024);
  std::vector<float> v2065(688);
  std::copy(v2064.begin() + 0, v2064.begin() + 0 + 688, v2065.begin());
  std::vector<float> v2066(336);
  std::copy(v2064.begin() + 688, v2064.begin() + 688 + 336, v2066.begin());
  std::copy(v2065.begin(), v2065.end(), v86.begin() + 336);
  std::copy(v2066.begin(), v2066.end(), v86.begin() + 0);
  std::vector<double> v2069(std::begin(v86), std::end(v86));
  auto pt346_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt346_filled = v2069;
  pt346_filled.clear();
  pt346_filled.reserve(pt346_filled_n);
  for (auto i = 0; i < pt346_filled_n; ++i) {
    pt346_filled.push_back(v2069[i % v2069.size()]);
  }
  auto pt346 = cc->MakeCKKSPackedPlaintext(pt346_filled);
  const auto& ct697 = cc->EvalMult(ct20, pt346);
  std::vector<float> v2070(std::begin(v28) + 347 * 512, std::begin(v28) + 347 * 512 + 1024);
  std::vector<float> v2071(688);
  std::copy(v2070.begin() + 0, v2070.begin() + 0 + 688, v2071.begin());
  std::vector<float> v2072(336);
  std::copy(v2070.begin() + 688, v2070.begin() + 688 + 336, v2072.begin());
  std::copy(v2071.begin(), v2071.end(), v86.begin() + 336);
  std::copy(v2072.begin(), v2072.end(), v86.begin() + 0);
  std::vector<double> v2075(std::begin(v86), std::end(v86));
  auto pt347_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt347_filled = v2075;
  pt347_filled.clear();
  pt347_filled.reserve(pt347_filled_n);
  for (auto i = 0; i < pt347_filled_n; ++i) {
    pt347_filled.push_back(v2075[i % v2075.size()]);
  }
  auto pt347 = cc->MakeCKKSPackedPlaintext(pt347_filled);
  const auto& ct698 = cc->EvalMult(ct22, pt347);
  std::vector<float> v2076(std::begin(v28) + 348 * 512, std::begin(v28) + 348 * 512 + 1024);
  std::vector<float> v2077(688);
  std::copy(v2076.begin() + 0, v2076.begin() + 0 + 688, v2077.begin());
  std::vector<float> v2078(336);
  std::copy(v2076.begin() + 688, v2076.begin() + 688 + 336, v2078.begin());
  std::copy(v2077.begin(), v2077.end(), v86.begin() + 336);
  std::copy(v2078.begin(), v2078.end(), v86.begin() + 0);
  std::vector<double> v2081(std::begin(v86), std::end(v86));
  auto pt348_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt348_filled = v2081;
  pt348_filled.clear();
  pt348_filled.reserve(pt348_filled_n);
  for (auto i = 0; i < pt348_filled_n; ++i) {
    pt348_filled.push_back(v2081[i % v2081.size()]);
  }
  auto pt348 = cc->MakeCKKSPackedPlaintext(pt348_filled);
  const auto& ct699 = cc->EvalMult(ct24, pt348);
  std::vector<float> v2082(std::begin(v28) + 349 * 512, std::begin(v28) + 349 * 512 + 1024);
  std::vector<float> v2083(688);
  std::copy(v2082.begin() + 0, v2082.begin() + 0 + 688, v2083.begin());
  std::vector<float> v2084(336);
  std::copy(v2082.begin() + 688, v2082.begin() + 688 + 336, v2084.begin());
  std::copy(v2083.begin(), v2083.end(), v86.begin() + 336);
  std::copy(v2084.begin(), v2084.end(), v86.begin() + 0);
  std::vector<double> v2087(std::begin(v86), std::end(v86));
  auto pt349_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt349_filled = v2087;
  pt349_filled.clear();
  pt349_filled.reserve(pt349_filled_n);
  for (auto i = 0; i < pt349_filled_n; ++i) {
    pt349_filled.push_back(v2087[i % v2087.size()]);
  }
  auto pt349 = cc->MakeCKKSPackedPlaintext(pt349_filled);
  const auto& ct700 = cc->EvalMult(ct26, pt349);
  std::vector<float> v2088(std::begin(v28) + 350 * 512, std::begin(v28) + 350 * 512 + 1024);
  std::vector<float> v2089(688);
  std::copy(v2088.begin() + 0, v2088.begin() + 0 + 688, v2089.begin());
  std::vector<float> v2090(336);
  std::copy(v2088.begin() + 688, v2088.begin() + 688 + 336, v2090.begin());
  std::copy(v2089.begin(), v2089.end(), v86.begin() + 336);
  std::copy(v2090.begin(), v2090.end(), v86.begin() + 0);
  std::vector<double> v2093(std::begin(v86), std::end(v86));
  auto pt350_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt350_filled = v2093;
  pt350_filled.clear();
  pt350_filled.reserve(pt350_filled_n);
  for (auto i = 0; i < pt350_filled_n; ++i) {
    pt350_filled.push_back(v2093[i % v2093.size()]);
  }
  auto pt350 = cc->MakeCKKSPackedPlaintext(pt350_filled);
  const auto& ct701 = cc->EvalMult(ct28, pt350);
  std::vector<float> v2094(std::begin(v28) + 351 * 512, std::begin(v28) + 351 * 512 + 1024);
  std::vector<float> v2095(688);
  std::copy(v2094.begin() + 0, v2094.begin() + 0 + 688, v2095.begin());
  std::vector<float> v2096(336);
  std::copy(v2094.begin() + 688, v2094.begin() + 688 + 336, v2096.begin());
  std::copy(v2095.begin(), v2095.end(), v86.begin() + 336);
  std::copy(v2096.begin(), v2096.end(), v86.begin() + 0);
  std::vector<double> v2099(std::begin(v86), std::end(v86));
  auto pt351_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt351_filled = v2099;
  pt351_filled.clear();
  pt351_filled.reserve(pt351_filled_n);
  for (auto i = 0; i < pt351_filled_n; ++i) {
    pt351_filled.push_back(v2099[i % v2099.size()]);
  }
  auto pt351 = cc->MakeCKKSPackedPlaintext(pt351_filled);
  const auto& ct702 = cc->EvalMult(ct30, pt351);
  const auto& ct703 = cc->EvalAdd(ct687, ct688);
  const auto& ct704 = cc->EvalAdd(ct689, ct690);
  const auto& ct705 = cc->EvalAdd(ct703, ct704);
  const auto& ct706 = cc->EvalAdd(ct691, ct692);
  const auto& ct707 = cc->EvalAdd(ct693, ct694);
  const auto& ct708 = cc->EvalAdd(ct706, ct707);
  const auto& ct709 = cc->EvalAdd(ct705, ct708);
  const auto& ct710 = cc->EvalAdd(ct695, ct696);
  const auto& ct711 = cc->EvalAdd(ct697, ct698);
  const auto& ct712 = cc->EvalAdd(ct710, ct711);
  const auto& ct713 = cc->EvalAdd(ct699, ct700);
  const auto& ct714 = cc->EvalAdd(ct701, ct702);
  const auto& ct715 = cc->EvalAdd(ct713, ct714);
  const auto& ct716 = cc->EvalAdd(ct712, ct715);
  const auto& ct717 = cc->EvalAdd(ct709, ct716);
  const auto& ct718 = cc->EvalRotate(ct717, 336);
  std::vector<float> v2100(std::begin(v28) + 352 * 512, std::begin(v28) + 352 * 512 + 1024);
  std::vector<float> v2101(672);
  std::copy(v2100.begin() + 0, v2100.begin() + 0 + 672, v2101.begin());
  std::vector<float> v2102(352);
  std::copy(v2100.begin() + 672, v2100.begin() + 672 + 352, v2102.begin());
  std::copy(v2101.begin(), v2101.end(), v86.begin() + 352);
  std::copy(v2102.begin(), v2102.end(), v86.begin() + 0);
  std::vector<double> v2105(std::begin(v86), std::end(v86));
  auto pt352_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt352_filled = v2105;
  pt352_filled.clear();
  pt352_filled.reserve(pt352_filled_n);
  for (auto i = 0; i < pt352_filled_n; ++i) {
    pt352_filled.push_back(v2105[i % v2105.size()]);
  }
  auto pt352 = cc->MakeCKKSPackedPlaintext(pt352_filled);
  const auto& ct719 = cc->EvalMult(ct, pt352);
  std::vector<float> v2106(std::begin(v28) + 353 * 512, std::begin(v28) + 353 * 512 + 1024);
  std::vector<float> v2107(672);
  std::copy(v2106.begin() + 0, v2106.begin() + 0 + 672, v2107.begin());
  std::vector<float> v2108(352);
  std::copy(v2106.begin() + 672, v2106.begin() + 672 + 352, v2108.begin());
  std::copy(v2107.begin(), v2107.end(), v86.begin() + 352);
  std::copy(v2108.begin(), v2108.end(), v86.begin() + 0);
  std::vector<double> v2111(std::begin(v86), std::end(v86));
  auto pt353_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt353_filled = v2111;
  pt353_filled.clear();
  pt353_filled.reserve(pt353_filled_n);
  for (auto i = 0; i < pt353_filled_n; ++i) {
    pt353_filled.push_back(v2111[i % v2111.size()]);
  }
  auto pt353 = cc->MakeCKKSPackedPlaintext(pt353_filled);
  const auto& ct720 = cc->EvalMult(ct2, pt353);
  std::vector<float> v2112(std::begin(v28) + 354 * 512, std::begin(v28) + 354 * 512 + 1024);
  std::vector<float> v2113(672);
  std::copy(v2112.begin() + 0, v2112.begin() + 0 + 672, v2113.begin());
  std::vector<float> v2114(352);
  std::copy(v2112.begin() + 672, v2112.begin() + 672 + 352, v2114.begin());
  std::copy(v2113.begin(), v2113.end(), v86.begin() + 352);
  std::copy(v2114.begin(), v2114.end(), v86.begin() + 0);
  std::vector<double> v2117(std::begin(v86), std::end(v86));
  auto pt354_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt354_filled = v2117;
  pt354_filled.clear();
  pt354_filled.reserve(pt354_filled_n);
  for (auto i = 0; i < pt354_filled_n; ++i) {
    pt354_filled.push_back(v2117[i % v2117.size()]);
  }
  auto pt354 = cc->MakeCKKSPackedPlaintext(pt354_filled);
  const auto& ct721 = cc->EvalMult(ct4, pt354);
  std::vector<float> v2118(std::begin(v28) + 355 * 512, std::begin(v28) + 355 * 512 + 1024);
  std::vector<float> v2119(672);
  std::copy(v2118.begin() + 0, v2118.begin() + 0 + 672, v2119.begin());
  std::vector<float> v2120(352);
  std::copy(v2118.begin() + 672, v2118.begin() + 672 + 352, v2120.begin());
  std::copy(v2119.begin(), v2119.end(), v86.begin() + 352);
  std::copy(v2120.begin(), v2120.end(), v86.begin() + 0);
  std::vector<double> v2123(std::begin(v86), std::end(v86));
  auto pt355_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt355_filled = v2123;
  pt355_filled.clear();
  pt355_filled.reserve(pt355_filled_n);
  for (auto i = 0; i < pt355_filled_n; ++i) {
    pt355_filled.push_back(v2123[i % v2123.size()]);
  }
  auto pt355 = cc->MakeCKKSPackedPlaintext(pt355_filled);
  const auto& ct722 = cc->EvalMult(ct6, pt355);
  std::vector<float> v2124(std::begin(v28) + 356 * 512, std::begin(v28) + 356 * 512 + 1024);
  std::vector<float> v2125(672);
  std::copy(v2124.begin() + 0, v2124.begin() + 0 + 672, v2125.begin());
  std::vector<float> v2126(352);
  std::copy(v2124.begin() + 672, v2124.begin() + 672 + 352, v2126.begin());
  std::copy(v2125.begin(), v2125.end(), v86.begin() + 352);
  std::copy(v2126.begin(), v2126.end(), v86.begin() + 0);
  std::vector<double> v2129(std::begin(v86), std::end(v86));
  auto pt356_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt356_filled = v2129;
  pt356_filled.clear();
  pt356_filled.reserve(pt356_filled_n);
  for (auto i = 0; i < pt356_filled_n; ++i) {
    pt356_filled.push_back(v2129[i % v2129.size()]);
  }
  auto pt356 = cc->MakeCKKSPackedPlaintext(pt356_filled);
  const auto& ct723 = cc->EvalMult(ct8, pt356);
  std::vector<float> v2130(std::begin(v28) + 357 * 512, std::begin(v28) + 357 * 512 + 1024);
  std::vector<float> v2131(672);
  std::copy(v2130.begin() + 0, v2130.begin() + 0 + 672, v2131.begin());
  std::vector<float> v2132(352);
  std::copy(v2130.begin() + 672, v2130.begin() + 672 + 352, v2132.begin());
  std::copy(v2131.begin(), v2131.end(), v86.begin() + 352);
  std::copy(v2132.begin(), v2132.end(), v86.begin() + 0);
  std::vector<double> v2135(std::begin(v86), std::end(v86));
  auto pt357_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt357_filled = v2135;
  pt357_filled.clear();
  pt357_filled.reserve(pt357_filled_n);
  for (auto i = 0; i < pt357_filled_n; ++i) {
    pt357_filled.push_back(v2135[i % v2135.size()]);
  }
  auto pt357 = cc->MakeCKKSPackedPlaintext(pt357_filled);
  const auto& ct724 = cc->EvalMult(ct10, pt357);
  std::vector<float> v2136(std::begin(v28) + 358 * 512, std::begin(v28) + 358 * 512 + 1024);
  std::vector<float> v2137(672);
  std::copy(v2136.begin() + 0, v2136.begin() + 0 + 672, v2137.begin());
  std::vector<float> v2138(352);
  std::copy(v2136.begin() + 672, v2136.begin() + 672 + 352, v2138.begin());
  std::copy(v2137.begin(), v2137.end(), v86.begin() + 352);
  std::copy(v2138.begin(), v2138.end(), v86.begin() + 0);
  std::vector<double> v2141(std::begin(v86), std::end(v86));
  auto pt358_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt358_filled = v2141;
  pt358_filled.clear();
  pt358_filled.reserve(pt358_filled_n);
  for (auto i = 0; i < pt358_filled_n; ++i) {
    pt358_filled.push_back(v2141[i % v2141.size()]);
  }
  auto pt358 = cc->MakeCKKSPackedPlaintext(pt358_filled);
  const auto& ct725 = cc->EvalMult(ct12, pt358);
  std::vector<float> v2142(std::begin(v28) + 359 * 512, std::begin(v28) + 359 * 512 + 1024);
  std::vector<float> v2143(672);
  std::copy(v2142.begin() + 0, v2142.begin() + 0 + 672, v2143.begin());
  std::vector<float> v2144(352);
  std::copy(v2142.begin() + 672, v2142.begin() + 672 + 352, v2144.begin());
  std::copy(v2143.begin(), v2143.end(), v86.begin() + 352);
  std::copy(v2144.begin(), v2144.end(), v86.begin() + 0);
  std::vector<double> v2147(std::begin(v86), std::end(v86));
  auto pt359_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt359_filled = v2147;
  pt359_filled.clear();
  pt359_filled.reserve(pt359_filled_n);
  for (auto i = 0; i < pt359_filled_n; ++i) {
    pt359_filled.push_back(v2147[i % v2147.size()]);
  }
  auto pt359 = cc->MakeCKKSPackedPlaintext(pt359_filled);
  const auto& ct726 = cc->EvalMult(ct14, pt359);
  std::vector<float> v2148(std::begin(v28) + 360 * 512, std::begin(v28) + 360 * 512 + 1024);
  std::vector<float> v2149(672);
  std::copy(v2148.begin() + 0, v2148.begin() + 0 + 672, v2149.begin());
  std::vector<float> v2150(352);
  std::copy(v2148.begin() + 672, v2148.begin() + 672 + 352, v2150.begin());
  std::copy(v2149.begin(), v2149.end(), v86.begin() + 352);
  std::copy(v2150.begin(), v2150.end(), v86.begin() + 0);
  std::vector<double> v2153(std::begin(v86), std::end(v86));
  auto pt360_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt360_filled = v2153;
  pt360_filled.clear();
  pt360_filled.reserve(pt360_filled_n);
  for (auto i = 0; i < pt360_filled_n; ++i) {
    pt360_filled.push_back(v2153[i % v2153.size()]);
  }
  auto pt360 = cc->MakeCKKSPackedPlaintext(pt360_filled);
  const auto& ct727 = cc->EvalMult(ct16, pt360);
  std::vector<float> v2154(std::begin(v28) + 361 * 512, std::begin(v28) + 361 * 512 + 1024);
  std::vector<float> v2155(672);
  std::copy(v2154.begin() + 0, v2154.begin() + 0 + 672, v2155.begin());
  std::vector<float> v2156(352);
  std::copy(v2154.begin() + 672, v2154.begin() + 672 + 352, v2156.begin());
  std::copy(v2155.begin(), v2155.end(), v86.begin() + 352);
  std::copy(v2156.begin(), v2156.end(), v86.begin() + 0);
  std::vector<double> v2159(std::begin(v86), std::end(v86));
  auto pt361_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt361_filled = v2159;
  pt361_filled.clear();
  pt361_filled.reserve(pt361_filled_n);
  for (auto i = 0; i < pt361_filled_n; ++i) {
    pt361_filled.push_back(v2159[i % v2159.size()]);
  }
  auto pt361 = cc->MakeCKKSPackedPlaintext(pt361_filled);
  const auto& ct728 = cc->EvalMult(ct18, pt361);
  std::vector<float> v2160(std::begin(v28) + 362 * 512, std::begin(v28) + 362 * 512 + 1024);
  std::vector<float> v2161(672);
  std::copy(v2160.begin() + 0, v2160.begin() + 0 + 672, v2161.begin());
  std::vector<float> v2162(352);
  std::copy(v2160.begin() + 672, v2160.begin() + 672 + 352, v2162.begin());
  std::copy(v2161.begin(), v2161.end(), v86.begin() + 352);
  std::copy(v2162.begin(), v2162.end(), v86.begin() + 0);
  std::vector<double> v2165(std::begin(v86), std::end(v86));
  auto pt362_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt362_filled = v2165;
  pt362_filled.clear();
  pt362_filled.reserve(pt362_filled_n);
  for (auto i = 0; i < pt362_filled_n; ++i) {
    pt362_filled.push_back(v2165[i % v2165.size()]);
  }
  auto pt362 = cc->MakeCKKSPackedPlaintext(pt362_filled);
  const auto& ct729 = cc->EvalMult(ct20, pt362);
  std::vector<float> v2166(std::begin(v28) + 363 * 512, std::begin(v28) + 363 * 512 + 1024);
  std::vector<float> v2167(672);
  std::copy(v2166.begin() + 0, v2166.begin() + 0 + 672, v2167.begin());
  std::vector<float> v2168(352);
  std::copy(v2166.begin() + 672, v2166.begin() + 672 + 352, v2168.begin());
  std::copy(v2167.begin(), v2167.end(), v86.begin() + 352);
  std::copy(v2168.begin(), v2168.end(), v86.begin() + 0);
  std::vector<double> v2171(std::begin(v86), std::end(v86));
  auto pt363_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt363_filled = v2171;
  pt363_filled.clear();
  pt363_filled.reserve(pt363_filled_n);
  for (auto i = 0; i < pt363_filled_n; ++i) {
    pt363_filled.push_back(v2171[i % v2171.size()]);
  }
  auto pt363 = cc->MakeCKKSPackedPlaintext(pt363_filled);
  const auto& ct730 = cc->EvalMult(ct22, pt363);
  std::vector<float> v2172(std::begin(v28) + 364 * 512, std::begin(v28) + 364 * 512 + 1024);
  std::vector<float> v2173(672);
  std::copy(v2172.begin() + 0, v2172.begin() + 0 + 672, v2173.begin());
  std::vector<float> v2174(352);
  std::copy(v2172.begin() + 672, v2172.begin() + 672 + 352, v2174.begin());
  std::copy(v2173.begin(), v2173.end(), v86.begin() + 352);
  std::copy(v2174.begin(), v2174.end(), v86.begin() + 0);
  std::vector<double> v2177(std::begin(v86), std::end(v86));
  auto pt364_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt364_filled = v2177;
  pt364_filled.clear();
  pt364_filled.reserve(pt364_filled_n);
  for (auto i = 0; i < pt364_filled_n; ++i) {
    pt364_filled.push_back(v2177[i % v2177.size()]);
  }
  auto pt364 = cc->MakeCKKSPackedPlaintext(pt364_filled);
  const auto& ct731 = cc->EvalMult(ct24, pt364);
  std::vector<float> v2178(std::begin(v28) + 365 * 512, std::begin(v28) + 365 * 512 + 1024);
  std::vector<float> v2179(672);
  std::copy(v2178.begin() + 0, v2178.begin() + 0 + 672, v2179.begin());
  std::vector<float> v2180(352);
  std::copy(v2178.begin() + 672, v2178.begin() + 672 + 352, v2180.begin());
  std::copy(v2179.begin(), v2179.end(), v86.begin() + 352);
  std::copy(v2180.begin(), v2180.end(), v86.begin() + 0);
  std::vector<double> v2183(std::begin(v86), std::end(v86));
  auto pt365_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt365_filled = v2183;
  pt365_filled.clear();
  pt365_filled.reserve(pt365_filled_n);
  for (auto i = 0; i < pt365_filled_n; ++i) {
    pt365_filled.push_back(v2183[i % v2183.size()]);
  }
  auto pt365 = cc->MakeCKKSPackedPlaintext(pt365_filled);
  const auto& ct732 = cc->EvalMult(ct26, pt365);
  std::vector<float> v2184(std::begin(v28) + 366 * 512, std::begin(v28) + 366 * 512 + 1024);
  std::vector<float> v2185(672);
  std::copy(v2184.begin() + 0, v2184.begin() + 0 + 672, v2185.begin());
  std::vector<float> v2186(352);
  std::copy(v2184.begin() + 672, v2184.begin() + 672 + 352, v2186.begin());
  std::copy(v2185.begin(), v2185.end(), v86.begin() + 352);
  std::copy(v2186.begin(), v2186.end(), v86.begin() + 0);
  std::vector<double> v2189(std::begin(v86), std::end(v86));
  auto pt366_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt366_filled = v2189;
  pt366_filled.clear();
  pt366_filled.reserve(pt366_filled_n);
  for (auto i = 0; i < pt366_filled_n; ++i) {
    pt366_filled.push_back(v2189[i % v2189.size()]);
  }
  auto pt366 = cc->MakeCKKSPackedPlaintext(pt366_filled);
  const auto& ct733 = cc->EvalMult(ct28, pt366);
  std::vector<float> v2190(std::begin(v28) + 367 * 512, std::begin(v28) + 367 * 512 + 1024);
  std::vector<float> v2191(672);
  std::copy(v2190.begin() + 0, v2190.begin() + 0 + 672, v2191.begin());
  std::vector<float> v2192(352);
  std::copy(v2190.begin() + 672, v2190.begin() + 672 + 352, v2192.begin());
  std::copy(v2191.begin(), v2191.end(), v86.begin() + 352);
  std::copy(v2192.begin(), v2192.end(), v86.begin() + 0);
  std::vector<double> v2195(std::begin(v86), std::end(v86));
  auto pt367_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt367_filled = v2195;
  pt367_filled.clear();
  pt367_filled.reserve(pt367_filled_n);
  for (auto i = 0; i < pt367_filled_n; ++i) {
    pt367_filled.push_back(v2195[i % v2195.size()]);
  }
  auto pt367 = cc->MakeCKKSPackedPlaintext(pt367_filled);
  const auto& ct734 = cc->EvalMult(ct30, pt367);
  const auto& ct735 = cc->EvalAdd(ct719, ct720);
  const auto& ct736 = cc->EvalAdd(ct721, ct722);
  const auto& ct737 = cc->EvalAdd(ct735, ct736);
  const auto& ct738 = cc->EvalAdd(ct723, ct724);
  const auto& ct739 = cc->EvalAdd(ct725, ct726);
  const auto& ct740 = cc->EvalAdd(ct738, ct739);
  const auto& ct741 = cc->EvalAdd(ct737, ct740);
  const auto& ct742 = cc->EvalAdd(ct727, ct728);
  const auto& ct743 = cc->EvalAdd(ct729, ct730);
  const auto& ct744 = cc->EvalAdd(ct742, ct743);
  const auto& ct745 = cc->EvalAdd(ct731, ct732);
  const auto& ct746 = cc->EvalAdd(ct733, ct734);
  const auto& ct747 = cc->EvalAdd(ct745, ct746);
  const auto& ct748 = cc->EvalAdd(ct744, ct747);
  const auto& ct749 = cc->EvalAdd(ct741, ct748);
  const auto& ct750 = cc->EvalRotate(ct749, 352);
  std::vector<float> v2196(std::begin(v28) + 368 * 512, std::begin(v28) + 368 * 512 + 1024);
  std::vector<float> v2197(656);
  std::copy(v2196.begin() + 0, v2196.begin() + 0 + 656, v2197.begin());
  std::vector<float> v2198(368);
  std::copy(v2196.begin() + 656, v2196.begin() + 656 + 368, v2198.begin());
  std::copy(v2197.begin(), v2197.end(), v86.begin() + 368);
  std::copy(v2198.begin(), v2198.end(), v86.begin() + 0);
  std::vector<double> v2201(std::begin(v86), std::end(v86));
  auto pt368_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt368_filled = v2201;
  pt368_filled.clear();
  pt368_filled.reserve(pt368_filled_n);
  for (auto i = 0; i < pt368_filled_n; ++i) {
    pt368_filled.push_back(v2201[i % v2201.size()]);
  }
  auto pt368 = cc->MakeCKKSPackedPlaintext(pt368_filled);
  const auto& ct751 = cc->EvalMult(ct, pt368);
  std::vector<float> v2202(std::begin(v28) + 369 * 512, std::begin(v28) + 369 * 512 + 1024);
  std::vector<float> v2203(656);
  std::copy(v2202.begin() + 0, v2202.begin() + 0 + 656, v2203.begin());
  std::vector<float> v2204(368);
  std::copy(v2202.begin() + 656, v2202.begin() + 656 + 368, v2204.begin());
  std::copy(v2203.begin(), v2203.end(), v86.begin() + 368);
  std::copy(v2204.begin(), v2204.end(), v86.begin() + 0);
  std::vector<double> v2207(std::begin(v86), std::end(v86));
  auto pt369_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt369_filled = v2207;
  pt369_filled.clear();
  pt369_filled.reserve(pt369_filled_n);
  for (auto i = 0; i < pt369_filled_n; ++i) {
    pt369_filled.push_back(v2207[i % v2207.size()]);
  }
  auto pt369 = cc->MakeCKKSPackedPlaintext(pt369_filled);
  const auto& ct752 = cc->EvalMult(ct2, pt369);
  std::vector<float> v2208(std::begin(v28) + 370 * 512, std::begin(v28) + 370 * 512 + 1024);
  std::vector<float> v2209(656);
  std::copy(v2208.begin() + 0, v2208.begin() + 0 + 656, v2209.begin());
  std::vector<float> v2210(368);
  std::copy(v2208.begin() + 656, v2208.begin() + 656 + 368, v2210.begin());
  std::copy(v2209.begin(), v2209.end(), v86.begin() + 368);
  std::copy(v2210.begin(), v2210.end(), v86.begin() + 0);
  std::vector<double> v2213(std::begin(v86), std::end(v86));
  auto pt370_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt370_filled = v2213;
  pt370_filled.clear();
  pt370_filled.reserve(pt370_filled_n);
  for (auto i = 0; i < pt370_filled_n; ++i) {
    pt370_filled.push_back(v2213[i % v2213.size()]);
  }
  auto pt370 = cc->MakeCKKSPackedPlaintext(pt370_filled);
  const auto& ct753 = cc->EvalMult(ct4, pt370);
  std::vector<float> v2214(std::begin(v28) + 371 * 512, std::begin(v28) + 371 * 512 + 1024);
  std::vector<float> v2215(656);
  std::copy(v2214.begin() + 0, v2214.begin() + 0 + 656, v2215.begin());
  std::vector<float> v2216(368);
  std::copy(v2214.begin() + 656, v2214.begin() + 656 + 368, v2216.begin());
  std::copy(v2215.begin(), v2215.end(), v86.begin() + 368);
  std::copy(v2216.begin(), v2216.end(), v86.begin() + 0);
  std::vector<double> v2219(std::begin(v86), std::end(v86));
  auto pt371_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt371_filled = v2219;
  pt371_filled.clear();
  pt371_filled.reserve(pt371_filled_n);
  for (auto i = 0; i < pt371_filled_n; ++i) {
    pt371_filled.push_back(v2219[i % v2219.size()]);
  }
  auto pt371 = cc->MakeCKKSPackedPlaintext(pt371_filled);
  const auto& ct754 = cc->EvalMult(ct6, pt371);
  std::vector<float> v2220(std::begin(v28) + 372 * 512, std::begin(v28) + 372 * 512 + 1024);
  std::vector<float> v2221(656);
  std::copy(v2220.begin() + 0, v2220.begin() + 0 + 656, v2221.begin());
  std::vector<float> v2222(368);
  std::copy(v2220.begin() + 656, v2220.begin() + 656 + 368, v2222.begin());
  std::copy(v2221.begin(), v2221.end(), v86.begin() + 368);
  std::copy(v2222.begin(), v2222.end(), v86.begin() + 0);
  std::vector<double> v2225(std::begin(v86), std::end(v86));
  auto pt372_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt372_filled = v2225;
  pt372_filled.clear();
  pt372_filled.reserve(pt372_filled_n);
  for (auto i = 0; i < pt372_filled_n; ++i) {
    pt372_filled.push_back(v2225[i % v2225.size()]);
  }
  auto pt372 = cc->MakeCKKSPackedPlaintext(pt372_filled);
  const auto& ct755 = cc->EvalMult(ct8, pt372);
  std::vector<float> v2226(std::begin(v28) + 373 * 512, std::begin(v28) + 373 * 512 + 1024);
  std::vector<float> v2227(656);
  std::copy(v2226.begin() + 0, v2226.begin() + 0 + 656, v2227.begin());
  std::vector<float> v2228(368);
  std::copy(v2226.begin() + 656, v2226.begin() + 656 + 368, v2228.begin());
  std::copy(v2227.begin(), v2227.end(), v86.begin() + 368);
  std::copy(v2228.begin(), v2228.end(), v86.begin() + 0);
  std::vector<double> v2231(std::begin(v86), std::end(v86));
  auto pt373_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt373_filled = v2231;
  pt373_filled.clear();
  pt373_filled.reserve(pt373_filled_n);
  for (auto i = 0; i < pt373_filled_n; ++i) {
    pt373_filled.push_back(v2231[i % v2231.size()]);
  }
  auto pt373 = cc->MakeCKKSPackedPlaintext(pt373_filled);
  const auto& ct756 = cc->EvalMult(ct10, pt373);
  std::vector<float> v2232(std::begin(v28) + 374 * 512, std::begin(v28) + 374 * 512 + 1024);
  std::vector<float> v2233(656);
  std::copy(v2232.begin() + 0, v2232.begin() + 0 + 656, v2233.begin());
  std::vector<float> v2234(368);
  std::copy(v2232.begin() + 656, v2232.begin() + 656 + 368, v2234.begin());
  std::copy(v2233.begin(), v2233.end(), v86.begin() + 368);
  std::copy(v2234.begin(), v2234.end(), v86.begin() + 0);
  std::vector<double> v2237(std::begin(v86), std::end(v86));
  auto pt374_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt374_filled = v2237;
  pt374_filled.clear();
  pt374_filled.reserve(pt374_filled_n);
  for (auto i = 0; i < pt374_filled_n; ++i) {
    pt374_filled.push_back(v2237[i % v2237.size()]);
  }
  auto pt374 = cc->MakeCKKSPackedPlaintext(pt374_filled);
  const auto& ct757 = cc->EvalMult(ct12, pt374);
  std::vector<float> v2238(std::begin(v28) + 375 * 512, std::begin(v28) + 375 * 512 + 1024);
  std::vector<float> v2239(656);
  std::copy(v2238.begin() + 0, v2238.begin() + 0 + 656, v2239.begin());
  std::vector<float> v2240(368);
  std::copy(v2238.begin() + 656, v2238.begin() + 656 + 368, v2240.begin());
  std::copy(v2239.begin(), v2239.end(), v86.begin() + 368);
  std::copy(v2240.begin(), v2240.end(), v86.begin() + 0);
  std::vector<double> v2243(std::begin(v86), std::end(v86));
  auto pt375_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt375_filled = v2243;
  pt375_filled.clear();
  pt375_filled.reserve(pt375_filled_n);
  for (auto i = 0; i < pt375_filled_n; ++i) {
    pt375_filled.push_back(v2243[i % v2243.size()]);
  }
  auto pt375 = cc->MakeCKKSPackedPlaintext(pt375_filled);
  const auto& ct758 = cc->EvalMult(ct14, pt375);
  std::vector<float> v2244(std::begin(v28) + 376 * 512, std::begin(v28) + 376 * 512 + 1024);
  std::vector<float> v2245(656);
  std::copy(v2244.begin() + 0, v2244.begin() + 0 + 656, v2245.begin());
  std::vector<float> v2246(368);
  std::copy(v2244.begin() + 656, v2244.begin() + 656 + 368, v2246.begin());
  std::copy(v2245.begin(), v2245.end(), v86.begin() + 368);
  std::copy(v2246.begin(), v2246.end(), v86.begin() + 0);
  std::vector<double> v2249(std::begin(v86), std::end(v86));
  auto pt376_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt376_filled = v2249;
  pt376_filled.clear();
  pt376_filled.reserve(pt376_filled_n);
  for (auto i = 0; i < pt376_filled_n; ++i) {
    pt376_filled.push_back(v2249[i % v2249.size()]);
  }
  auto pt376 = cc->MakeCKKSPackedPlaintext(pt376_filled);
  const auto& ct759 = cc->EvalMult(ct16, pt376);
  std::vector<float> v2250(std::begin(v28) + 377 * 512, std::begin(v28) + 377 * 512 + 1024);
  std::vector<float> v2251(656);
  std::copy(v2250.begin() + 0, v2250.begin() + 0 + 656, v2251.begin());
  std::vector<float> v2252(368);
  std::copy(v2250.begin() + 656, v2250.begin() + 656 + 368, v2252.begin());
  std::copy(v2251.begin(), v2251.end(), v86.begin() + 368);
  std::copy(v2252.begin(), v2252.end(), v86.begin() + 0);
  std::vector<double> v2255(std::begin(v86), std::end(v86));
  auto pt377_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt377_filled = v2255;
  pt377_filled.clear();
  pt377_filled.reserve(pt377_filled_n);
  for (auto i = 0; i < pt377_filled_n; ++i) {
    pt377_filled.push_back(v2255[i % v2255.size()]);
  }
  auto pt377 = cc->MakeCKKSPackedPlaintext(pt377_filled);
  const auto& ct760 = cc->EvalMult(ct18, pt377);
  std::vector<float> v2256(std::begin(v28) + 378 * 512, std::begin(v28) + 378 * 512 + 1024);
  std::vector<float> v2257(656);
  std::copy(v2256.begin() + 0, v2256.begin() + 0 + 656, v2257.begin());
  std::vector<float> v2258(368);
  std::copy(v2256.begin() + 656, v2256.begin() + 656 + 368, v2258.begin());
  std::copy(v2257.begin(), v2257.end(), v86.begin() + 368);
  std::copy(v2258.begin(), v2258.end(), v86.begin() + 0);
  std::vector<double> v2261(std::begin(v86), std::end(v86));
  auto pt378_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt378_filled = v2261;
  pt378_filled.clear();
  pt378_filled.reserve(pt378_filled_n);
  for (auto i = 0; i < pt378_filled_n; ++i) {
    pt378_filled.push_back(v2261[i % v2261.size()]);
  }
  auto pt378 = cc->MakeCKKSPackedPlaintext(pt378_filled);
  const auto& ct761 = cc->EvalMult(ct20, pt378);
  std::vector<float> v2262(std::begin(v28) + 379 * 512, std::begin(v28) + 379 * 512 + 1024);
  std::vector<float> v2263(656);
  std::copy(v2262.begin() + 0, v2262.begin() + 0 + 656, v2263.begin());
  std::vector<float> v2264(368);
  std::copy(v2262.begin() + 656, v2262.begin() + 656 + 368, v2264.begin());
  std::copy(v2263.begin(), v2263.end(), v86.begin() + 368);
  std::copy(v2264.begin(), v2264.end(), v86.begin() + 0);
  std::vector<double> v2267(std::begin(v86), std::end(v86));
  auto pt379_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt379_filled = v2267;
  pt379_filled.clear();
  pt379_filled.reserve(pt379_filled_n);
  for (auto i = 0; i < pt379_filled_n; ++i) {
    pt379_filled.push_back(v2267[i % v2267.size()]);
  }
  auto pt379 = cc->MakeCKKSPackedPlaintext(pt379_filled);
  const auto& ct762 = cc->EvalMult(ct22, pt379);
  std::vector<float> v2268(std::begin(v28) + 380 * 512, std::begin(v28) + 380 * 512 + 1024);
  std::vector<float> v2269(656);
  std::copy(v2268.begin() + 0, v2268.begin() + 0 + 656, v2269.begin());
  std::vector<float> v2270(368);
  std::copy(v2268.begin() + 656, v2268.begin() + 656 + 368, v2270.begin());
  std::copy(v2269.begin(), v2269.end(), v86.begin() + 368);
  std::copy(v2270.begin(), v2270.end(), v86.begin() + 0);
  std::vector<double> v2273(std::begin(v86), std::end(v86));
  auto pt380_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt380_filled = v2273;
  pt380_filled.clear();
  pt380_filled.reserve(pt380_filled_n);
  for (auto i = 0; i < pt380_filled_n; ++i) {
    pt380_filled.push_back(v2273[i % v2273.size()]);
  }
  auto pt380 = cc->MakeCKKSPackedPlaintext(pt380_filled);
  const auto& ct763 = cc->EvalMult(ct24, pt380);
  std::vector<float> v2274(std::begin(v28) + 381 * 512, std::begin(v28) + 381 * 512 + 1024);
  std::vector<float> v2275(656);
  std::copy(v2274.begin() + 0, v2274.begin() + 0 + 656, v2275.begin());
  std::vector<float> v2276(368);
  std::copy(v2274.begin() + 656, v2274.begin() + 656 + 368, v2276.begin());
  std::copy(v2275.begin(), v2275.end(), v86.begin() + 368);
  std::copy(v2276.begin(), v2276.end(), v86.begin() + 0);
  std::vector<double> v2279(std::begin(v86), std::end(v86));
  auto pt381_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt381_filled = v2279;
  pt381_filled.clear();
  pt381_filled.reserve(pt381_filled_n);
  for (auto i = 0; i < pt381_filled_n; ++i) {
    pt381_filled.push_back(v2279[i % v2279.size()]);
  }
  auto pt381 = cc->MakeCKKSPackedPlaintext(pt381_filled);
  const auto& ct764 = cc->EvalMult(ct26, pt381);
  std::vector<float> v2280(std::begin(v28) + 382 * 512, std::begin(v28) + 382 * 512 + 1024);
  std::vector<float> v2281(656);
  std::copy(v2280.begin() + 0, v2280.begin() + 0 + 656, v2281.begin());
  std::vector<float> v2282(368);
  std::copy(v2280.begin() + 656, v2280.begin() + 656 + 368, v2282.begin());
  std::copy(v2281.begin(), v2281.end(), v86.begin() + 368);
  std::copy(v2282.begin(), v2282.end(), v86.begin() + 0);
  std::vector<double> v2285(std::begin(v86), std::end(v86));
  auto pt382_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt382_filled = v2285;
  pt382_filled.clear();
  pt382_filled.reserve(pt382_filled_n);
  for (auto i = 0; i < pt382_filled_n; ++i) {
    pt382_filled.push_back(v2285[i % v2285.size()]);
  }
  auto pt382 = cc->MakeCKKSPackedPlaintext(pt382_filled);
  const auto& ct765 = cc->EvalMult(ct28, pt382);
  std::vector<float> v2286(std::begin(v28) + 383 * 512, std::begin(v28) + 383 * 512 + 1024);
  std::vector<float> v2287(656);
  std::copy(v2286.begin() + 0, v2286.begin() + 0 + 656, v2287.begin());
  std::vector<float> v2288(368);
  std::copy(v2286.begin() + 656, v2286.begin() + 656 + 368, v2288.begin());
  std::copy(v2287.begin(), v2287.end(), v86.begin() + 368);
  std::copy(v2288.begin(), v2288.end(), v86.begin() + 0);
  std::vector<double> v2291(std::begin(v86), std::end(v86));
  auto pt383_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt383_filled = v2291;
  pt383_filled.clear();
  pt383_filled.reserve(pt383_filled_n);
  for (auto i = 0; i < pt383_filled_n; ++i) {
    pt383_filled.push_back(v2291[i % v2291.size()]);
  }
  auto pt383 = cc->MakeCKKSPackedPlaintext(pt383_filled);
  const auto& ct766 = cc->EvalMult(ct30, pt383);
  const auto& ct767 = cc->EvalAdd(ct751, ct752);
  const auto& ct768 = cc->EvalAdd(ct753, ct754);
  const auto& ct769 = cc->EvalAdd(ct767, ct768);
  const auto& ct770 = cc->EvalAdd(ct755, ct756);
  const auto& ct771 = cc->EvalAdd(ct757, ct758);
  const auto& ct772 = cc->EvalAdd(ct770, ct771);
  const auto& ct773 = cc->EvalAdd(ct769, ct772);
  const auto& ct774 = cc->EvalAdd(ct759, ct760);
  const auto& ct775 = cc->EvalAdd(ct761, ct762);
  const auto& ct776 = cc->EvalAdd(ct774, ct775);
  const auto& ct777 = cc->EvalAdd(ct763, ct764);
  const auto& ct778 = cc->EvalAdd(ct765, ct766);
  const auto& ct779 = cc->EvalAdd(ct777, ct778);
  const auto& ct780 = cc->EvalAdd(ct776, ct779);
  const auto& ct781 = cc->EvalAdd(ct773, ct780);
  const auto& ct782 = cc->EvalRotate(ct781, 368);
  std::vector<float> v2292(std::begin(v28) + 384 * 512, std::begin(v28) + 384 * 512 + 1024);
  std::vector<float> v2293(640);
  std::copy(v2292.begin() + 0, v2292.begin() + 0 + 640, v2293.begin());
  std::vector<float> v2294(384);
  std::copy(v2292.begin() + 640, v2292.begin() + 640 + 384, v2294.begin());
  std::copy(v2293.begin(), v2293.end(), v86.begin() + 384);
  std::copy(v2294.begin(), v2294.end(), v86.begin() + 0);
  std::vector<double> v2297(std::begin(v86), std::end(v86));
  auto pt384_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt384_filled = v2297;
  pt384_filled.clear();
  pt384_filled.reserve(pt384_filled_n);
  for (auto i = 0; i < pt384_filled_n; ++i) {
    pt384_filled.push_back(v2297[i % v2297.size()]);
  }
  auto pt384 = cc->MakeCKKSPackedPlaintext(pt384_filled);
  const auto& ct783 = cc->EvalMult(ct, pt384);
  std::vector<float> v2298(std::begin(v28) + 385 * 512, std::begin(v28) + 385 * 512 + 1024);
  std::vector<float> v2299(640);
  std::copy(v2298.begin() + 0, v2298.begin() + 0 + 640, v2299.begin());
  std::vector<float> v2300(384);
  std::copy(v2298.begin() + 640, v2298.begin() + 640 + 384, v2300.begin());
  std::copy(v2299.begin(), v2299.end(), v86.begin() + 384);
  std::copy(v2300.begin(), v2300.end(), v86.begin() + 0);
  std::vector<double> v2303(std::begin(v86), std::end(v86));
  auto pt385_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt385_filled = v2303;
  pt385_filled.clear();
  pt385_filled.reserve(pt385_filled_n);
  for (auto i = 0; i < pt385_filled_n; ++i) {
    pt385_filled.push_back(v2303[i % v2303.size()]);
  }
  auto pt385 = cc->MakeCKKSPackedPlaintext(pt385_filled);
  const auto& ct784 = cc->EvalMult(ct2, pt385);
  std::vector<float> v2304(std::begin(v28) + 386 * 512, std::begin(v28) + 386 * 512 + 1024);
  std::vector<float> v2305(640);
  std::copy(v2304.begin() + 0, v2304.begin() + 0 + 640, v2305.begin());
  std::vector<float> v2306(384);
  std::copy(v2304.begin() + 640, v2304.begin() + 640 + 384, v2306.begin());
  std::copy(v2305.begin(), v2305.end(), v86.begin() + 384);
  std::copy(v2306.begin(), v2306.end(), v86.begin() + 0);
  std::vector<double> v2309(std::begin(v86), std::end(v86));
  auto pt386_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt386_filled = v2309;
  pt386_filled.clear();
  pt386_filled.reserve(pt386_filled_n);
  for (auto i = 0; i < pt386_filled_n; ++i) {
    pt386_filled.push_back(v2309[i % v2309.size()]);
  }
  auto pt386 = cc->MakeCKKSPackedPlaintext(pt386_filled);
  const auto& ct785 = cc->EvalMult(ct4, pt386);
  std::vector<float> v2310(std::begin(v28) + 387 * 512, std::begin(v28) + 387 * 512 + 1024);
  std::vector<float> v2311(640);
  std::copy(v2310.begin() + 0, v2310.begin() + 0 + 640, v2311.begin());
  std::vector<float> v2312(384);
  std::copy(v2310.begin() + 640, v2310.begin() + 640 + 384, v2312.begin());
  std::copy(v2311.begin(), v2311.end(), v86.begin() + 384);
  std::copy(v2312.begin(), v2312.end(), v86.begin() + 0);
  std::vector<double> v2315(std::begin(v86), std::end(v86));
  auto pt387_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt387_filled = v2315;
  pt387_filled.clear();
  pt387_filled.reserve(pt387_filled_n);
  for (auto i = 0; i < pt387_filled_n; ++i) {
    pt387_filled.push_back(v2315[i % v2315.size()]);
  }
  auto pt387 = cc->MakeCKKSPackedPlaintext(pt387_filled);
  const auto& ct786 = cc->EvalMult(ct6, pt387);
  std::vector<float> v2316(std::begin(v28) + 388 * 512, std::begin(v28) + 388 * 512 + 1024);
  std::vector<float> v2317(640);
  std::copy(v2316.begin() + 0, v2316.begin() + 0 + 640, v2317.begin());
  std::vector<float> v2318(384);
  std::copy(v2316.begin() + 640, v2316.begin() + 640 + 384, v2318.begin());
  std::copy(v2317.begin(), v2317.end(), v86.begin() + 384);
  std::copy(v2318.begin(), v2318.end(), v86.begin() + 0);
  std::vector<double> v2321(std::begin(v86), std::end(v86));
  auto pt388_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt388_filled = v2321;
  pt388_filled.clear();
  pt388_filled.reserve(pt388_filled_n);
  for (auto i = 0; i < pt388_filled_n; ++i) {
    pt388_filled.push_back(v2321[i % v2321.size()]);
  }
  auto pt388 = cc->MakeCKKSPackedPlaintext(pt388_filled);
  const auto& ct787 = cc->EvalMult(ct8, pt388);
  std::vector<float> v2322(std::begin(v28) + 389 * 512, std::begin(v28) + 389 * 512 + 1024);
  std::vector<float> v2323(640);
  std::copy(v2322.begin() + 0, v2322.begin() + 0 + 640, v2323.begin());
  std::vector<float> v2324(384);
  std::copy(v2322.begin() + 640, v2322.begin() + 640 + 384, v2324.begin());
  std::copy(v2323.begin(), v2323.end(), v86.begin() + 384);
  std::copy(v2324.begin(), v2324.end(), v86.begin() + 0);
  std::vector<double> v2327(std::begin(v86), std::end(v86));
  auto pt389_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt389_filled = v2327;
  pt389_filled.clear();
  pt389_filled.reserve(pt389_filled_n);
  for (auto i = 0; i < pt389_filled_n; ++i) {
    pt389_filled.push_back(v2327[i % v2327.size()]);
  }
  auto pt389 = cc->MakeCKKSPackedPlaintext(pt389_filled);
  const auto& ct788 = cc->EvalMult(ct10, pt389);
  std::vector<float> v2328(std::begin(v28) + 390 * 512, std::begin(v28) + 390 * 512 + 1024);
  std::vector<float> v2329(640);
  std::copy(v2328.begin() + 0, v2328.begin() + 0 + 640, v2329.begin());
  std::vector<float> v2330(384);
  std::copy(v2328.begin() + 640, v2328.begin() + 640 + 384, v2330.begin());
  std::copy(v2329.begin(), v2329.end(), v86.begin() + 384);
  std::copy(v2330.begin(), v2330.end(), v86.begin() + 0);
  std::vector<double> v2333(std::begin(v86), std::end(v86));
  auto pt390_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt390_filled = v2333;
  pt390_filled.clear();
  pt390_filled.reserve(pt390_filled_n);
  for (auto i = 0; i < pt390_filled_n; ++i) {
    pt390_filled.push_back(v2333[i % v2333.size()]);
  }
  auto pt390 = cc->MakeCKKSPackedPlaintext(pt390_filled);
  const auto& ct789 = cc->EvalMult(ct12, pt390);
  std::vector<float> v2334(std::begin(v28) + 391 * 512, std::begin(v28) + 391 * 512 + 1024);
  std::vector<float> v2335(640);
  std::copy(v2334.begin() + 0, v2334.begin() + 0 + 640, v2335.begin());
  std::vector<float> v2336(384);
  std::copy(v2334.begin() + 640, v2334.begin() + 640 + 384, v2336.begin());
  std::copy(v2335.begin(), v2335.end(), v86.begin() + 384);
  std::copy(v2336.begin(), v2336.end(), v86.begin() + 0);
  std::vector<double> v2339(std::begin(v86), std::end(v86));
  auto pt391_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt391_filled = v2339;
  pt391_filled.clear();
  pt391_filled.reserve(pt391_filled_n);
  for (auto i = 0; i < pt391_filled_n; ++i) {
    pt391_filled.push_back(v2339[i % v2339.size()]);
  }
  auto pt391 = cc->MakeCKKSPackedPlaintext(pt391_filled);
  const auto& ct790 = cc->EvalMult(ct14, pt391);
  std::vector<float> v2340(std::begin(v28) + 392 * 512, std::begin(v28) + 392 * 512 + 1024);
  std::vector<float> v2341(640);
  std::copy(v2340.begin() + 0, v2340.begin() + 0 + 640, v2341.begin());
  std::vector<float> v2342(384);
  std::copy(v2340.begin() + 640, v2340.begin() + 640 + 384, v2342.begin());
  std::copy(v2341.begin(), v2341.end(), v86.begin() + 384);
  std::copy(v2342.begin(), v2342.end(), v86.begin() + 0);
  std::vector<double> v2345(std::begin(v86), std::end(v86));
  auto pt392_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt392_filled = v2345;
  pt392_filled.clear();
  pt392_filled.reserve(pt392_filled_n);
  for (auto i = 0; i < pt392_filled_n; ++i) {
    pt392_filled.push_back(v2345[i % v2345.size()]);
  }
  auto pt392 = cc->MakeCKKSPackedPlaintext(pt392_filled);
  const auto& ct791 = cc->EvalMult(ct16, pt392);
  std::vector<float> v2346(std::begin(v28) + 393 * 512, std::begin(v28) + 393 * 512 + 1024);
  std::vector<float> v2347(640);
  std::copy(v2346.begin() + 0, v2346.begin() + 0 + 640, v2347.begin());
  std::vector<float> v2348(384);
  std::copy(v2346.begin() + 640, v2346.begin() + 640 + 384, v2348.begin());
  std::copy(v2347.begin(), v2347.end(), v86.begin() + 384);
  std::copy(v2348.begin(), v2348.end(), v86.begin() + 0);
  std::vector<double> v2351(std::begin(v86), std::end(v86));
  auto pt393_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt393_filled = v2351;
  pt393_filled.clear();
  pt393_filled.reserve(pt393_filled_n);
  for (auto i = 0; i < pt393_filled_n; ++i) {
    pt393_filled.push_back(v2351[i % v2351.size()]);
  }
  auto pt393 = cc->MakeCKKSPackedPlaintext(pt393_filled);
  const auto& ct792 = cc->EvalMult(ct18, pt393);
  std::vector<float> v2352(std::begin(v28) + 394 * 512, std::begin(v28) + 394 * 512 + 1024);
  std::vector<float> v2353(640);
  std::copy(v2352.begin() + 0, v2352.begin() + 0 + 640, v2353.begin());
  std::vector<float> v2354(384);
  std::copy(v2352.begin() + 640, v2352.begin() + 640 + 384, v2354.begin());
  std::copy(v2353.begin(), v2353.end(), v86.begin() + 384);
  std::copy(v2354.begin(), v2354.end(), v86.begin() + 0);
  std::vector<double> v2357(std::begin(v86), std::end(v86));
  auto pt394_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt394_filled = v2357;
  pt394_filled.clear();
  pt394_filled.reserve(pt394_filled_n);
  for (auto i = 0; i < pt394_filled_n; ++i) {
    pt394_filled.push_back(v2357[i % v2357.size()]);
  }
  auto pt394 = cc->MakeCKKSPackedPlaintext(pt394_filled);
  const auto& ct793 = cc->EvalMult(ct20, pt394);
  std::vector<float> v2358(std::begin(v28) + 395 * 512, std::begin(v28) + 395 * 512 + 1024);
  std::vector<float> v2359(640);
  std::copy(v2358.begin() + 0, v2358.begin() + 0 + 640, v2359.begin());
  std::vector<float> v2360(384);
  std::copy(v2358.begin() + 640, v2358.begin() + 640 + 384, v2360.begin());
  std::copy(v2359.begin(), v2359.end(), v86.begin() + 384);
  std::copy(v2360.begin(), v2360.end(), v86.begin() + 0);
  std::vector<double> v2363(std::begin(v86), std::end(v86));
  auto pt395_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt395_filled = v2363;
  pt395_filled.clear();
  pt395_filled.reserve(pt395_filled_n);
  for (auto i = 0; i < pt395_filled_n; ++i) {
    pt395_filled.push_back(v2363[i % v2363.size()]);
  }
  auto pt395 = cc->MakeCKKSPackedPlaintext(pt395_filled);
  const auto& ct794 = cc->EvalMult(ct22, pt395);
  std::vector<float> v2364(std::begin(v28) + 396 * 512, std::begin(v28) + 396 * 512 + 1024);
  std::vector<float> v2365(640);
  std::copy(v2364.begin() + 0, v2364.begin() + 0 + 640, v2365.begin());
  std::vector<float> v2366(384);
  std::copy(v2364.begin() + 640, v2364.begin() + 640 + 384, v2366.begin());
  std::copy(v2365.begin(), v2365.end(), v86.begin() + 384);
  std::copy(v2366.begin(), v2366.end(), v86.begin() + 0);
  std::vector<double> v2369(std::begin(v86), std::end(v86));
  auto pt396_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt396_filled = v2369;
  pt396_filled.clear();
  pt396_filled.reserve(pt396_filled_n);
  for (auto i = 0; i < pt396_filled_n; ++i) {
    pt396_filled.push_back(v2369[i % v2369.size()]);
  }
  auto pt396 = cc->MakeCKKSPackedPlaintext(pt396_filled);
  const auto& ct795 = cc->EvalMult(ct24, pt396);
  std::vector<float> v2370(std::begin(v28) + 397 * 512, std::begin(v28) + 397 * 512 + 1024);
  std::vector<float> v2371(640);
  std::copy(v2370.begin() + 0, v2370.begin() + 0 + 640, v2371.begin());
  std::vector<float> v2372(384);
  std::copy(v2370.begin() + 640, v2370.begin() + 640 + 384, v2372.begin());
  std::copy(v2371.begin(), v2371.end(), v86.begin() + 384);
  std::copy(v2372.begin(), v2372.end(), v86.begin() + 0);
  std::vector<double> v2375(std::begin(v86), std::end(v86));
  auto pt397_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt397_filled = v2375;
  pt397_filled.clear();
  pt397_filled.reserve(pt397_filled_n);
  for (auto i = 0; i < pt397_filled_n; ++i) {
    pt397_filled.push_back(v2375[i % v2375.size()]);
  }
  auto pt397 = cc->MakeCKKSPackedPlaintext(pt397_filled);
  const auto& ct796 = cc->EvalMult(ct26, pt397);
  std::vector<float> v2376(std::begin(v28) + 398 * 512, std::begin(v28) + 398 * 512 + 1024);
  std::vector<float> v2377(640);
  std::copy(v2376.begin() + 0, v2376.begin() + 0 + 640, v2377.begin());
  std::vector<float> v2378(384);
  std::copy(v2376.begin() + 640, v2376.begin() + 640 + 384, v2378.begin());
  std::copy(v2377.begin(), v2377.end(), v86.begin() + 384);
  std::copy(v2378.begin(), v2378.end(), v86.begin() + 0);
  std::vector<double> v2381(std::begin(v86), std::end(v86));
  auto pt398_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt398_filled = v2381;
  pt398_filled.clear();
  pt398_filled.reserve(pt398_filled_n);
  for (auto i = 0; i < pt398_filled_n; ++i) {
    pt398_filled.push_back(v2381[i % v2381.size()]);
  }
  auto pt398 = cc->MakeCKKSPackedPlaintext(pt398_filled);
  const auto& ct797 = cc->EvalMult(ct28, pt398);
  std::vector<float> v2382(std::begin(v28) + 399 * 512, std::begin(v28) + 399 * 512 + 1024);
  std::vector<float> v2383(640);
  std::copy(v2382.begin() + 0, v2382.begin() + 0 + 640, v2383.begin());
  std::vector<float> v2384(384);
  std::copy(v2382.begin() + 640, v2382.begin() + 640 + 384, v2384.begin());
  std::copy(v2383.begin(), v2383.end(), v86.begin() + 384);
  std::copy(v2384.begin(), v2384.end(), v86.begin() + 0);
  std::vector<double> v2387(std::begin(v86), std::end(v86));
  auto pt399_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt399_filled = v2387;
  pt399_filled.clear();
  pt399_filled.reserve(pt399_filled_n);
  for (auto i = 0; i < pt399_filled_n; ++i) {
    pt399_filled.push_back(v2387[i % v2387.size()]);
  }
  auto pt399 = cc->MakeCKKSPackedPlaintext(pt399_filled);
  const auto& ct798 = cc->EvalMult(ct30, pt399);
  const auto& ct799 = cc->EvalAdd(ct783, ct784);
  const auto& ct800 = cc->EvalAdd(ct785, ct786);
  const auto& ct801 = cc->EvalAdd(ct799, ct800);
  const auto& ct802 = cc->EvalAdd(ct787, ct788);
  const auto& ct803 = cc->EvalAdd(ct789, ct790);
  const auto& ct804 = cc->EvalAdd(ct802, ct803);
  const auto& ct805 = cc->EvalAdd(ct801, ct804);
  const auto& ct806 = cc->EvalAdd(ct791, ct792);
  const auto& ct807 = cc->EvalAdd(ct793, ct794);
  const auto& ct808 = cc->EvalAdd(ct806, ct807);
  const auto& ct809 = cc->EvalAdd(ct795, ct796);
  const auto& ct810 = cc->EvalAdd(ct797, ct798);
  const auto& ct811 = cc->EvalAdd(ct809, ct810);
  const auto& ct812 = cc->EvalAdd(ct808, ct811);
  const auto& ct813 = cc->EvalAdd(ct805, ct812);
  const auto& ct814 = cc->EvalRotate(ct813, 384);
  std::vector<float> v2388(std::begin(v28) + 400 * 512, std::begin(v28) + 400 * 512 + 1024);
  std::vector<float> v2389(624);
  std::copy(v2388.begin() + 0, v2388.begin() + 0 + 624, v2389.begin());
  std::vector<float> v2390(400);
  std::copy(v2388.begin() + 624, v2388.begin() + 624 + 400, v2390.begin());
  std::copy(v2389.begin(), v2389.end(), v86.begin() + 400);
  std::copy(v2390.begin(), v2390.end(), v86.begin() + 0);
  std::vector<double> v2393(std::begin(v86), std::end(v86));
  auto pt400_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt400_filled = v2393;
  pt400_filled.clear();
  pt400_filled.reserve(pt400_filled_n);
  for (auto i = 0; i < pt400_filled_n; ++i) {
    pt400_filled.push_back(v2393[i % v2393.size()]);
  }
  auto pt400 = cc->MakeCKKSPackedPlaintext(pt400_filled);
  const auto& ct815 = cc->EvalMult(ct, pt400);
  std::vector<float> v2394(std::begin(v28) + 401 * 512, std::begin(v28) + 401 * 512 + 1024);
  std::vector<float> v2395(624);
  std::copy(v2394.begin() + 0, v2394.begin() + 0 + 624, v2395.begin());
  std::vector<float> v2396(400);
  std::copy(v2394.begin() + 624, v2394.begin() + 624 + 400, v2396.begin());
  std::copy(v2395.begin(), v2395.end(), v86.begin() + 400);
  std::copy(v2396.begin(), v2396.end(), v86.begin() + 0);
  std::vector<double> v2399(std::begin(v86), std::end(v86));
  auto pt401_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt401_filled = v2399;
  pt401_filled.clear();
  pt401_filled.reserve(pt401_filled_n);
  for (auto i = 0; i < pt401_filled_n; ++i) {
    pt401_filled.push_back(v2399[i % v2399.size()]);
  }
  auto pt401 = cc->MakeCKKSPackedPlaintext(pt401_filled);
  const auto& ct816 = cc->EvalMult(ct2, pt401);
  std::vector<float> v2400(std::begin(v28) + 402 * 512, std::begin(v28) + 402 * 512 + 1024);
  std::vector<float> v2401(624);
  std::copy(v2400.begin() + 0, v2400.begin() + 0 + 624, v2401.begin());
  std::vector<float> v2402(400);
  std::copy(v2400.begin() + 624, v2400.begin() + 624 + 400, v2402.begin());
  std::copy(v2401.begin(), v2401.end(), v86.begin() + 400);
  std::copy(v2402.begin(), v2402.end(), v86.begin() + 0);
  std::vector<double> v2405(std::begin(v86), std::end(v86));
  auto pt402_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt402_filled = v2405;
  pt402_filled.clear();
  pt402_filled.reserve(pt402_filled_n);
  for (auto i = 0; i < pt402_filled_n; ++i) {
    pt402_filled.push_back(v2405[i % v2405.size()]);
  }
  auto pt402 = cc->MakeCKKSPackedPlaintext(pt402_filled);
  const auto& ct817 = cc->EvalMult(ct4, pt402);
  std::vector<float> v2406(std::begin(v28) + 403 * 512, std::begin(v28) + 403 * 512 + 1024);
  std::vector<float> v2407(624);
  std::copy(v2406.begin() + 0, v2406.begin() + 0 + 624, v2407.begin());
  std::vector<float> v2408(400);
  std::copy(v2406.begin() + 624, v2406.begin() + 624 + 400, v2408.begin());
  std::copy(v2407.begin(), v2407.end(), v86.begin() + 400);
  std::copy(v2408.begin(), v2408.end(), v86.begin() + 0);
  std::vector<double> v2411(std::begin(v86), std::end(v86));
  auto pt403_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt403_filled = v2411;
  pt403_filled.clear();
  pt403_filled.reserve(pt403_filled_n);
  for (auto i = 0; i < pt403_filled_n; ++i) {
    pt403_filled.push_back(v2411[i % v2411.size()]);
  }
  auto pt403 = cc->MakeCKKSPackedPlaintext(pt403_filled);
  const auto& ct818 = cc->EvalMult(ct6, pt403);
  std::vector<float> v2412(std::begin(v28) + 404 * 512, std::begin(v28) + 404 * 512 + 1024);
  std::vector<float> v2413(624);
  std::copy(v2412.begin() + 0, v2412.begin() + 0 + 624, v2413.begin());
  std::vector<float> v2414(400);
  std::copy(v2412.begin() + 624, v2412.begin() + 624 + 400, v2414.begin());
  std::copy(v2413.begin(), v2413.end(), v86.begin() + 400);
  std::copy(v2414.begin(), v2414.end(), v86.begin() + 0);
  std::vector<double> v2417(std::begin(v86), std::end(v86));
  auto pt404_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt404_filled = v2417;
  pt404_filled.clear();
  pt404_filled.reserve(pt404_filled_n);
  for (auto i = 0; i < pt404_filled_n; ++i) {
    pt404_filled.push_back(v2417[i % v2417.size()]);
  }
  auto pt404 = cc->MakeCKKSPackedPlaintext(pt404_filled);
  const auto& ct819 = cc->EvalMult(ct8, pt404);
  std::vector<float> v2418(std::begin(v28) + 405 * 512, std::begin(v28) + 405 * 512 + 1024);
  std::vector<float> v2419(624);
  std::copy(v2418.begin() + 0, v2418.begin() + 0 + 624, v2419.begin());
  std::vector<float> v2420(400);
  std::copy(v2418.begin() + 624, v2418.begin() + 624 + 400, v2420.begin());
  std::copy(v2419.begin(), v2419.end(), v86.begin() + 400);
  std::copy(v2420.begin(), v2420.end(), v86.begin() + 0);
  std::vector<double> v2423(std::begin(v86), std::end(v86));
  auto pt405_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt405_filled = v2423;
  pt405_filled.clear();
  pt405_filled.reserve(pt405_filled_n);
  for (auto i = 0; i < pt405_filled_n; ++i) {
    pt405_filled.push_back(v2423[i % v2423.size()]);
  }
  auto pt405 = cc->MakeCKKSPackedPlaintext(pt405_filled);
  const auto& ct820 = cc->EvalMult(ct10, pt405);
  std::vector<float> v2424(std::begin(v28) + 406 * 512, std::begin(v28) + 406 * 512 + 1024);
  std::vector<float> v2425(624);
  std::copy(v2424.begin() + 0, v2424.begin() + 0 + 624, v2425.begin());
  std::vector<float> v2426(400);
  std::copy(v2424.begin() + 624, v2424.begin() + 624 + 400, v2426.begin());
  std::copy(v2425.begin(), v2425.end(), v86.begin() + 400);
  std::copy(v2426.begin(), v2426.end(), v86.begin() + 0);
  std::vector<double> v2429(std::begin(v86), std::end(v86));
  auto pt406_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt406_filled = v2429;
  pt406_filled.clear();
  pt406_filled.reserve(pt406_filled_n);
  for (auto i = 0; i < pt406_filled_n; ++i) {
    pt406_filled.push_back(v2429[i % v2429.size()]);
  }
  auto pt406 = cc->MakeCKKSPackedPlaintext(pt406_filled);
  const auto& ct821 = cc->EvalMult(ct12, pt406);
  std::vector<float> v2430(std::begin(v28) + 407 * 512, std::begin(v28) + 407 * 512 + 1024);
  std::vector<float> v2431(624);
  std::copy(v2430.begin() + 0, v2430.begin() + 0 + 624, v2431.begin());
  std::vector<float> v2432(400);
  std::copy(v2430.begin() + 624, v2430.begin() + 624 + 400, v2432.begin());
  std::copy(v2431.begin(), v2431.end(), v86.begin() + 400);
  std::copy(v2432.begin(), v2432.end(), v86.begin() + 0);
  std::vector<double> v2435(std::begin(v86), std::end(v86));
  auto pt407_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt407_filled = v2435;
  pt407_filled.clear();
  pt407_filled.reserve(pt407_filled_n);
  for (auto i = 0; i < pt407_filled_n; ++i) {
    pt407_filled.push_back(v2435[i % v2435.size()]);
  }
  auto pt407 = cc->MakeCKKSPackedPlaintext(pt407_filled);
  const auto& ct822 = cc->EvalMult(ct14, pt407);
  std::vector<float> v2436(std::begin(v28) + 408 * 512, std::begin(v28) + 408 * 512 + 1024);
  std::vector<float> v2437(624);
  std::copy(v2436.begin() + 0, v2436.begin() + 0 + 624, v2437.begin());
  std::vector<float> v2438(400);
  std::copy(v2436.begin() + 624, v2436.begin() + 624 + 400, v2438.begin());
  std::copy(v2437.begin(), v2437.end(), v86.begin() + 400);
  std::copy(v2438.begin(), v2438.end(), v86.begin() + 0);
  std::vector<double> v2441(std::begin(v86), std::end(v86));
  auto pt408_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt408_filled = v2441;
  pt408_filled.clear();
  pt408_filled.reserve(pt408_filled_n);
  for (auto i = 0; i < pt408_filled_n; ++i) {
    pt408_filled.push_back(v2441[i % v2441.size()]);
  }
  auto pt408 = cc->MakeCKKSPackedPlaintext(pt408_filled);
  const auto& ct823 = cc->EvalMult(ct16, pt408);
  std::vector<float> v2442(std::begin(v28) + 409 * 512, std::begin(v28) + 409 * 512 + 1024);
  std::vector<float> v2443(624);
  std::copy(v2442.begin() + 0, v2442.begin() + 0 + 624, v2443.begin());
  std::vector<float> v2444(400);
  std::copy(v2442.begin() + 624, v2442.begin() + 624 + 400, v2444.begin());
  std::copy(v2443.begin(), v2443.end(), v86.begin() + 400);
  std::copy(v2444.begin(), v2444.end(), v86.begin() + 0);
  std::vector<double> v2447(std::begin(v86), std::end(v86));
  auto pt409_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt409_filled = v2447;
  pt409_filled.clear();
  pt409_filled.reserve(pt409_filled_n);
  for (auto i = 0; i < pt409_filled_n; ++i) {
    pt409_filled.push_back(v2447[i % v2447.size()]);
  }
  auto pt409 = cc->MakeCKKSPackedPlaintext(pt409_filled);
  const auto& ct824 = cc->EvalMult(ct18, pt409);
  std::vector<float> v2448(std::begin(v28) + 410 * 512, std::begin(v28) + 410 * 512 + 1024);
  std::vector<float> v2449(624);
  std::copy(v2448.begin() + 0, v2448.begin() + 0 + 624, v2449.begin());
  std::vector<float> v2450(400);
  std::copy(v2448.begin() + 624, v2448.begin() + 624 + 400, v2450.begin());
  std::copy(v2449.begin(), v2449.end(), v86.begin() + 400);
  std::copy(v2450.begin(), v2450.end(), v86.begin() + 0);
  std::vector<double> v2453(std::begin(v86), std::end(v86));
  auto pt410_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt410_filled = v2453;
  pt410_filled.clear();
  pt410_filled.reserve(pt410_filled_n);
  for (auto i = 0; i < pt410_filled_n; ++i) {
    pt410_filled.push_back(v2453[i % v2453.size()]);
  }
  auto pt410 = cc->MakeCKKSPackedPlaintext(pt410_filled);
  const auto& ct825 = cc->EvalMult(ct20, pt410);
  std::vector<float> v2454(std::begin(v28) + 411 * 512, std::begin(v28) + 411 * 512 + 1024);
  std::vector<float> v2455(624);
  std::copy(v2454.begin() + 0, v2454.begin() + 0 + 624, v2455.begin());
  std::vector<float> v2456(400);
  std::copy(v2454.begin() + 624, v2454.begin() + 624 + 400, v2456.begin());
  std::copy(v2455.begin(), v2455.end(), v86.begin() + 400);
  std::copy(v2456.begin(), v2456.end(), v86.begin() + 0);
  std::vector<double> v2459(std::begin(v86), std::end(v86));
  auto pt411_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt411_filled = v2459;
  pt411_filled.clear();
  pt411_filled.reserve(pt411_filled_n);
  for (auto i = 0; i < pt411_filled_n; ++i) {
    pt411_filled.push_back(v2459[i % v2459.size()]);
  }
  auto pt411 = cc->MakeCKKSPackedPlaintext(pt411_filled);
  const auto& ct826 = cc->EvalMult(ct22, pt411);
  std::vector<float> v2460(std::begin(v28) + 412 * 512, std::begin(v28) + 412 * 512 + 1024);
  std::vector<float> v2461(624);
  std::copy(v2460.begin() + 0, v2460.begin() + 0 + 624, v2461.begin());
  std::vector<float> v2462(400);
  std::copy(v2460.begin() + 624, v2460.begin() + 624 + 400, v2462.begin());
  std::copy(v2461.begin(), v2461.end(), v86.begin() + 400);
  std::copy(v2462.begin(), v2462.end(), v86.begin() + 0);
  std::vector<double> v2465(std::begin(v86), std::end(v86));
  auto pt412_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt412_filled = v2465;
  pt412_filled.clear();
  pt412_filled.reserve(pt412_filled_n);
  for (auto i = 0; i < pt412_filled_n; ++i) {
    pt412_filled.push_back(v2465[i % v2465.size()]);
  }
  auto pt412 = cc->MakeCKKSPackedPlaintext(pt412_filled);
  const auto& ct827 = cc->EvalMult(ct24, pt412);
  std::vector<float> v2466(std::begin(v28) + 413 * 512, std::begin(v28) + 413 * 512 + 1024);
  std::vector<float> v2467(624);
  std::copy(v2466.begin() + 0, v2466.begin() + 0 + 624, v2467.begin());
  std::vector<float> v2468(400);
  std::copy(v2466.begin() + 624, v2466.begin() + 624 + 400, v2468.begin());
  std::copy(v2467.begin(), v2467.end(), v86.begin() + 400);
  std::copy(v2468.begin(), v2468.end(), v86.begin() + 0);
  std::vector<double> v2471(std::begin(v86), std::end(v86));
  auto pt413_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt413_filled = v2471;
  pt413_filled.clear();
  pt413_filled.reserve(pt413_filled_n);
  for (auto i = 0; i < pt413_filled_n; ++i) {
    pt413_filled.push_back(v2471[i % v2471.size()]);
  }
  auto pt413 = cc->MakeCKKSPackedPlaintext(pt413_filled);
  const auto& ct828 = cc->EvalMult(ct26, pt413);
  std::vector<float> v2472(std::begin(v28) + 414 * 512, std::begin(v28) + 414 * 512 + 1024);
  std::vector<float> v2473(624);
  std::copy(v2472.begin() + 0, v2472.begin() + 0 + 624, v2473.begin());
  std::vector<float> v2474(400);
  std::copy(v2472.begin() + 624, v2472.begin() + 624 + 400, v2474.begin());
  std::copy(v2473.begin(), v2473.end(), v86.begin() + 400);
  std::copy(v2474.begin(), v2474.end(), v86.begin() + 0);
  std::vector<double> v2477(std::begin(v86), std::end(v86));
  auto pt414_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt414_filled = v2477;
  pt414_filled.clear();
  pt414_filled.reserve(pt414_filled_n);
  for (auto i = 0; i < pt414_filled_n; ++i) {
    pt414_filled.push_back(v2477[i % v2477.size()]);
  }
  auto pt414 = cc->MakeCKKSPackedPlaintext(pt414_filled);
  const auto& ct829 = cc->EvalMult(ct28, pt414);
  std::vector<float> v2478(std::begin(v28) + 415 * 512, std::begin(v28) + 415 * 512 + 1024);
  std::vector<float> v2479(624);
  std::copy(v2478.begin() + 0, v2478.begin() + 0 + 624, v2479.begin());
  std::vector<float> v2480(400);
  std::copy(v2478.begin() + 624, v2478.begin() + 624 + 400, v2480.begin());
  std::copy(v2479.begin(), v2479.end(), v86.begin() + 400);
  std::copy(v2480.begin(), v2480.end(), v86.begin() + 0);
  std::vector<double> v2483(std::begin(v86), std::end(v86));
  auto pt415_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt415_filled = v2483;
  pt415_filled.clear();
  pt415_filled.reserve(pt415_filled_n);
  for (auto i = 0; i < pt415_filled_n; ++i) {
    pt415_filled.push_back(v2483[i % v2483.size()]);
  }
  auto pt415 = cc->MakeCKKSPackedPlaintext(pt415_filled);
  const auto& ct830 = cc->EvalMult(ct30, pt415);
  const auto& ct831 = cc->EvalAdd(ct815, ct816);
  const auto& ct832 = cc->EvalAdd(ct817, ct818);
  const auto& ct833 = cc->EvalAdd(ct831, ct832);
  const auto& ct834 = cc->EvalAdd(ct819, ct820);
  const auto& ct835 = cc->EvalAdd(ct821, ct822);
  const auto& ct836 = cc->EvalAdd(ct834, ct835);
  const auto& ct837 = cc->EvalAdd(ct833, ct836);
  const auto& ct838 = cc->EvalAdd(ct823, ct824);
  const auto& ct839 = cc->EvalAdd(ct825, ct826);
  const auto& ct840 = cc->EvalAdd(ct838, ct839);
  const auto& ct841 = cc->EvalAdd(ct827, ct828);
  const auto& ct842 = cc->EvalAdd(ct829, ct830);
  const auto& ct843 = cc->EvalAdd(ct841, ct842);
  const auto& ct844 = cc->EvalAdd(ct840, ct843);
  const auto& ct845 = cc->EvalAdd(ct837, ct844);
  const auto& ct846 = cc->EvalRotate(ct845, 400);
  std::vector<float> v2484(std::begin(v28) + 416 * 512, std::begin(v28) + 416 * 512 + 1024);
  std::vector<float> v2485(608);
  std::copy(v2484.begin() + 0, v2484.begin() + 0 + 608, v2485.begin());
  std::vector<float> v2486(416);
  std::copy(v2484.begin() + 608, v2484.begin() + 608 + 416, v2486.begin());
  std::copy(v2485.begin(), v2485.end(), v86.begin() + 416);
  std::copy(v2486.begin(), v2486.end(), v86.begin() + 0);
  std::vector<double> v2489(std::begin(v86), std::end(v86));
  auto pt416_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt416_filled = v2489;
  pt416_filled.clear();
  pt416_filled.reserve(pt416_filled_n);
  for (auto i = 0; i < pt416_filled_n; ++i) {
    pt416_filled.push_back(v2489[i % v2489.size()]);
  }
  auto pt416 = cc->MakeCKKSPackedPlaintext(pt416_filled);
  const auto& ct847 = cc->EvalMult(ct, pt416);
  std::vector<float> v2490(std::begin(v28) + 417 * 512, std::begin(v28) + 417 * 512 + 1024);
  std::vector<float> v2491(608);
  std::copy(v2490.begin() + 0, v2490.begin() + 0 + 608, v2491.begin());
  std::vector<float> v2492(416);
  std::copy(v2490.begin() + 608, v2490.begin() + 608 + 416, v2492.begin());
  std::copy(v2491.begin(), v2491.end(), v86.begin() + 416);
  std::copy(v2492.begin(), v2492.end(), v86.begin() + 0);
  std::vector<double> v2495(std::begin(v86), std::end(v86));
  auto pt417_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt417_filled = v2495;
  pt417_filled.clear();
  pt417_filled.reserve(pt417_filled_n);
  for (auto i = 0; i < pt417_filled_n; ++i) {
    pt417_filled.push_back(v2495[i % v2495.size()]);
  }
  auto pt417 = cc->MakeCKKSPackedPlaintext(pt417_filled);
  const auto& ct848 = cc->EvalMult(ct2, pt417);
  std::vector<float> v2496(std::begin(v28) + 418 * 512, std::begin(v28) + 418 * 512 + 1024);
  std::vector<float> v2497(608);
  std::copy(v2496.begin() + 0, v2496.begin() + 0 + 608, v2497.begin());
  std::vector<float> v2498(416);
  std::copy(v2496.begin() + 608, v2496.begin() + 608 + 416, v2498.begin());
  std::copy(v2497.begin(), v2497.end(), v86.begin() + 416);
  std::copy(v2498.begin(), v2498.end(), v86.begin() + 0);
  std::vector<double> v2501(std::begin(v86), std::end(v86));
  auto pt418_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt418_filled = v2501;
  pt418_filled.clear();
  pt418_filled.reserve(pt418_filled_n);
  for (auto i = 0; i < pt418_filled_n; ++i) {
    pt418_filled.push_back(v2501[i % v2501.size()]);
  }
  auto pt418 = cc->MakeCKKSPackedPlaintext(pt418_filled);
  const auto& ct849 = cc->EvalMult(ct4, pt418);
  std::vector<float> v2502(std::begin(v28) + 419 * 512, std::begin(v28) + 419 * 512 + 1024);
  std::vector<float> v2503(608);
  std::copy(v2502.begin() + 0, v2502.begin() + 0 + 608, v2503.begin());
  std::vector<float> v2504(416);
  std::copy(v2502.begin() + 608, v2502.begin() + 608 + 416, v2504.begin());
  std::copy(v2503.begin(), v2503.end(), v86.begin() + 416);
  std::copy(v2504.begin(), v2504.end(), v86.begin() + 0);
  std::vector<double> v2507(std::begin(v86), std::end(v86));
  auto pt419_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt419_filled = v2507;
  pt419_filled.clear();
  pt419_filled.reserve(pt419_filled_n);
  for (auto i = 0; i < pt419_filled_n; ++i) {
    pt419_filled.push_back(v2507[i % v2507.size()]);
  }
  auto pt419 = cc->MakeCKKSPackedPlaintext(pt419_filled);
  const auto& ct850 = cc->EvalMult(ct6, pt419);
  std::vector<float> v2508(std::begin(v28) + 420 * 512, std::begin(v28) + 420 * 512 + 1024);
  std::vector<float> v2509(608);
  std::copy(v2508.begin() + 0, v2508.begin() + 0 + 608, v2509.begin());
  std::vector<float> v2510(416);
  std::copy(v2508.begin() + 608, v2508.begin() + 608 + 416, v2510.begin());
  std::copy(v2509.begin(), v2509.end(), v86.begin() + 416);
  std::copy(v2510.begin(), v2510.end(), v86.begin() + 0);
  std::vector<double> v2513(std::begin(v86), std::end(v86));
  auto pt420_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt420_filled = v2513;
  pt420_filled.clear();
  pt420_filled.reserve(pt420_filled_n);
  for (auto i = 0; i < pt420_filled_n; ++i) {
    pt420_filled.push_back(v2513[i % v2513.size()]);
  }
  auto pt420 = cc->MakeCKKSPackedPlaintext(pt420_filled);
  const auto& ct851 = cc->EvalMult(ct8, pt420);
  std::vector<float> v2514(std::begin(v28) + 421 * 512, std::begin(v28) + 421 * 512 + 1024);
  std::vector<float> v2515(608);
  std::copy(v2514.begin() + 0, v2514.begin() + 0 + 608, v2515.begin());
  std::vector<float> v2516(416);
  std::copy(v2514.begin() + 608, v2514.begin() + 608 + 416, v2516.begin());
  std::copy(v2515.begin(), v2515.end(), v86.begin() + 416);
  std::copy(v2516.begin(), v2516.end(), v86.begin() + 0);
  std::vector<double> v2519(std::begin(v86), std::end(v86));
  auto pt421_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt421_filled = v2519;
  pt421_filled.clear();
  pt421_filled.reserve(pt421_filled_n);
  for (auto i = 0; i < pt421_filled_n; ++i) {
    pt421_filled.push_back(v2519[i % v2519.size()]);
  }
  auto pt421 = cc->MakeCKKSPackedPlaintext(pt421_filled);
  const auto& ct852 = cc->EvalMult(ct10, pt421);
  std::vector<float> v2520(std::begin(v28) + 422 * 512, std::begin(v28) + 422 * 512 + 1024);
  std::vector<float> v2521(608);
  std::copy(v2520.begin() + 0, v2520.begin() + 0 + 608, v2521.begin());
  std::vector<float> v2522(416);
  std::copy(v2520.begin() + 608, v2520.begin() + 608 + 416, v2522.begin());
  std::copy(v2521.begin(), v2521.end(), v86.begin() + 416);
  std::copy(v2522.begin(), v2522.end(), v86.begin() + 0);
  std::vector<double> v2525(std::begin(v86), std::end(v86));
  auto pt422_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt422_filled = v2525;
  pt422_filled.clear();
  pt422_filled.reserve(pt422_filled_n);
  for (auto i = 0; i < pt422_filled_n; ++i) {
    pt422_filled.push_back(v2525[i % v2525.size()]);
  }
  auto pt422 = cc->MakeCKKSPackedPlaintext(pt422_filled);
  const auto& ct853 = cc->EvalMult(ct12, pt422);
  std::vector<float> v2526(std::begin(v28) + 423 * 512, std::begin(v28) + 423 * 512 + 1024);
  std::vector<float> v2527(608);
  std::copy(v2526.begin() + 0, v2526.begin() + 0 + 608, v2527.begin());
  std::vector<float> v2528(416);
  std::copy(v2526.begin() + 608, v2526.begin() + 608 + 416, v2528.begin());
  std::copy(v2527.begin(), v2527.end(), v86.begin() + 416);
  std::copy(v2528.begin(), v2528.end(), v86.begin() + 0);
  std::vector<double> v2531(std::begin(v86), std::end(v86));
  auto pt423_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt423_filled = v2531;
  pt423_filled.clear();
  pt423_filled.reserve(pt423_filled_n);
  for (auto i = 0; i < pt423_filled_n; ++i) {
    pt423_filled.push_back(v2531[i % v2531.size()]);
  }
  auto pt423 = cc->MakeCKKSPackedPlaintext(pt423_filled);
  const auto& ct854 = cc->EvalMult(ct14, pt423);
  std::vector<float> v2532(std::begin(v28) + 424 * 512, std::begin(v28) + 424 * 512 + 1024);
  std::vector<float> v2533(608);
  std::copy(v2532.begin() + 0, v2532.begin() + 0 + 608, v2533.begin());
  std::vector<float> v2534(416);
  std::copy(v2532.begin() + 608, v2532.begin() + 608 + 416, v2534.begin());
  std::copy(v2533.begin(), v2533.end(), v86.begin() + 416);
  std::copy(v2534.begin(), v2534.end(), v86.begin() + 0);
  std::vector<double> v2537(std::begin(v86), std::end(v86));
  auto pt424_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt424_filled = v2537;
  pt424_filled.clear();
  pt424_filled.reserve(pt424_filled_n);
  for (auto i = 0; i < pt424_filled_n; ++i) {
    pt424_filled.push_back(v2537[i % v2537.size()]);
  }
  auto pt424 = cc->MakeCKKSPackedPlaintext(pt424_filled);
  const auto& ct855 = cc->EvalMult(ct16, pt424);
  std::vector<float> v2538(std::begin(v28) + 425 * 512, std::begin(v28) + 425 * 512 + 1024);
  std::vector<float> v2539(608);
  std::copy(v2538.begin() + 0, v2538.begin() + 0 + 608, v2539.begin());
  std::vector<float> v2540(416);
  std::copy(v2538.begin() + 608, v2538.begin() + 608 + 416, v2540.begin());
  std::copy(v2539.begin(), v2539.end(), v86.begin() + 416);
  std::copy(v2540.begin(), v2540.end(), v86.begin() + 0);
  std::vector<double> v2543(std::begin(v86), std::end(v86));
  auto pt425_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt425_filled = v2543;
  pt425_filled.clear();
  pt425_filled.reserve(pt425_filled_n);
  for (auto i = 0; i < pt425_filled_n; ++i) {
    pt425_filled.push_back(v2543[i % v2543.size()]);
  }
  auto pt425 = cc->MakeCKKSPackedPlaintext(pt425_filled);
  const auto& ct856 = cc->EvalMult(ct18, pt425);
  std::vector<float> v2544(std::begin(v28) + 426 * 512, std::begin(v28) + 426 * 512 + 1024);
  std::vector<float> v2545(608);
  std::copy(v2544.begin() + 0, v2544.begin() + 0 + 608, v2545.begin());
  std::vector<float> v2546(416);
  std::copy(v2544.begin() + 608, v2544.begin() + 608 + 416, v2546.begin());
  std::copy(v2545.begin(), v2545.end(), v86.begin() + 416);
  std::copy(v2546.begin(), v2546.end(), v86.begin() + 0);
  std::vector<double> v2549(std::begin(v86), std::end(v86));
  auto pt426_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt426_filled = v2549;
  pt426_filled.clear();
  pt426_filled.reserve(pt426_filled_n);
  for (auto i = 0; i < pt426_filled_n; ++i) {
    pt426_filled.push_back(v2549[i % v2549.size()]);
  }
  auto pt426 = cc->MakeCKKSPackedPlaintext(pt426_filled);
  const auto& ct857 = cc->EvalMult(ct20, pt426);
  std::vector<float> v2550(std::begin(v28) + 427 * 512, std::begin(v28) + 427 * 512 + 1024);
  std::vector<float> v2551(608);
  std::copy(v2550.begin() + 0, v2550.begin() + 0 + 608, v2551.begin());
  std::vector<float> v2552(416);
  std::copy(v2550.begin() + 608, v2550.begin() + 608 + 416, v2552.begin());
  std::copy(v2551.begin(), v2551.end(), v86.begin() + 416);
  std::copy(v2552.begin(), v2552.end(), v86.begin() + 0);
  std::vector<double> v2555(std::begin(v86), std::end(v86));
  auto pt427_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt427_filled = v2555;
  pt427_filled.clear();
  pt427_filled.reserve(pt427_filled_n);
  for (auto i = 0; i < pt427_filled_n; ++i) {
    pt427_filled.push_back(v2555[i % v2555.size()]);
  }
  auto pt427 = cc->MakeCKKSPackedPlaintext(pt427_filled);
  const auto& ct858 = cc->EvalMult(ct22, pt427);
  std::vector<float> v2556(std::begin(v28) + 428 * 512, std::begin(v28) + 428 * 512 + 1024);
  std::vector<float> v2557(608);
  std::copy(v2556.begin() + 0, v2556.begin() + 0 + 608, v2557.begin());
  std::vector<float> v2558(416);
  std::copy(v2556.begin() + 608, v2556.begin() + 608 + 416, v2558.begin());
  std::copy(v2557.begin(), v2557.end(), v86.begin() + 416);
  std::copy(v2558.begin(), v2558.end(), v86.begin() + 0);
  std::vector<double> v2561(std::begin(v86), std::end(v86));
  auto pt428_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt428_filled = v2561;
  pt428_filled.clear();
  pt428_filled.reserve(pt428_filled_n);
  for (auto i = 0; i < pt428_filled_n; ++i) {
    pt428_filled.push_back(v2561[i % v2561.size()]);
  }
  auto pt428 = cc->MakeCKKSPackedPlaintext(pt428_filled);
  const auto& ct859 = cc->EvalMult(ct24, pt428);
  std::vector<float> v2562(std::begin(v28) + 429 * 512, std::begin(v28) + 429 * 512 + 1024);
  std::vector<float> v2563(608);
  std::copy(v2562.begin() + 0, v2562.begin() + 0 + 608, v2563.begin());
  std::vector<float> v2564(416);
  std::copy(v2562.begin() + 608, v2562.begin() + 608 + 416, v2564.begin());
  std::copy(v2563.begin(), v2563.end(), v86.begin() + 416);
  std::copy(v2564.begin(), v2564.end(), v86.begin() + 0);
  std::vector<double> v2567(std::begin(v86), std::end(v86));
  auto pt429_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt429_filled = v2567;
  pt429_filled.clear();
  pt429_filled.reserve(pt429_filled_n);
  for (auto i = 0; i < pt429_filled_n; ++i) {
    pt429_filled.push_back(v2567[i % v2567.size()]);
  }
  auto pt429 = cc->MakeCKKSPackedPlaintext(pt429_filled);
  const auto& ct860 = cc->EvalMult(ct26, pt429);
  std::vector<float> v2568(std::begin(v28) + 430 * 512, std::begin(v28) + 430 * 512 + 1024);
  std::vector<float> v2569(608);
  std::copy(v2568.begin() + 0, v2568.begin() + 0 + 608, v2569.begin());
  std::vector<float> v2570(416);
  std::copy(v2568.begin() + 608, v2568.begin() + 608 + 416, v2570.begin());
  std::copy(v2569.begin(), v2569.end(), v86.begin() + 416);
  std::copy(v2570.begin(), v2570.end(), v86.begin() + 0);
  std::vector<double> v2573(std::begin(v86), std::end(v86));
  auto pt430_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt430_filled = v2573;
  pt430_filled.clear();
  pt430_filled.reserve(pt430_filled_n);
  for (auto i = 0; i < pt430_filled_n; ++i) {
    pt430_filled.push_back(v2573[i % v2573.size()]);
  }
  auto pt430 = cc->MakeCKKSPackedPlaintext(pt430_filled);
  const auto& ct861 = cc->EvalMult(ct28, pt430);
  std::vector<float> v2574(std::begin(v28) + 431 * 512, std::begin(v28) + 431 * 512 + 1024);
  std::vector<float> v2575(608);
  std::copy(v2574.begin() + 0, v2574.begin() + 0 + 608, v2575.begin());
  std::vector<float> v2576(416);
  std::copy(v2574.begin() + 608, v2574.begin() + 608 + 416, v2576.begin());
  std::copy(v2575.begin(), v2575.end(), v86.begin() + 416);
  std::copy(v2576.begin(), v2576.end(), v86.begin() + 0);
  std::vector<double> v2579(std::begin(v86), std::end(v86));
  auto pt431_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt431_filled = v2579;
  pt431_filled.clear();
  pt431_filled.reserve(pt431_filled_n);
  for (auto i = 0; i < pt431_filled_n; ++i) {
    pt431_filled.push_back(v2579[i % v2579.size()]);
  }
  auto pt431 = cc->MakeCKKSPackedPlaintext(pt431_filled);
  const auto& ct862 = cc->EvalMult(ct30, pt431);
  const auto& ct863 = cc->EvalAdd(ct847, ct848);
  const auto& ct864 = cc->EvalAdd(ct849, ct850);
  const auto& ct865 = cc->EvalAdd(ct863, ct864);
  const auto& ct866 = cc->EvalAdd(ct851, ct852);
  const auto& ct867 = cc->EvalAdd(ct853, ct854);
  const auto& ct868 = cc->EvalAdd(ct866, ct867);
  const auto& ct869 = cc->EvalAdd(ct865, ct868);
  const auto& ct870 = cc->EvalAdd(ct855, ct856);
  const auto& ct871 = cc->EvalAdd(ct857, ct858);
  const auto& ct872 = cc->EvalAdd(ct870, ct871);
  const auto& ct873 = cc->EvalAdd(ct859, ct860);
  const auto& ct874 = cc->EvalAdd(ct861, ct862);
  const auto& ct875 = cc->EvalAdd(ct873, ct874);
  const auto& ct876 = cc->EvalAdd(ct872, ct875);
  const auto& ct877 = cc->EvalAdd(ct869, ct876);
  const auto& ct878 = cc->EvalRotate(ct877, 416);
  std::vector<float> v2580(std::begin(v28) + 432 * 512, std::begin(v28) + 432 * 512 + 1024);
  std::vector<float> v2581(592);
  std::copy(v2580.begin() + 0, v2580.begin() + 0 + 592, v2581.begin());
  std::vector<float> v2582(432);
  std::copy(v2580.begin() + 592, v2580.begin() + 592 + 432, v2582.begin());
  std::copy(v2581.begin(), v2581.end(), v86.begin() + 432);
  std::copy(v2582.begin(), v2582.end(), v86.begin() + 0);
  std::vector<double> v2585(std::begin(v86), std::end(v86));
  auto pt432_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt432_filled = v2585;
  pt432_filled.clear();
  pt432_filled.reserve(pt432_filled_n);
  for (auto i = 0; i < pt432_filled_n; ++i) {
    pt432_filled.push_back(v2585[i % v2585.size()]);
  }
  auto pt432 = cc->MakeCKKSPackedPlaintext(pt432_filled);
  const auto& ct879 = cc->EvalMult(ct, pt432);
  std::vector<float> v2586(std::begin(v28) + 433 * 512, std::begin(v28) + 433 * 512 + 1024);
  std::vector<float> v2587(592);
  std::copy(v2586.begin() + 0, v2586.begin() + 0 + 592, v2587.begin());
  std::vector<float> v2588(432);
  std::copy(v2586.begin() + 592, v2586.begin() + 592 + 432, v2588.begin());
  std::copy(v2587.begin(), v2587.end(), v86.begin() + 432);
  std::copy(v2588.begin(), v2588.end(), v86.begin() + 0);
  std::vector<double> v2591(std::begin(v86), std::end(v86));
  auto pt433_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt433_filled = v2591;
  pt433_filled.clear();
  pt433_filled.reserve(pt433_filled_n);
  for (auto i = 0; i < pt433_filled_n; ++i) {
    pt433_filled.push_back(v2591[i % v2591.size()]);
  }
  auto pt433 = cc->MakeCKKSPackedPlaintext(pt433_filled);
  const auto& ct880 = cc->EvalMult(ct2, pt433);
  std::vector<float> v2592(std::begin(v28) + 434 * 512, std::begin(v28) + 434 * 512 + 1024);
  std::vector<float> v2593(592);
  std::copy(v2592.begin() + 0, v2592.begin() + 0 + 592, v2593.begin());
  std::vector<float> v2594(432);
  std::copy(v2592.begin() + 592, v2592.begin() + 592 + 432, v2594.begin());
  std::copy(v2593.begin(), v2593.end(), v86.begin() + 432);
  std::copy(v2594.begin(), v2594.end(), v86.begin() + 0);
  std::vector<double> v2597(std::begin(v86), std::end(v86));
  auto pt434_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt434_filled = v2597;
  pt434_filled.clear();
  pt434_filled.reserve(pt434_filled_n);
  for (auto i = 0; i < pt434_filled_n; ++i) {
    pt434_filled.push_back(v2597[i % v2597.size()]);
  }
  auto pt434 = cc->MakeCKKSPackedPlaintext(pt434_filled);
  const auto& ct881 = cc->EvalMult(ct4, pt434);
  std::vector<float> v2598(std::begin(v28) + 435 * 512, std::begin(v28) + 435 * 512 + 1024);
  std::vector<float> v2599(592);
  std::copy(v2598.begin() + 0, v2598.begin() + 0 + 592, v2599.begin());
  std::vector<float> v2600(432);
  std::copy(v2598.begin() + 592, v2598.begin() + 592 + 432, v2600.begin());
  std::copy(v2599.begin(), v2599.end(), v86.begin() + 432);
  std::copy(v2600.begin(), v2600.end(), v86.begin() + 0);
  std::vector<double> v2603(std::begin(v86), std::end(v86));
  auto pt435_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt435_filled = v2603;
  pt435_filled.clear();
  pt435_filled.reserve(pt435_filled_n);
  for (auto i = 0; i < pt435_filled_n; ++i) {
    pt435_filled.push_back(v2603[i % v2603.size()]);
  }
  auto pt435 = cc->MakeCKKSPackedPlaintext(pt435_filled);
  const auto& ct882 = cc->EvalMult(ct6, pt435);
  std::vector<float> v2604(std::begin(v28) + 436 * 512, std::begin(v28) + 436 * 512 + 1024);
  std::vector<float> v2605(592);
  std::copy(v2604.begin() + 0, v2604.begin() + 0 + 592, v2605.begin());
  std::vector<float> v2606(432);
  std::copy(v2604.begin() + 592, v2604.begin() + 592 + 432, v2606.begin());
  std::copy(v2605.begin(), v2605.end(), v86.begin() + 432);
  std::copy(v2606.begin(), v2606.end(), v86.begin() + 0);
  std::vector<double> v2609(std::begin(v86), std::end(v86));
  auto pt436_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt436_filled = v2609;
  pt436_filled.clear();
  pt436_filled.reserve(pt436_filled_n);
  for (auto i = 0; i < pt436_filled_n; ++i) {
    pt436_filled.push_back(v2609[i % v2609.size()]);
  }
  auto pt436 = cc->MakeCKKSPackedPlaintext(pt436_filled);
  const auto& ct883 = cc->EvalMult(ct8, pt436);
  std::vector<float> v2610(std::begin(v28) + 437 * 512, std::begin(v28) + 437 * 512 + 1024);
  std::vector<float> v2611(592);
  std::copy(v2610.begin() + 0, v2610.begin() + 0 + 592, v2611.begin());
  std::vector<float> v2612(432);
  std::copy(v2610.begin() + 592, v2610.begin() + 592 + 432, v2612.begin());
  std::copy(v2611.begin(), v2611.end(), v86.begin() + 432);
  std::copy(v2612.begin(), v2612.end(), v86.begin() + 0);
  std::vector<double> v2615(std::begin(v86), std::end(v86));
  auto pt437_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt437_filled = v2615;
  pt437_filled.clear();
  pt437_filled.reserve(pt437_filled_n);
  for (auto i = 0; i < pt437_filled_n; ++i) {
    pt437_filled.push_back(v2615[i % v2615.size()]);
  }
  auto pt437 = cc->MakeCKKSPackedPlaintext(pt437_filled);
  const auto& ct884 = cc->EvalMult(ct10, pt437);
  std::vector<float> v2616(std::begin(v28) + 438 * 512, std::begin(v28) + 438 * 512 + 1024);
  std::vector<float> v2617(592);
  std::copy(v2616.begin() + 0, v2616.begin() + 0 + 592, v2617.begin());
  std::vector<float> v2618(432);
  std::copy(v2616.begin() + 592, v2616.begin() + 592 + 432, v2618.begin());
  std::copy(v2617.begin(), v2617.end(), v86.begin() + 432);
  std::copy(v2618.begin(), v2618.end(), v86.begin() + 0);
  std::vector<double> v2621(std::begin(v86), std::end(v86));
  auto pt438_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt438_filled = v2621;
  pt438_filled.clear();
  pt438_filled.reserve(pt438_filled_n);
  for (auto i = 0; i < pt438_filled_n; ++i) {
    pt438_filled.push_back(v2621[i % v2621.size()]);
  }
  auto pt438 = cc->MakeCKKSPackedPlaintext(pt438_filled);
  const auto& ct885 = cc->EvalMult(ct12, pt438);
  std::vector<float> v2622(std::begin(v28) + 439 * 512, std::begin(v28) + 439 * 512 + 1024);
  std::vector<float> v2623(592);
  std::copy(v2622.begin() + 0, v2622.begin() + 0 + 592, v2623.begin());
  std::vector<float> v2624(432);
  std::copy(v2622.begin() + 592, v2622.begin() + 592 + 432, v2624.begin());
  std::copy(v2623.begin(), v2623.end(), v86.begin() + 432);
  std::copy(v2624.begin(), v2624.end(), v86.begin() + 0);
  std::vector<double> v2627(std::begin(v86), std::end(v86));
  auto pt439_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt439_filled = v2627;
  pt439_filled.clear();
  pt439_filled.reserve(pt439_filled_n);
  for (auto i = 0; i < pt439_filled_n; ++i) {
    pt439_filled.push_back(v2627[i % v2627.size()]);
  }
  auto pt439 = cc->MakeCKKSPackedPlaintext(pt439_filled);
  const auto& ct886 = cc->EvalMult(ct14, pt439);
  std::vector<float> v2628(std::begin(v28) + 440 * 512, std::begin(v28) + 440 * 512 + 1024);
  std::vector<float> v2629(592);
  std::copy(v2628.begin() + 0, v2628.begin() + 0 + 592, v2629.begin());
  std::vector<float> v2630(432);
  std::copy(v2628.begin() + 592, v2628.begin() + 592 + 432, v2630.begin());
  std::copy(v2629.begin(), v2629.end(), v86.begin() + 432);
  std::copy(v2630.begin(), v2630.end(), v86.begin() + 0);
  std::vector<double> v2633(std::begin(v86), std::end(v86));
  auto pt440_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt440_filled = v2633;
  pt440_filled.clear();
  pt440_filled.reserve(pt440_filled_n);
  for (auto i = 0; i < pt440_filled_n; ++i) {
    pt440_filled.push_back(v2633[i % v2633.size()]);
  }
  auto pt440 = cc->MakeCKKSPackedPlaintext(pt440_filled);
  const auto& ct887 = cc->EvalMult(ct16, pt440);
  std::vector<float> v2634(std::begin(v28) + 441 * 512, std::begin(v28) + 441 * 512 + 1024);
  std::vector<float> v2635(592);
  std::copy(v2634.begin() + 0, v2634.begin() + 0 + 592, v2635.begin());
  std::vector<float> v2636(432);
  std::copy(v2634.begin() + 592, v2634.begin() + 592 + 432, v2636.begin());
  std::copy(v2635.begin(), v2635.end(), v86.begin() + 432);
  std::copy(v2636.begin(), v2636.end(), v86.begin() + 0);
  std::vector<double> v2639(std::begin(v86), std::end(v86));
  auto pt441_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt441_filled = v2639;
  pt441_filled.clear();
  pt441_filled.reserve(pt441_filled_n);
  for (auto i = 0; i < pt441_filled_n; ++i) {
    pt441_filled.push_back(v2639[i % v2639.size()]);
  }
  auto pt441 = cc->MakeCKKSPackedPlaintext(pt441_filled);
  const auto& ct888 = cc->EvalMult(ct18, pt441);
  std::vector<float> v2640(std::begin(v28) + 442 * 512, std::begin(v28) + 442 * 512 + 1024);
  std::vector<float> v2641(592);
  std::copy(v2640.begin() + 0, v2640.begin() + 0 + 592, v2641.begin());
  std::vector<float> v2642(432);
  std::copy(v2640.begin() + 592, v2640.begin() + 592 + 432, v2642.begin());
  std::copy(v2641.begin(), v2641.end(), v86.begin() + 432);
  std::copy(v2642.begin(), v2642.end(), v86.begin() + 0);
  std::vector<double> v2645(std::begin(v86), std::end(v86));
  auto pt442_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt442_filled = v2645;
  pt442_filled.clear();
  pt442_filled.reserve(pt442_filled_n);
  for (auto i = 0; i < pt442_filled_n; ++i) {
    pt442_filled.push_back(v2645[i % v2645.size()]);
  }
  auto pt442 = cc->MakeCKKSPackedPlaintext(pt442_filled);
  const auto& ct889 = cc->EvalMult(ct20, pt442);
  std::vector<float> v2646(std::begin(v28) + 443 * 512, std::begin(v28) + 443 * 512 + 1024);
  std::vector<float> v2647(592);
  std::copy(v2646.begin() + 0, v2646.begin() + 0 + 592, v2647.begin());
  std::vector<float> v2648(432);
  std::copy(v2646.begin() + 592, v2646.begin() + 592 + 432, v2648.begin());
  std::copy(v2647.begin(), v2647.end(), v86.begin() + 432);
  std::copy(v2648.begin(), v2648.end(), v86.begin() + 0);
  std::vector<double> v2651(std::begin(v86), std::end(v86));
  auto pt443_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt443_filled = v2651;
  pt443_filled.clear();
  pt443_filled.reserve(pt443_filled_n);
  for (auto i = 0; i < pt443_filled_n; ++i) {
    pt443_filled.push_back(v2651[i % v2651.size()]);
  }
  auto pt443 = cc->MakeCKKSPackedPlaintext(pt443_filled);
  const auto& ct890 = cc->EvalMult(ct22, pt443);
  std::vector<float> v2652(std::begin(v28) + 444 * 512, std::begin(v28) + 444 * 512 + 1024);
  std::vector<float> v2653(592);
  std::copy(v2652.begin() + 0, v2652.begin() + 0 + 592, v2653.begin());
  std::vector<float> v2654(432);
  std::copy(v2652.begin() + 592, v2652.begin() + 592 + 432, v2654.begin());
  std::copy(v2653.begin(), v2653.end(), v86.begin() + 432);
  std::copy(v2654.begin(), v2654.end(), v86.begin() + 0);
  std::vector<double> v2657(std::begin(v86), std::end(v86));
  auto pt444_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt444_filled = v2657;
  pt444_filled.clear();
  pt444_filled.reserve(pt444_filled_n);
  for (auto i = 0; i < pt444_filled_n; ++i) {
    pt444_filled.push_back(v2657[i % v2657.size()]);
  }
  auto pt444 = cc->MakeCKKSPackedPlaintext(pt444_filled);
  const auto& ct891 = cc->EvalMult(ct24, pt444);
  std::vector<float> v2658(std::begin(v28) + 445 * 512, std::begin(v28) + 445 * 512 + 1024);
  std::vector<float> v2659(592);
  std::copy(v2658.begin() + 0, v2658.begin() + 0 + 592, v2659.begin());
  std::vector<float> v2660(432);
  std::copy(v2658.begin() + 592, v2658.begin() + 592 + 432, v2660.begin());
  std::copy(v2659.begin(), v2659.end(), v86.begin() + 432);
  std::copy(v2660.begin(), v2660.end(), v86.begin() + 0);
  std::vector<double> v2663(std::begin(v86), std::end(v86));
  auto pt445_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt445_filled = v2663;
  pt445_filled.clear();
  pt445_filled.reserve(pt445_filled_n);
  for (auto i = 0; i < pt445_filled_n; ++i) {
    pt445_filled.push_back(v2663[i % v2663.size()]);
  }
  auto pt445 = cc->MakeCKKSPackedPlaintext(pt445_filled);
  const auto& ct892 = cc->EvalMult(ct26, pt445);
  std::vector<float> v2664(std::begin(v28) + 446 * 512, std::begin(v28) + 446 * 512 + 1024);
  std::vector<float> v2665(592);
  std::copy(v2664.begin() + 0, v2664.begin() + 0 + 592, v2665.begin());
  std::vector<float> v2666(432);
  std::copy(v2664.begin() + 592, v2664.begin() + 592 + 432, v2666.begin());
  std::copy(v2665.begin(), v2665.end(), v86.begin() + 432);
  std::copy(v2666.begin(), v2666.end(), v86.begin() + 0);
  std::vector<double> v2669(std::begin(v86), std::end(v86));
  auto pt446_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt446_filled = v2669;
  pt446_filled.clear();
  pt446_filled.reserve(pt446_filled_n);
  for (auto i = 0; i < pt446_filled_n; ++i) {
    pt446_filled.push_back(v2669[i % v2669.size()]);
  }
  auto pt446 = cc->MakeCKKSPackedPlaintext(pt446_filled);
  const auto& ct893 = cc->EvalMult(ct28, pt446);
  std::vector<float> v2670(std::begin(v28) + 447 * 512, std::begin(v28) + 447 * 512 + 1024);
  std::vector<float> v2671(592);
  std::copy(v2670.begin() + 0, v2670.begin() + 0 + 592, v2671.begin());
  std::vector<float> v2672(432);
  std::copy(v2670.begin() + 592, v2670.begin() + 592 + 432, v2672.begin());
  std::copy(v2671.begin(), v2671.end(), v86.begin() + 432);
  std::copy(v2672.begin(), v2672.end(), v86.begin() + 0);
  std::vector<double> v2675(std::begin(v86), std::end(v86));
  auto pt447_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt447_filled = v2675;
  pt447_filled.clear();
  pt447_filled.reserve(pt447_filled_n);
  for (auto i = 0; i < pt447_filled_n; ++i) {
    pt447_filled.push_back(v2675[i % v2675.size()]);
  }
  auto pt447 = cc->MakeCKKSPackedPlaintext(pt447_filled);
  const auto& ct894 = cc->EvalMult(ct30, pt447);
  const auto& ct895 = cc->EvalAdd(ct879, ct880);
  const auto& ct896 = cc->EvalAdd(ct881, ct882);
  const auto& ct897 = cc->EvalAdd(ct895, ct896);
  const auto& ct898 = cc->EvalAdd(ct883, ct884);
  const auto& ct899 = cc->EvalAdd(ct885, ct886);
  const auto& ct900 = cc->EvalAdd(ct898, ct899);
  const auto& ct901 = cc->EvalAdd(ct897, ct900);
  const auto& ct902 = cc->EvalAdd(ct887, ct888);
  const auto& ct903 = cc->EvalAdd(ct889, ct890);
  const auto& ct904 = cc->EvalAdd(ct902, ct903);
  const auto& ct905 = cc->EvalAdd(ct891, ct892);
  const auto& ct906 = cc->EvalAdd(ct893, ct894);
  const auto& ct907 = cc->EvalAdd(ct905, ct906);
  const auto& ct908 = cc->EvalAdd(ct904, ct907);
  const auto& ct909 = cc->EvalAdd(ct901, ct908);
  const auto& ct910 = cc->EvalRotate(ct909, 432);
  std::vector<float> v2676(std::begin(v28) + 448 * 512, std::begin(v28) + 448 * 512 + 1024);
  std::vector<float> v2677(576);
  std::copy(v2676.begin() + 0, v2676.begin() + 0 + 576, v2677.begin());
  std::vector<float> v2678(448);
  std::copy(v2676.begin() + 576, v2676.begin() + 576 + 448, v2678.begin());
  std::copy(v2677.begin(), v2677.end(), v86.begin() + 448);
  std::copy(v2678.begin(), v2678.end(), v86.begin() + 0);
  std::vector<double> v2681(std::begin(v86), std::end(v86));
  auto pt448_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt448_filled = v2681;
  pt448_filled.clear();
  pt448_filled.reserve(pt448_filled_n);
  for (auto i = 0; i < pt448_filled_n; ++i) {
    pt448_filled.push_back(v2681[i % v2681.size()]);
  }
  auto pt448 = cc->MakeCKKSPackedPlaintext(pt448_filled);
  const auto& ct911 = cc->EvalMult(ct, pt448);
  std::vector<float> v2682(std::begin(v28) + 449 * 512, std::begin(v28) + 449 * 512 + 1024);
  std::vector<float> v2683(576);
  std::copy(v2682.begin() + 0, v2682.begin() + 0 + 576, v2683.begin());
  std::vector<float> v2684(448);
  std::copy(v2682.begin() + 576, v2682.begin() + 576 + 448, v2684.begin());
  std::copy(v2683.begin(), v2683.end(), v86.begin() + 448);
  std::copy(v2684.begin(), v2684.end(), v86.begin() + 0);
  std::vector<double> v2687(std::begin(v86), std::end(v86));
  auto pt449_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt449_filled = v2687;
  pt449_filled.clear();
  pt449_filled.reserve(pt449_filled_n);
  for (auto i = 0; i < pt449_filled_n; ++i) {
    pt449_filled.push_back(v2687[i % v2687.size()]);
  }
  auto pt449 = cc->MakeCKKSPackedPlaintext(pt449_filled);
  const auto& ct912 = cc->EvalMult(ct2, pt449);
  std::vector<float> v2688(std::begin(v28) + 450 * 512, std::begin(v28) + 450 * 512 + 1024);
  std::vector<float> v2689(576);
  std::copy(v2688.begin() + 0, v2688.begin() + 0 + 576, v2689.begin());
  std::vector<float> v2690(448);
  std::copy(v2688.begin() + 576, v2688.begin() + 576 + 448, v2690.begin());
  std::copy(v2689.begin(), v2689.end(), v86.begin() + 448);
  std::copy(v2690.begin(), v2690.end(), v86.begin() + 0);
  std::vector<double> v2693(std::begin(v86), std::end(v86));
  auto pt450_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt450_filled = v2693;
  pt450_filled.clear();
  pt450_filled.reserve(pt450_filled_n);
  for (auto i = 0; i < pt450_filled_n; ++i) {
    pt450_filled.push_back(v2693[i % v2693.size()]);
  }
  auto pt450 = cc->MakeCKKSPackedPlaintext(pt450_filled);
  const auto& ct913 = cc->EvalMult(ct4, pt450);
  std::vector<float> v2694(std::begin(v28) + 451 * 512, std::begin(v28) + 451 * 512 + 1024);
  std::vector<float> v2695(576);
  std::copy(v2694.begin() + 0, v2694.begin() + 0 + 576, v2695.begin());
  std::vector<float> v2696(448);
  std::copy(v2694.begin() + 576, v2694.begin() + 576 + 448, v2696.begin());
  std::copy(v2695.begin(), v2695.end(), v86.begin() + 448);
  std::copy(v2696.begin(), v2696.end(), v86.begin() + 0);
  std::vector<double> v2699(std::begin(v86), std::end(v86));
  auto pt451_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt451_filled = v2699;
  pt451_filled.clear();
  pt451_filled.reserve(pt451_filled_n);
  for (auto i = 0; i < pt451_filled_n; ++i) {
    pt451_filled.push_back(v2699[i % v2699.size()]);
  }
  auto pt451 = cc->MakeCKKSPackedPlaintext(pt451_filled);
  const auto& ct914 = cc->EvalMult(ct6, pt451);
  std::vector<float> v2700(std::begin(v28) + 452 * 512, std::begin(v28) + 452 * 512 + 1024);
  std::vector<float> v2701(576);
  std::copy(v2700.begin() + 0, v2700.begin() + 0 + 576, v2701.begin());
  std::vector<float> v2702(448);
  std::copy(v2700.begin() + 576, v2700.begin() + 576 + 448, v2702.begin());
  std::copy(v2701.begin(), v2701.end(), v86.begin() + 448);
  std::copy(v2702.begin(), v2702.end(), v86.begin() + 0);
  std::vector<double> v2705(std::begin(v86), std::end(v86));
  auto pt452_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt452_filled = v2705;
  pt452_filled.clear();
  pt452_filled.reserve(pt452_filled_n);
  for (auto i = 0; i < pt452_filled_n; ++i) {
    pt452_filled.push_back(v2705[i % v2705.size()]);
  }
  auto pt452 = cc->MakeCKKSPackedPlaintext(pt452_filled);
  const auto& ct915 = cc->EvalMult(ct8, pt452);
  std::vector<float> v2706(std::begin(v28) + 453 * 512, std::begin(v28) + 453 * 512 + 1024);
  std::vector<float> v2707(576);
  std::copy(v2706.begin() + 0, v2706.begin() + 0 + 576, v2707.begin());
  std::vector<float> v2708(448);
  std::copy(v2706.begin() + 576, v2706.begin() + 576 + 448, v2708.begin());
  std::copy(v2707.begin(), v2707.end(), v86.begin() + 448);
  std::copy(v2708.begin(), v2708.end(), v86.begin() + 0);
  std::vector<double> v2711(std::begin(v86), std::end(v86));
  auto pt453_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt453_filled = v2711;
  pt453_filled.clear();
  pt453_filled.reserve(pt453_filled_n);
  for (auto i = 0; i < pt453_filled_n; ++i) {
    pt453_filled.push_back(v2711[i % v2711.size()]);
  }
  auto pt453 = cc->MakeCKKSPackedPlaintext(pt453_filled);
  const auto& ct916 = cc->EvalMult(ct10, pt453);
  std::vector<float> v2712(std::begin(v28) + 454 * 512, std::begin(v28) + 454 * 512 + 1024);
  std::vector<float> v2713(576);
  std::copy(v2712.begin() + 0, v2712.begin() + 0 + 576, v2713.begin());
  std::vector<float> v2714(448);
  std::copy(v2712.begin() + 576, v2712.begin() + 576 + 448, v2714.begin());
  std::copy(v2713.begin(), v2713.end(), v86.begin() + 448);
  std::copy(v2714.begin(), v2714.end(), v86.begin() + 0);
  std::vector<double> v2717(std::begin(v86), std::end(v86));
  auto pt454_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt454_filled = v2717;
  pt454_filled.clear();
  pt454_filled.reserve(pt454_filled_n);
  for (auto i = 0; i < pt454_filled_n; ++i) {
    pt454_filled.push_back(v2717[i % v2717.size()]);
  }
  auto pt454 = cc->MakeCKKSPackedPlaintext(pt454_filled);
  const auto& ct917 = cc->EvalMult(ct12, pt454);
  std::vector<float> v2718(std::begin(v28) + 455 * 512, std::begin(v28) + 455 * 512 + 1024);
  std::vector<float> v2719(576);
  std::copy(v2718.begin() + 0, v2718.begin() + 0 + 576, v2719.begin());
  std::vector<float> v2720(448);
  std::copy(v2718.begin() + 576, v2718.begin() + 576 + 448, v2720.begin());
  std::copy(v2719.begin(), v2719.end(), v86.begin() + 448);
  std::copy(v2720.begin(), v2720.end(), v86.begin() + 0);
  std::vector<double> v2723(std::begin(v86), std::end(v86));
  auto pt455_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt455_filled = v2723;
  pt455_filled.clear();
  pt455_filled.reserve(pt455_filled_n);
  for (auto i = 0; i < pt455_filled_n; ++i) {
    pt455_filled.push_back(v2723[i % v2723.size()]);
  }
  auto pt455 = cc->MakeCKKSPackedPlaintext(pt455_filled);
  const auto& ct918 = cc->EvalMult(ct14, pt455);
  std::vector<float> v2724(std::begin(v28) + 456 * 512, std::begin(v28) + 456 * 512 + 1024);
  std::vector<float> v2725(576);
  std::copy(v2724.begin() + 0, v2724.begin() + 0 + 576, v2725.begin());
  std::vector<float> v2726(448);
  std::copy(v2724.begin() + 576, v2724.begin() + 576 + 448, v2726.begin());
  std::copy(v2725.begin(), v2725.end(), v86.begin() + 448);
  std::copy(v2726.begin(), v2726.end(), v86.begin() + 0);
  std::vector<double> v2729(std::begin(v86), std::end(v86));
  auto pt456_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt456_filled = v2729;
  pt456_filled.clear();
  pt456_filled.reserve(pt456_filled_n);
  for (auto i = 0; i < pt456_filled_n; ++i) {
    pt456_filled.push_back(v2729[i % v2729.size()]);
  }
  auto pt456 = cc->MakeCKKSPackedPlaintext(pt456_filled);
  const auto& ct919 = cc->EvalMult(ct16, pt456);
  std::vector<float> v2730(std::begin(v28) + 457 * 512, std::begin(v28) + 457 * 512 + 1024);
  std::vector<float> v2731(576);
  std::copy(v2730.begin() + 0, v2730.begin() + 0 + 576, v2731.begin());
  std::vector<float> v2732(448);
  std::copy(v2730.begin() + 576, v2730.begin() + 576 + 448, v2732.begin());
  std::copy(v2731.begin(), v2731.end(), v86.begin() + 448);
  std::copy(v2732.begin(), v2732.end(), v86.begin() + 0);
  std::vector<double> v2735(std::begin(v86), std::end(v86));
  auto pt457_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt457_filled = v2735;
  pt457_filled.clear();
  pt457_filled.reserve(pt457_filled_n);
  for (auto i = 0; i < pt457_filled_n; ++i) {
    pt457_filled.push_back(v2735[i % v2735.size()]);
  }
  auto pt457 = cc->MakeCKKSPackedPlaintext(pt457_filled);
  const auto& ct920 = cc->EvalMult(ct18, pt457);
  std::vector<float> v2736(std::begin(v28) + 458 * 512, std::begin(v28) + 458 * 512 + 1024);
  std::vector<float> v2737(576);
  std::copy(v2736.begin() + 0, v2736.begin() + 0 + 576, v2737.begin());
  std::vector<float> v2738(448);
  std::copy(v2736.begin() + 576, v2736.begin() + 576 + 448, v2738.begin());
  std::copy(v2737.begin(), v2737.end(), v86.begin() + 448);
  std::copy(v2738.begin(), v2738.end(), v86.begin() + 0);
  std::vector<double> v2741(std::begin(v86), std::end(v86));
  auto pt458_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt458_filled = v2741;
  pt458_filled.clear();
  pt458_filled.reserve(pt458_filled_n);
  for (auto i = 0; i < pt458_filled_n; ++i) {
    pt458_filled.push_back(v2741[i % v2741.size()]);
  }
  auto pt458 = cc->MakeCKKSPackedPlaintext(pt458_filled);
  const auto& ct921 = cc->EvalMult(ct20, pt458);
  std::vector<float> v2742(std::begin(v28) + 459 * 512, std::begin(v28) + 459 * 512 + 1024);
  std::vector<float> v2743(576);
  std::copy(v2742.begin() + 0, v2742.begin() + 0 + 576, v2743.begin());
  std::vector<float> v2744(448);
  std::copy(v2742.begin() + 576, v2742.begin() + 576 + 448, v2744.begin());
  std::copy(v2743.begin(), v2743.end(), v86.begin() + 448);
  std::copy(v2744.begin(), v2744.end(), v86.begin() + 0);
  std::vector<double> v2747(std::begin(v86), std::end(v86));
  auto pt459_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt459_filled = v2747;
  pt459_filled.clear();
  pt459_filled.reserve(pt459_filled_n);
  for (auto i = 0; i < pt459_filled_n; ++i) {
    pt459_filled.push_back(v2747[i % v2747.size()]);
  }
  auto pt459 = cc->MakeCKKSPackedPlaintext(pt459_filled);
  const auto& ct922 = cc->EvalMult(ct22, pt459);
  std::vector<float> v2748(std::begin(v28) + 460 * 512, std::begin(v28) + 460 * 512 + 1024);
  std::vector<float> v2749(576);
  std::copy(v2748.begin() + 0, v2748.begin() + 0 + 576, v2749.begin());
  std::vector<float> v2750(448);
  std::copy(v2748.begin() + 576, v2748.begin() + 576 + 448, v2750.begin());
  std::copy(v2749.begin(), v2749.end(), v86.begin() + 448);
  std::copy(v2750.begin(), v2750.end(), v86.begin() + 0);
  std::vector<double> v2753(std::begin(v86), std::end(v86));
  auto pt460_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt460_filled = v2753;
  pt460_filled.clear();
  pt460_filled.reserve(pt460_filled_n);
  for (auto i = 0; i < pt460_filled_n; ++i) {
    pt460_filled.push_back(v2753[i % v2753.size()]);
  }
  auto pt460 = cc->MakeCKKSPackedPlaintext(pt460_filled);
  const auto& ct923 = cc->EvalMult(ct24, pt460);
  std::vector<float> v2754(std::begin(v28) + 461 * 512, std::begin(v28) + 461 * 512 + 1024);
  std::vector<float> v2755(576);
  std::copy(v2754.begin() + 0, v2754.begin() + 0 + 576, v2755.begin());
  std::vector<float> v2756(448);
  std::copy(v2754.begin() + 576, v2754.begin() + 576 + 448, v2756.begin());
  std::copy(v2755.begin(), v2755.end(), v86.begin() + 448);
  std::copy(v2756.begin(), v2756.end(), v86.begin() + 0);
  std::vector<double> v2759(std::begin(v86), std::end(v86));
  auto pt461_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt461_filled = v2759;
  pt461_filled.clear();
  pt461_filled.reserve(pt461_filled_n);
  for (auto i = 0; i < pt461_filled_n; ++i) {
    pt461_filled.push_back(v2759[i % v2759.size()]);
  }
  auto pt461 = cc->MakeCKKSPackedPlaintext(pt461_filled);
  const auto& ct924 = cc->EvalMult(ct26, pt461);
  std::vector<float> v2760(std::begin(v28) + 462 * 512, std::begin(v28) + 462 * 512 + 1024);
  std::vector<float> v2761(576);
  std::copy(v2760.begin() + 0, v2760.begin() + 0 + 576, v2761.begin());
  std::vector<float> v2762(448);
  std::copy(v2760.begin() + 576, v2760.begin() + 576 + 448, v2762.begin());
  std::copy(v2761.begin(), v2761.end(), v86.begin() + 448);
  std::copy(v2762.begin(), v2762.end(), v86.begin() + 0);
  std::vector<double> v2765(std::begin(v86), std::end(v86));
  auto pt462_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt462_filled = v2765;
  pt462_filled.clear();
  pt462_filled.reserve(pt462_filled_n);
  for (auto i = 0; i < pt462_filled_n; ++i) {
    pt462_filled.push_back(v2765[i % v2765.size()]);
  }
  auto pt462 = cc->MakeCKKSPackedPlaintext(pt462_filled);
  const auto& ct925 = cc->EvalMult(ct28, pt462);
  std::vector<float> v2766(std::begin(v28) + 463 * 512, std::begin(v28) + 463 * 512 + 1024);
  std::vector<float> v2767(576);
  std::copy(v2766.begin() + 0, v2766.begin() + 0 + 576, v2767.begin());
  std::vector<float> v2768(448);
  std::copy(v2766.begin() + 576, v2766.begin() + 576 + 448, v2768.begin());
  std::copy(v2767.begin(), v2767.end(), v86.begin() + 448);
  std::copy(v2768.begin(), v2768.end(), v86.begin() + 0);
  std::vector<double> v2771(std::begin(v86), std::end(v86));
  auto pt463_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt463_filled = v2771;
  pt463_filled.clear();
  pt463_filled.reserve(pt463_filled_n);
  for (auto i = 0; i < pt463_filled_n; ++i) {
    pt463_filled.push_back(v2771[i % v2771.size()]);
  }
  auto pt463 = cc->MakeCKKSPackedPlaintext(pt463_filled);
  const auto& ct926 = cc->EvalMult(ct30, pt463);
  const auto& ct927 = cc->EvalAdd(ct911, ct912);
  const auto& ct928 = cc->EvalAdd(ct913, ct914);
  const auto& ct929 = cc->EvalAdd(ct927, ct928);
  const auto& ct930 = cc->EvalAdd(ct915, ct916);
  const auto& ct931 = cc->EvalAdd(ct917, ct918);
  const auto& ct932 = cc->EvalAdd(ct930, ct931);
  const auto& ct933 = cc->EvalAdd(ct929, ct932);
  const auto& ct934 = cc->EvalAdd(ct919, ct920);
  const auto& ct935 = cc->EvalAdd(ct921, ct922);
  const auto& ct936 = cc->EvalAdd(ct934, ct935);
  const auto& ct937 = cc->EvalAdd(ct923, ct924);
  const auto& ct938 = cc->EvalAdd(ct925, ct926);
  const auto& ct939 = cc->EvalAdd(ct937, ct938);
  const auto& ct940 = cc->EvalAdd(ct936, ct939);
  const auto& ct941 = cc->EvalAdd(ct933, ct940);
  const auto& ct942 = cc->EvalRotate(ct941, 448);
  std::vector<float> v2772(std::begin(v28) + 464 * 512, std::begin(v28) + 464 * 512 + 1024);
  std::vector<float> v2773(560);
  std::copy(v2772.begin() + 0, v2772.begin() + 0 + 560, v2773.begin());
  std::vector<float> v2774(464);
  std::copy(v2772.begin() + 560, v2772.begin() + 560 + 464, v2774.begin());
  std::copy(v2773.begin(), v2773.end(), v86.begin() + 464);
  std::copy(v2774.begin(), v2774.end(), v86.begin() + 0);
  std::vector<double> v2777(std::begin(v86), std::end(v86));
  auto pt464_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt464_filled = v2777;
  pt464_filled.clear();
  pt464_filled.reserve(pt464_filled_n);
  for (auto i = 0; i < pt464_filled_n; ++i) {
    pt464_filled.push_back(v2777[i % v2777.size()]);
  }
  auto pt464 = cc->MakeCKKSPackedPlaintext(pt464_filled);
  const auto& ct943 = cc->EvalMult(ct, pt464);
  std::vector<float> v2778(std::begin(v28) + 465 * 512, std::begin(v28) + 465 * 512 + 1024);
  std::vector<float> v2779(560);
  std::copy(v2778.begin() + 0, v2778.begin() + 0 + 560, v2779.begin());
  std::vector<float> v2780(464);
  std::copy(v2778.begin() + 560, v2778.begin() + 560 + 464, v2780.begin());
  std::copy(v2779.begin(), v2779.end(), v86.begin() + 464);
  std::copy(v2780.begin(), v2780.end(), v86.begin() + 0);
  std::vector<double> v2783(std::begin(v86), std::end(v86));
  auto pt465_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt465_filled = v2783;
  pt465_filled.clear();
  pt465_filled.reserve(pt465_filled_n);
  for (auto i = 0; i < pt465_filled_n; ++i) {
    pt465_filled.push_back(v2783[i % v2783.size()]);
  }
  auto pt465 = cc->MakeCKKSPackedPlaintext(pt465_filled);
  const auto& ct944 = cc->EvalMult(ct2, pt465);
  std::vector<float> v2784(std::begin(v28) + 466 * 512, std::begin(v28) + 466 * 512 + 1024);
  std::vector<float> v2785(560);
  std::copy(v2784.begin() + 0, v2784.begin() + 0 + 560, v2785.begin());
  std::vector<float> v2786(464);
  std::copy(v2784.begin() + 560, v2784.begin() + 560 + 464, v2786.begin());
  std::copy(v2785.begin(), v2785.end(), v86.begin() + 464);
  std::copy(v2786.begin(), v2786.end(), v86.begin() + 0);
  std::vector<double> v2789(std::begin(v86), std::end(v86));
  auto pt466_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt466_filled = v2789;
  pt466_filled.clear();
  pt466_filled.reserve(pt466_filled_n);
  for (auto i = 0; i < pt466_filled_n; ++i) {
    pt466_filled.push_back(v2789[i % v2789.size()]);
  }
  auto pt466 = cc->MakeCKKSPackedPlaintext(pt466_filled);
  const auto& ct945 = cc->EvalMult(ct4, pt466);
  std::vector<float> v2790(std::begin(v28) + 467 * 512, std::begin(v28) + 467 * 512 + 1024);
  std::vector<float> v2791(560);
  std::copy(v2790.begin() + 0, v2790.begin() + 0 + 560, v2791.begin());
  std::vector<float> v2792(464);
  std::copy(v2790.begin() + 560, v2790.begin() + 560 + 464, v2792.begin());
  std::copy(v2791.begin(), v2791.end(), v86.begin() + 464);
  std::copy(v2792.begin(), v2792.end(), v86.begin() + 0);
  std::vector<double> v2795(std::begin(v86), std::end(v86));
  auto pt467_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt467_filled = v2795;
  pt467_filled.clear();
  pt467_filled.reserve(pt467_filled_n);
  for (auto i = 0; i < pt467_filled_n; ++i) {
    pt467_filled.push_back(v2795[i % v2795.size()]);
  }
  auto pt467 = cc->MakeCKKSPackedPlaintext(pt467_filled);
  const auto& ct946 = cc->EvalMult(ct6, pt467);
  std::vector<float> v2796(std::begin(v28) + 468 * 512, std::begin(v28) + 468 * 512 + 1024);
  std::vector<float> v2797(560);
  std::copy(v2796.begin() + 0, v2796.begin() + 0 + 560, v2797.begin());
  std::vector<float> v2798(464);
  std::copy(v2796.begin() + 560, v2796.begin() + 560 + 464, v2798.begin());
  std::copy(v2797.begin(), v2797.end(), v86.begin() + 464);
  std::copy(v2798.begin(), v2798.end(), v86.begin() + 0);
  std::vector<double> v2801(std::begin(v86), std::end(v86));
  auto pt468_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt468_filled = v2801;
  pt468_filled.clear();
  pt468_filled.reserve(pt468_filled_n);
  for (auto i = 0; i < pt468_filled_n; ++i) {
    pt468_filled.push_back(v2801[i % v2801.size()]);
  }
  auto pt468 = cc->MakeCKKSPackedPlaintext(pt468_filled);
  const auto& ct947 = cc->EvalMult(ct8, pt468);
  std::vector<float> v2802(std::begin(v28) + 469 * 512, std::begin(v28) + 469 * 512 + 1024);
  std::vector<float> v2803(560);
  std::copy(v2802.begin() + 0, v2802.begin() + 0 + 560, v2803.begin());
  std::vector<float> v2804(464);
  std::copy(v2802.begin() + 560, v2802.begin() + 560 + 464, v2804.begin());
  std::copy(v2803.begin(), v2803.end(), v86.begin() + 464);
  std::copy(v2804.begin(), v2804.end(), v86.begin() + 0);
  std::vector<double> v2807(std::begin(v86), std::end(v86));
  auto pt469_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt469_filled = v2807;
  pt469_filled.clear();
  pt469_filled.reserve(pt469_filled_n);
  for (auto i = 0; i < pt469_filled_n; ++i) {
    pt469_filled.push_back(v2807[i % v2807.size()]);
  }
  auto pt469 = cc->MakeCKKSPackedPlaintext(pt469_filled);
  const auto& ct948 = cc->EvalMult(ct10, pt469);
  std::vector<float> v2808(std::begin(v28) + 470 * 512, std::begin(v28) + 470 * 512 + 1024);
  std::vector<float> v2809(560);
  std::copy(v2808.begin() + 0, v2808.begin() + 0 + 560, v2809.begin());
  std::vector<float> v2810(464);
  std::copy(v2808.begin() + 560, v2808.begin() + 560 + 464, v2810.begin());
  std::copy(v2809.begin(), v2809.end(), v86.begin() + 464);
  std::copy(v2810.begin(), v2810.end(), v86.begin() + 0);
  std::vector<double> v2813(std::begin(v86), std::end(v86));
  auto pt470_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt470_filled = v2813;
  pt470_filled.clear();
  pt470_filled.reserve(pt470_filled_n);
  for (auto i = 0; i < pt470_filled_n; ++i) {
    pt470_filled.push_back(v2813[i % v2813.size()]);
  }
  auto pt470 = cc->MakeCKKSPackedPlaintext(pt470_filled);
  const auto& ct949 = cc->EvalMult(ct12, pt470);
  std::vector<float> v2814(std::begin(v28) + 471 * 512, std::begin(v28) + 471 * 512 + 1024);
  std::vector<float> v2815(560);
  std::copy(v2814.begin() + 0, v2814.begin() + 0 + 560, v2815.begin());
  std::vector<float> v2816(464);
  std::copy(v2814.begin() + 560, v2814.begin() + 560 + 464, v2816.begin());
  std::copy(v2815.begin(), v2815.end(), v86.begin() + 464);
  std::copy(v2816.begin(), v2816.end(), v86.begin() + 0);
  std::vector<double> v2819(std::begin(v86), std::end(v86));
  auto pt471_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt471_filled = v2819;
  pt471_filled.clear();
  pt471_filled.reserve(pt471_filled_n);
  for (auto i = 0; i < pt471_filled_n; ++i) {
    pt471_filled.push_back(v2819[i % v2819.size()]);
  }
  auto pt471 = cc->MakeCKKSPackedPlaintext(pt471_filled);
  const auto& ct950 = cc->EvalMult(ct14, pt471);
  std::vector<float> v2820(std::begin(v28) + 472 * 512, std::begin(v28) + 472 * 512 + 1024);
  std::vector<float> v2821(560);
  std::copy(v2820.begin() + 0, v2820.begin() + 0 + 560, v2821.begin());
  std::vector<float> v2822(464);
  std::copy(v2820.begin() + 560, v2820.begin() + 560 + 464, v2822.begin());
  std::copy(v2821.begin(), v2821.end(), v86.begin() + 464);
  std::copy(v2822.begin(), v2822.end(), v86.begin() + 0);
  std::vector<double> v2825(std::begin(v86), std::end(v86));
  auto pt472_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt472_filled = v2825;
  pt472_filled.clear();
  pt472_filled.reserve(pt472_filled_n);
  for (auto i = 0; i < pt472_filled_n; ++i) {
    pt472_filled.push_back(v2825[i % v2825.size()]);
  }
  auto pt472 = cc->MakeCKKSPackedPlaintext(pt472_filled);
  const auto& ct951 = cc->EvalMult(ct16, pt472);
  std::vector<float> v2826(std::begin(v28) + 473 * 512, std::begin(v28) + 473 * 512 + 1024);
  std::vector<float> v2827(560);
  std::copy(v2826.begin() + 0, v2826.begin() + 0 + 560, v2827.begin());
  std::vector<float> v2828(464);
  std::copy(v2826.begin() + 560, v2826.begin() + 560 + 464, v2828.begin());
  std::copy(v2827.begin(), v2827.end(), v86.begin() + 464);
  std::copy(v2828.begin(), v2828.end(), v86.begin() + 0);
  std::vector<double> v2831(std::begin(v86), std::end(v86));
  auto pt473_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt473_filled = v2831;
  pt473_filled.clear();
  pt473_filled.reserve(pt473_filled_n);
  for (auto i = 0; i < pt473_filled_n; ++i) {
    pt473_filled.push_back(v2831[i % v2831.size()]);
  }
  auto pt473 = cc->MakeCKKSPackedPlaintext(pt473_filled);
  const auto& ct952 = cc->EvalMult(ct18, pt473);
  std::vector<float> v2832(std::begin(v28) + 474 * 512, std::begin(v28) + 474 * 512 + 1024);
  std::vector<float> v2833(560);
  std::copy(v2832.begin() + 0, v2832.begin() + 0 + 560, v2833.begin());
  std::vector<float> v2834(464);
  std::copy(v2832.begin() + 560, v2832.begin() + 560 + 464, v2834.begin());
  std::copy(v2833.begin(), v2833.end(), v86.begin() + 464);
  std::copy(v2834.begin(), v2834.end(), v86.begin() + 0);
  std::vector<double> v2837(std::begin(v86), std::end(v86));
  auto pt474_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt474_filled = v2837;
  pt474_filled.clear();
  pt474_filled.reserve(pt474_filled_n);
  for (auto i = 0; i < pt474_filled_n; ++i) {
    pt474_filled.push_back(v2837[i % v2837.size()]);
  }
  auto pt474 = cc->MakeCKKSPackedPlaintext(pt474_filled);
  const auto& ct953 = cc->EvalMult(ct20, pt474);
  std::vector<float> v2838(std::begin(v28) + 475 * 512, std::begin(v28) + 475 * 512 + 1024);
  std::vector<float> v2839(560);
  std::copy(v2838.begin() + 0, v2838.begin() + 0 + 560, v2839.begin());
  std::vector<float> v2840(464);
  std::copy(v2838.begin() + 560, v2838.begin() + 560 + 464, v2840.begin());
  std::copy(v2839.begin(), v2839.end(), v86.begin() + 464);
  std::copy(v2840.begin(), v2840.end(), v86.begin() + 0);
  std::vector<double> v2843(std::begin(v86), std::end(v86));
  auto pt475_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt475_filled = v2843;
  pt475_filled.clear();
  pt475_filled.reserve(pt475_filled_n);
  for (auto i = 0; i < pt475_filled_n; ++i) {
    pt475_filled.push_back(v2843[i % v2843.size()]);
  }
  auto pt475 = cc->MakeCKKSPackedPlaintext(pt475_filled);
  const auto& ct954 = cc->EvalMult(ct22, pt475);
  std::vector<float> v2844(std::begin(v28) + 476 * 512, std::begin(v28) + 476 * 512 + 1024);
  std::vector<float> v2845(560);
  std::copy(v2844.begin() + 0, v2844.begin() + 0 + 560, v2845.begin());
  std::vector<float> v2846(464);
  std::copy(v2844.begin() + 560, v2844.begin() + 560 + 464, v2846.begin());
  std::copy(v2845.begin(), v2845.end(), v86.begin() + 464);
  std::copy(v2846.begin(), v2846.end(), v86.begin() + 0);
  std::vector<double> v2849(std::begin(v86), std::end(v86));
  auto pt476_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt476_filled = v2849;
  pt476_filled.clear();
  pt476_filled.reserve(pt476_filled_n);
  for (auto i = 0; i < pt476_filled_n; ++i) {
    pt476_filled.push_back(v2849[i % v2849.size()]);
  }
  auto pt476 = cc->MakeCKKSPackedPlaintext(pt476_filled);
  const auto& ct955 = cc->EvalMult(ct24, pt476);
  std::vector<float> v2850(std::begin(v28) + 477 * 512, std::begin(v28) + 477 * 512 + 1024);
  std::vector<float> v2851(560);
  std::copy(v2850.begin() + 0, v2850.begin() + 0 + 560, v2851.begin());
  std::vector<float> v2852(464);
  std::copy(v2850.begin() + 560, v2850.begin() + 560 + 464, v2852.begin());
  std::copy(v2851.begin(), v2851.end(), v86.begin() + 464);
  std::copy(v2852.begin(), v2852.end(), v86.begin() + 0);
  std::vector<double> v2855(std::begin(v86), std::end(v86));
  auto pt477_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt477_filled = v2855;
  pt477_filled.clear();
  pt477_filled.reserve(pt477_filled_n);
  for (auto i = 0; i < pt477_filled_n; ++i) {
    pt477_filled.push_back(v2855[i % v2855.size()]);
  }
  auto pt477 = cc->MakeCKKSPackedPlaintext(pt477_filled);
  const auto& ct956 = cc->EvalMult(ct26, pt477);
  std::vector<float> v2856(std::begin(v28) + 478 * 512, std::begin(v28) + 478 * 512 + 1024);
  std::vector<float> v2857(560);
  std::copy(v2856.begin() + 0, v2856.begin() + 0 + 560, v2857.begin());
  std::vector<float> v2858(464);
  std::copy(v2856.begin() + 560, v2856.begin() + 560 + 464, v2858.begin());
  std::copy(v2857.begin(), v2857.end(), v86.begin() + 464);
  std::copy(v2858.begin(), v2858.end(), v86.begin() + 0);
  std::vector<double> v2861(std::begin(v86), std::end(v86));
  auto pt478_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt478_filled = v2861;
  pt478_filled.clear();
  pt478_filled.reserve(pt478_filled_n);
  for (auto i = 0; i < pt478_filled_n; ++i) {
    pt478_filled.push_back(v2861[i % v2861.size()]);
  }
  auto pt478 = cc->MakeCKKSPackedPlaintext(pt478_filled);
  const auto& ct957 = cc->EvalMult(ct28, pt478);
  std::vector<float> v2862(std::begin(v28) + 479 * 512, std::begin(v28) + 479 * 512 + 1024);
  std::vector<float> v2863(560);
  std::copy(v2862.begin() + 0, v2862.begin() + 0 + 560, v2863.begin());
  std::vector<float> v2864(464);
  std::copy(v2862.begin() + 560, v2862.begin() + 560 + 464, v2864.begin());
  std::copy(v2863.begin(), v2863.end(), v86.begin() + 464);
  std::copy(v2864.begin(), v2864.end(), v86.begin() + 0);
  std::vector<double> v2867(std::begin(v86), std::end(v86));
  auto pt479_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt479_filled = v2867;
  pt479_filled.clear();
  pt479_filled.reserve(pt479_filled_n);
  for (auto i = 0; i < pt479_filled_n; ++i) {
    pt479_filled.push_back(v2867[i % v2867.size()]);
  }
  auto pt479 = cc->MakeCKKSPackedPlaintext(pt479_filled);
  const auto& ct958 = cc->EvalMult(ct30, pt479);
  const auto& ct959 = cc->EvalAdd(ct943, ct944);
  const auto& ct960 = cc->EvalAdd(ct945, ct946);
  const auto& ct961 = cc->EvalAdd(ct959, ct960);
  const auto& ct962 = cc->EvalAdd(ct947, ct948);
  const auto& ct963 = cc->EvalAdd(ct949, ct950);
  const auto& ct964 = cc->EvalAdd(ct962, ct963);
  const auto& ct965 = cc->EvalAdd(ct961, ct964);
  const auto& ct966 = cc->EvalAdd(ct951, ct952);
  const auto& ct967 = cc->EvalAdd(ct953, ct954);
  const auto& ct968 = cc->EvalAdd(ct966, ct967);
  const auto& ct969 = cc->EvalAdd(ct955, ct956);
  const auto& ct970 = cc->EvalAdd(ct957, ct958);
  const auto& ct971 = cc->EvalAdd(ct969, ct970);
  const auto& ct972 = cc->EvalAdd(ct968, ct971);
  const auto& ct973 = cc->EvalAdd(ct965, ct972);
  const auto& ct974 = cc->EvalRotate(ct973, 464);
  std::vector<float> v2868(std::begin(v28) + 480 * 512, std::begin(v28) + 480 * 512 + 1024);
  std::vector<float> v2869(544);
  std::copy(v2868.begin() + 0, v2868.begin() + 0 + 544, v2869.begin());
  std::vector<float> v2870(480);
  std::copy(v2868.begin() + 544, v2868.begin() + 544 + 480, v2870.begin());
  std::copy(v2869.begin(), v2869.end(), v86.begin() + 480);
  std::copy(v2870.begin(), v2870.end(), v86.begin() + 0);
  std::vector<double> v2873(std::begin(v86), std::end(v86));
  auto pt480_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt480_filled = v2873;
  pt480_filled.clear();
  pt480_filled.reserve(pt480_filled_n);
  for (auto i = 0; i < pt480_filled_n; ++i) {
    pt480_filled.push_back(v2873[i % v2873.size()]);
  }
  auto pt480 = cc->MakeCKKSPackedPlaintext(pt480_filled);
  const auto& ct975 = cc->EvalMult(ct, pt480);
  std::vector<float> v2874(std::begin(v28) + 481 * 512, std::begin(v28) + 481 * 512 + 1024);
  std::vector<float> v2875(544);
  std::copy(v2874.begin() + 0, v2874.begin() + 0 + 544, v2875.begin());
  std::vector<float> v2876(480);
  std::copy(v2874.begin() + 544, v2874.begin() + 544 + 480, v2876.begin());
  std::copy(v2875.begin(), v2875.end(), v86.begin() + 480);
  std::copy(v2876.begin(), v2876.end(), v86.begin() + 0);
  std::vector<double> v2879(std::begin(v86), std::end(v86));
  auto pt481_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt481_filled = v2879;
  pt481_filled.clear();
  pt481_filled.reserve(pt481_filled_n);
  for (auto i = 0; i < pt481_filled_n; ++i) {
    pt481_filled.push_back(v2879[i % v2879.size()]);
  }
  auto pt481 = cc->MakeCKKSPackedPlaintext(pt481_filled);
  const auto& ct976 = cc->EvalMult(ct2, pt481);
  std::vector<float> v2880(std::begin(v28) + 482 * 512, std::begin(v28) + 482 * 512 + 1024);
  std::vector<float> v2881(544);
  std::copy(v2880.begin() + 0, v2880.begin() + 0 + 544, v2881.begin());
  std::vector<float> v2882(480);
  std::copy(v2880.begin() + 544, v2880.begin() + 544 + 480, v2882.begin());
  std::copy(v2881.begin(), v2881.end(), v86.begin() + 480);
  std::copy(v2882.begin(), v2882.end(), v86.begin() + 0);
  std::vector<double> v2885(std::begin(v86), std::end(v86));
  auto pt482_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt482_filled = v2885;
  pt482_filled.clear();
  pt482_filled.reserve(pt482_filled_n);
  for (auto i = 0; i < pt482_filled_n; ++i) {
    pt482_filled.push_back(v2885[i % v2885.size()]);
  }
  auto pt482 = cc->MakeCKKSPackedPlaintext(pt482_filled);
  const auto& ct977 = cc->EvalMult(ct4, pt482);
  std::vector<float> v2886(std::begin(v28) + 483 * 512, std::begin(v28) + 483 * 512 + 1024);
  std::vector<float> v2887(544);
  std::copy(v2886.begin() + 0, v2886.begin() + 0 + 544, v2887.begin());
  std::vector<float> v2888(480);
  std::copy(v2886.begin() + 544, v2886.begin() + 544 + 480, v2888.begin());
  std::copy(v2887.begin(), v2887.end(), v86.begin() + 480);
  std::copy(v2888.begin(), v2888.end(), v86.begin() + 0);
  std::vector<double> v2891(std::begin(v86), std::end(v86));
  auto pt483_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt483_filled = v2891;
  pt483_filled.clear();
  pt483_filled.reserve(pt483_filled_n);
  for (auto i = 0; i < pt483_filled_n; ++i) {
    pt483_filled.push_back(v2891[i % v2891.size()]);
  }
  auto pt483 = cc->MakeCKKSPackedPlaintext(pt483_filled);
  const auto& ct978 = cc->EvalMult(ct6, pt483);
  std::vector<float> v2892(std::begin(v28) + 484 * 512, std::begin(v28) + 484 * 512 + 1024);
  std::vector<float> v2893(544);
  std::copy(v2892.begin() + 0, v2892.begin() + 0 + 544, v2893.begin());
  std::vector<float> v2894(480);
  std::copy(v2892.begin() + 544, v2892.begin() + 544 + 480, v2894.begin());
  std::copy(v2893.begin(), v2893.end(), v86.begin() + 480);
  std::copy(v2894.begin(), v2894.end(), v86.begin() + 0);
  std::vector<double> v2897(std::begin(v86), std::end(v86));
  auto pt484_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt484_filled = v2897;
  pt484_filled.clear();
  pt484_filled.reserve(pt484_filled_n);
  for (auto i = 0; i < pt484_filled_n; ++i) {
    pt484_filled.push_back(v2897[i % v2897.size()]);
  }
  auto pt484 = cc->MakeCKKSPackedPlaintext(pt484_filled);
  const auto& ct979 = cc->EvalMult(ct8, pt484);
  std::vector<float> v2898(std::begin(v28) + 485 * 512, std::begin(v28) + 485 * 512 + 1024);
  std::vector<float> v2899(544);
  std::copy(v2898.begin() + 0, v2898.begin() + 0 + 544, v2899.begin());
  std::vector<float> v2900(480);
  std::copy(v2898.begin() + 544, v2898.begin() + 544 + 480, v2900.begin());
  std::copy(v2899.begin(), v2899.end(), v86.begin() + 480);
  std::copy(v2900.begin(), v2900.end(), v86.begin() + 0);
  std::vector<double> v2903(std::begin(v86), std::end(v86));
  auto pt485_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt485_filled = v2903;
  pt485_filled.clear();
  pt485_filled.reserve(pt485_filled_n);
  for (auto i = 0; i < pt485_filled_n; ++i) {
    pt485_filled.push_back(v2903[i % v2903.size()]);
  }
  auto pt485 = cc->MakeCKKSPackedPlaintext(pt485_filled);
  const auto& ct980 = cc->EvalMult(ct10, pt485);
  std::vector<float> v2904(std::begin(v28) + 486 * 512, std::begin(v28) + 486 * 512 + 1024);
  std::vector<float> v2905(544);
  std::copy(v2904.begin() + 0, v2904.begin() + 0 + 544, v2905.begin());
  std::vector<float> v2906(480);
  std::copy(v2904.begin() + 544, v2904.begin() + 544 + 480, v2906.begin());
  std::copy(v2905.begin(), v2905.end(), v86.begin() + 480);
  std::copy(v2906.begin(), v2906.end(), v86.begin() + 0);
  std::vector<double> v2909(std::begin(v86), std::end(v86));
  auto pt486_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt486_filled = v2909;
  pt486_filled.clear();
  pt486_filled.reserve(pt486_filled_n);
  for (auto i = 0; i < pt486_filled_n; ++i) {
    pt486_filled.push_back(v2909[i % v2909.size()]);
  }
  auto pt486 = cc->MakeCKKSPackedPlaintext(pt486_filled);
  const auto& ct981 = cc->EvalMult(ct12, pt486);
  std::vector<float> v2910(std::begin(v28) + 487 * 512, std::begin(v28) + 487 * 512 + 1024);
  std::vector<float> v2911(544);
  std::copy(v2910.begin() + 0, v2910.begin() + 0 + 544, v2911.begin());
  std::vector<float> v2912(480);
  std::copy(v2910.begin() + 544, v2910.begin() + 544 + 480, v2912.begin());
  std::copy(v2911.begin(), v2911.end(), v86.begin() + 480);
  std::copy(v2912.begin(), v2912.end(), v86.begin() + 0);
  std::vector<double> v2915(std::begin(v86), std::end(v86));
  auto pt487_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt487_filled = v2915;
  pt487_filled.clear();
  pt487_filled.reserve(pt487_filled_n);
  for (auto i = 0; i < pt487_filled_n; ++i) {
    pt487_filled.push_back(v2915[i % v2915.size()]);
  }
  auto pt487 = cc->MakeCKKSPackedPlaintext(pt487_filled);
  const auto& ct982 = cc->EvalMult(ct14, pt487);
  std::vector<float> v2916(std::begin(v28) + 488 * 512, std::begin(v28) + 488 * 512 + 1024);
  std::vector<float> v2917(544);
  std::copy(v2916.begin() + 0, v2916.begin() + 0 + 544, v2917.begin());
  std::vector<float> v2918(480);
  std::copy(v2916.begin() + 544, v2916.begin() + 544 + 480, v2918.begin());
  std::copy(v2917.begin(), v2917.end(), v86.begin() + 480);
  std::copy(v2918.begin(), v2918.end(), v86.begin() + 0);
  std::vector<double> v2921(std::begin(v86), std::end(v86));
  auto pt488_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt488_filled = v2921;
  pt488_filled.clear();
  pt488_filled.reserve(pt488_filled_n);
  for (auto i = 0; i < pt488_filled_n; ++i) {
    pt488_filled.push_back(v2921[i % v2921.size()]);
  }
  auto pt488 = cc->MakeCKKSPackedPlaintext(pt488_filled);
  const auto& ct983 = cc->EvalMult(ct16, pt488);
  std::vector<float> v2922(std::begin(v28) + 489 * 512, std::begin(v28) + 489 * 512 + 1024);
  std::vector<float> v2923(544);
  std::copy(v2922.begin() + 0, v2922.begin() + 0 + 544, v2923.begin());
  std::vector<float> v2924(480);
  std::copy(v2922.begin() + 544, v2922.begin() + 544 + 480, v2924.begin());
  std::copy(v2923.begin(), v2923.end(), v86.begin() + 480);
  std::copy(v2924.begin(), v2924.end(), v86.begin() + 0);
  std::vector<double> v2927(std::begin(v86), std::end(v86));
  auto pt489_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt489_filled = v2927;
  pt489_filled.clear();
  pt489_filled.reserve(pt489_filled_n);
  for (auto i = 0; i < pt489_filled_n; ++i) {
    pt489_filled.push_back(v2927[i % v2927.size()]);
  }
  auto pt489 = cc->MakeCKKSPackedPlaintext(pt489_filled);
  const auto& ct984 = cc->EvalMult(ct18, pt489);
  std::vector<float> v2928(std::begin(v28) + 490 * 512, std::begin(v28) + 490 * 512 + 1024);
  std::vector<float> v2929(544);
  std::copy(v2928.begin() + 0, v2928.begin() + 0 + 544, v2929.begin());
  std::vector<float> v2930(480);
  std::copy(v2928.begin() + 544, v2928.begin() + 544 + 480, v2930.begin());
  std::copy(v2929.begin(), v2929.end(), v86.begin() + 480);
  std::copy(v2930.begin(), v2930.end(), v86.begin() + 0);
  std::vector<double> v2933(std::begin(v86), std::end(v86));
  auto pt490_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt490_filled = v2933;
  pt490_filled.clear();
  pt490_filled.reserve(pt490_filled_n);
  for (auto i = 0; i < pt490_filled_n; ++i) {
    pt490_filled.push_back(v2933[i % v2933.size()]);
  }
  auto pt490 = cc->MakeCKKSPackedPlaintext(pt490_filled);
  const auto& ct985 = cc->EvalMult(ct20, pt490);
  std::vector<float> v2934(std::begin(v28) + 491 * 512, std::begin(v28) + 491 * 512 + 1024);
  std::vector<float> v2935(544);
  std::copy(v2934.begin() + 0, v2934.begin() + 0 + 544, v2935.begin());
  std::vector<float> v2936(480);
  std::copy(v2934.begin() + 544, v2934.begin() + 544 + 480, v2936.begin());
  std::copy(v2935.begin(), v2935.end(), v86.begin() + 480);
  std::copy(v2936.begin(), v2936.end(), v86.begin() + 0);
  std::vector<double> v2939(std::begin(v86), std::end(v86));
  auto pt491_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt491_filled = v2939;
  pt491_filled.clear();
  pt491_filled.reserve(pt491_filled_n);
  for (auto i = 0; i < pt491_filled_n; ++i) {
    pt491_filled.push_back(v2939[i % v2939.size()]);
  }
  auto pt491 = cc->MakeCKKSPackedPlaintext(pt491_filled);
  const auto& ct986 = cc->EvalMult(ct22, pt491);
  std::vector<float> v2940(std::begin(v28) + 492 * 512, std::begin(v28) + 492 * 512 + 1024);
  std::vector<float> v2941(544);
  std::copy(v2940.begin() + 0, v2940.begin() + 0 + 544, v2941.begin());
  std::vector<float> v2942(480);
  std::copy(v2940.begin() + 544, v2940.begin() + 544 + 480, v2942.begin());
  std::copy(v2941.begin(), v2941.end(), v86.begin() + 480);
  std::copy(v2942.begin(), v2942.end(), v86.begin() + 0);
  std::vector<double> v2945(std::begin(v86), std::end(v86));
  auto pt492_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt492_filled = v2945;
  pt492_filled.clear();
  pt492_filled.reserve(pt492_filled_n);
  for (auto i = 0; i < pt492_filled_n; ++i) {
    pt492_filled.push_back(v2945[i % v2945.size()]);
  }
  auto pt492 = cc->MakeCKKSPackedPlaintext(pt492_filled);
  const auto& ct987 = cc->EvalMult(ct24, pt492);
  std::vector<float> v2946(std::begin(v28) + 493 * 512, std::begin(v28) + 493 * 512 + 1024);
  std::vector<float> v2947(544);
  std::copy(v2946.begin() + 0, v2946.begin() + 0 + 544, v2947.begin());
  std::vector<float> v2948(480);
  std::copy(v2946.begin() + 544, v2946.begin() + 544 + 480, v2948.begin());
  std::copy(v2947.begin(), v2947.end(), v86.begin() + 480);
  std::copy(v2948.begin(), v2948.end(), v86.begin() + 0);
  std::vector<double> v2951(std::begin(v86), std::end(v86));
  auto pt493_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt493_filled = v2951;
  pt493_filled.clear();
  pt493_filled.reserve(pt493_filled_n);
  for (auto i = 0; i < pt493_filled_n; ++i) {
    pt493_filled.push_back(v2951[i % v2951.size()]);
  }
  auto pt493 = cc->MakeCKKSPackedPlaintext(pt493_filled);
  const auto& ct988 = cc->EvalMult(ct26, pt493);
  std::vector<float> v2952(std::begin(v28) + 494 * 512, std::begin(v28) + 494 * 512 + 1024);
  std::vector<float> v2953(544);
  std::copy(v2952.begin() + 0, v2952.begin() + 0 + 544, v2953.begin());
  std::vector<float> v2954(480);
  std::copy(v2952.begin() + 544, v2952.begin() + 544 + 480, v2954.begin());
  std::copy(v2953.begin(), v2953.end(), v86.begin() + 480);
  std::copy(v2954.begin(), v2954.end(), v86.begin() + 0);
  std::vector<double> v2957(std::begin(v86), std::end(v86));
  auto pt494_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt494_filled = v2957;
  pt494_filled.clear();
  pt494_filled.reserve(pt494_filled_n);
  for (auto i = 0; i < pt494_filled_n; ++i) {
    pt494_filled.push_back(v2957[i % v2957.size()]);
  }
  auto pt494 = cc->MakeCKKSPackedPlaintext(pt494_filled);
  const auto& ct989 = cc->EvalMult(ct28, pt494);
  std::vector<float> v2958(std::begin(v28) + 495 * 512, std::begin(v28) + 495 * 512 + 1024);
  std::vector<float> v2959(544);
  std::copy(v2958.begin() + 0, v2958.begin() + 0 + 544, v2959.begin());
  std::vector<float> v2960(480);
  std::copy(v2958.begin() + 544, v2958.begin() + 544 + 480, v2960.begin());
  std::copy(v2959.begin(), v2959.end(), v86.begin() + 480);
  std::copy(v2960.begin(), v2960.end(), v86.begin() + 0);
  std::vector<double> v2963(std::begin(v86), std::end(v86));
  auto pt495_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt495_filled = v2963;
  pt495_filled.clear();
  pt495_filled.reserve(pt495_filled_n);
  for (auto i = 0; i < pt495_filled_n; ++i) {
    pt495_filled.push_back(v2963[i % v2963.size()]);
  }
  auto pt495 = cc->MakeCKKSPackedPlaintext(pt495_filled);
  const auto& ct990 = cc->EvalMult(ct30, pt495);
  const auto& ct991 = cc->EvalAdd(ct975, ct976);
  const auto& ct992 = cc->EvalAdd(ct977, ct978);
  const auto& ct993 = cc->EvalAdd(ct991, ct992);
  const auto& ct994 = cc->EvalAdd(ct979, ct980);
  const auto& ct995 = cc->EvalAdd(ct981, ct982);
  const auto& ct996 = cc->EvalAdd(ct994, ct995);
  const auto& ct997 = cc->EvalAdd(ct993, ct996);
  const auto& ct998 = cc->EvalAdd(ct983, ct984);
  const auto& ct999 = cc->EvalAdd(ct985, ct986);
  const auto& ct1000 = cc->EvalAdd(ct998, ct999);
  const auto& ct1001 = cc->EvalAdd(ct987, ct988);
  const auto& ct1002 = cc->EvalAdd(ct989, ct990);
  const auto& ct1003 = cc->EvalAdd(ct1001, ct1002);
  const auto& ct1004 = cc->EvalAdd(ct1000, ct1003);
  const auto& ct1005 = cc->EvalAdd(ct997, ct1004);
  const auto& ct1006 = cc->EvalRotate(ct1005, 480);
  std::vector<float> v2964(std::begin(v28) + 496 * 512, std::begin(v28) + 496 * 512 + 1024);
  std::vector<float> v2965(528);
  std::copy(v2964.begin() + 0, v2964.begin() + 0 + 528, v2965.begin());
  std::vector<float> v2966(496);
  std::copy(v2964.begin() + 528, v2964.begin() + 528 + 496, v2966.begin());
  std::copy(v2965.begin(), v2965.end(), v86.begin() + 496);
  std::copy(v2966.begin(), v2966.end(), v86.begin() + 0);
  std::vector<double> v2969(std::begin(v86), std::end(v86));
  auto pt496_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt496_filled = v2969;
  pt496_filled.clear();
  pt496_filled.reserve(pt496_filled_n);
  for (auto i = 0; i < pt496_filled_n; ++i) {
    pt496_filled.push_back(v2969[i % v2969.size()]);
  }
  auto pt496 = cc->MakeCKKSPackedPlaintext(pt496_filled);
  const auto& ct1007 = cc->EvalMult(ct, pt496);
  std::vector<float> v2970(std::begin(v28) + 497 * 512, std::begin(v28) + 497 * 512 + 1024);
  std::vector<float> v2971(528);
  std::copy(v2970.begin() + 0, v2970.begin() + 0 + 528, v2971.begin());
  std::vector<float> v2972(496);
  std::copy(v2970.begin() + 528, v2970.begin() + 528 + 496, v2972.begin());
  std::copy(v2971.begin(), v2971.end(), v86.begin() + 496);
  std::copy(v2972.begin(), v2972.end(), v86.begin() + 0);
  std::vector<double> v2975(std::begin(v86), std::end(v86));
  auto pt497_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt497_filled = v2975;
  pt497_filled.clear();
  pt497_filled.reserve(pt497_filled_n);
  for (auto i = 0; i < pt497_filled_n; ++i) {
    pt497_filled.push_back(v2975[i % v2975.size()]);
  }
  auto pt497 = cc->MakeCKKSPackedPlaintext(pt497_filled);
  const auto& ct1008 = cc->EvalMult(ct2, pt497);
  std::vector<float> v2976(std::begin(v28) + 498 * 512, std::begin(v28) + 498 * 512 + 1024);
  std::vector<float> v2977(528);
  std::copy(v2976.begin() + 0, v2976.begin() + 0 + 528, v2977.begin());
  std::vector<float> v2978(496);
  std::copy(v2976.begin() + 528, v2976.begin() + 528 + 496, v2978.begin());
  std::copy(v2977.begin(), v2977.end(), v86.begin() + 496);
  std::copy(v2978.begin(), v2978.end(), v86.begin() + 0);
  std::vector<double> v2981(std::begin(v86), std::end(v86));
  auto pt498_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt498_filled = v2981;
  pt498_filled.clear();
  pt498_filled.reserve(pt498_filled_n);
  for (auto i = 0; i < pt498_filled_n; ++i) {
    pt498_filled.push_back(v2981[i % v2981.size()]);
  }
  auto pt498 = cc->MakeCKKSPackedPlaintext(pt498_filled);
  const auto& ct1009 = cc->EvalMult(ct4, pt498);
  std::vector<float> v2982(std::begin(v28) + 499 * 512, std::begin(v28) + 499 * 512 + 1024);
  std::vector<float> v2983(528);
  std::copy(v2982.begin() + 0, v2982.begin() + 0 + 528, v2983.begin());
  std::vector<float> v2984(496);
  std::copy(v2982.begin() + 528, v2982.begin() + 528 + 496, v2984.begin());
  std::copy(v2983.begin(), v2983.end(), v86.begin() + 496);
  std::copy(v2984.begin(), v2984.end(), v86.begin() + 0);
  std::vector<double> v2987(std::begin(v86), std::end(v86));
  auto pt499_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt499_filled = v2987;
  pt499_filled.clear();
  pt499_filled.reserve(pt499_filled_n);
  for (auto i = 0; i < pt499_filled_n; ++i) {
    pt499_filled.push_back(v2987[i % v2987.size()]);
  }
  auto pt499 = cc->MakeCKKSPackedPlaintext(pt499_filled);
  const auto& ct1010 = cc->EvalMult(ct6, pt499);
  std::vector<float> v2988(std::begin(v28) + 500 * 512, std::begin(v28) + 500 * 512 + 1024);
  std::vector<float> v2989(528);
  std::copy(v2988.begin() + 0, v2988.begin() + 0 + 528, v2989.begin());
  std::vector<float> v2990(496);
  std::copy(v2988.begin() + 528, v2988.begin() + 528 + 496, v2990.begin());
  std::copy(v2989.begin(), v2989.end(), v86.begin() + 496);
  std::copy(v2990.begin(), v2990.end(), v86.begin() + 0);
  std::vector<double> v2993(std::begin(v86), std::end(v86));
  auto pt500_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt500_filled = v2993;
  pt500_filled.clear();
  pt500_filled.reserve(pt500_filled_n);
  for (auto i = 0; i < pt500_filled_n; ++i) {
    pt500_filled.push_back(v2993[i % v2993.size()]);
  }
  auto pt500 = cc->MakeCKKSPackedPlaintext(pt500_filled);
  const auto& ct1011 = cc->EvalMult(ct8, pt500);
  std::vector<float> v2994(std::begin(v28) + 501 * 512, std::begin(v28) + 501 * 512 + 1024);
  std::vector<float> v2995(528);
  std::copy(v2994.begin() + 0, v2994.begin() + 0 + 528, v2995.begin());
  std::vector<float> v2996(496);
  std::copy(v2994.begin() + 528, v2994.begin() + 528 + 496, v2996.begin());
  std::copy(v2995.begin(), v2995.end(), v86.begin() + 496);
  std::copy(v2996.begin(), v2996.end(), v86.begin() + 0);
  std::vector<double> v2999(std::begin(v86), std::end(v86));
  auto pt501_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt501_filled = v2999;
  pt501_filled.clear();
  pt501_filled.reserve(pt501_filled_n);
  for (auto i = 0; i < pt501_filled_n; ++i) {
    pt501_filled.push_back(v2999[i % v2999.size()]);
  }
  auto pt501 = cc->MakeCKKSPackedPlaintext(pt501_filled);
  const auto& ct1012 = cc->EvalMult(ct10, pt501);
  std::vector<float> v3000(std::begin(v28) + 502 * 512, std::begin(v28) + 502 * 512 + 1024);
  std::vector<float> v3001(528);
  std::copy(v3000.begin() + 0, v3000.begin() + 0 + 528, v3001.begin());
  std::vector<float> v3002(496);
  std::copy(v3000.begin() + 528, v3000.begin() + 528 + 496, v3002.begin());
  std::copy(v3001.begin(), v3001.end(), v86.begin() + 496);
  std::copy(v3002.begin(), v3002.end(), v86.begin() + 0);
  std::vector<double> v3005(std::begin(v86), std::end(v86));
  auto pt502_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt502_filled = v3005;
  pt502_filled.clear();
  pt502_filled.reserve(pt502_filled_n);
  for (auto i = 0; i < pt502_filled_n; ++i) {
    pt502_filled.push_back(v3005[i % v3005.size()]);
  }
  auto pt502 = cc->MakeCKKSPackedPlaintext(pt502_filled);
  const auto& ct1013 = cc->EvalMult(ct12, pt502);
  std::vector<float> v3006(std::begin(v28) + 503 * 512, std::begin(v28) + 503 * 512 + 1024);
  std::vector<float> v3007(528);
  std::copy(v3006.begin() + 0, v3006.begin() + 0 + 528, v3007.begin());
  std::vector<float> v3008(496);
  std::copy(v3006.begin() + 528, v3006.begin() + 528 + 496, v3008.begin());
  std::copy(v3007.begin(), v3007.end(), v86.begin() + 496);
  std::copy(v3008.begin(), v3008.end(), v86.begin() + 0);
  std::vector<double> v3011(std::begin(v86), std::end(v86));
  auto pt503_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt503_filled = v3011;
  pt503_filled.clear();
  pt503_filled.reserve(pt503_filled_n);
  for (auto i = 0; i < pt503_filled_n; ++i) {
    pt503_filled.push_back(v3011[i % v3011.size()]);
  }
  auto pt503 = cc->MakeCKKSPackedPlaintext(pt503_filled);
  const auto& ct1014 = cc->EvalMult(ct14, pt503);
  std::vector<float> v3012(std::begin(v28) + 504 * 512, std::begin(v28) + 504 * 512 + 1024);
  std::vector<float> v3013(528);
  std::copy(v3012.begin() + 0, v3012.begin() + 0 + 528, v3013.begin());
  std::vector<float> v3014(496);
  std::copy(v3012.begin() + 528, v3012.begin() + 528 + 496, v3014.begin());
  std::copy(v3013.begin(), v3013.end(), v86.begin() + 496);
  std::copy(v3014.begin(), v3014.end(), v86.begin() + 0);
  std::vector<double> v3017(std::begin(v86), std::end(v86));
  auto pt504_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt504_filled = v3017;
  pt504_filled.clear();
  pt504_filled.reserve(pt504_filled_n);
  for (auto i = 0; i < pt504_filled_n; ++i) {
    pt504_filled.push_back(v3017[i % v3017.size()]);
  }
  auto pt504 = cc->MakeCKKSPackedPlaintext(pt504_filled);
  const auto& ct1015 = cc->EvalMult(ct16, pt504);
  std::vector<float> v3018(std::begin(v28) + 505 * 512, std::begin(v28) + 505 * 512 + 1024);
  std::vector<float> v3019(528);
  std::copy(v3018.begin() + 0, v3018.begin() + 0 + 528, v3019.begin());
  std::vector<float> v3020(496);
  std::copy(v3018.begin() + 528, v3018.begin() + 528 + 496, v3020.begin());
  std::copy(v3019.begin(), v3019.end(), v86.begin() + 496);
  std::copy(v3020.begin(), v3020.end(), v86.begin() + 0);
  std::vector<double> v3023(std::begin(v86), std::end(v86));
  auto pt505_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt505_filled = v3023;
  pt505_filled.clear();
  pt505_filled.reserve(pt505_filled_n);
  for (auto i = 0; i < pt505_filled_n; ++i) {
    pt505_filled.push_back(v3023[i % v3023.size()]);
  }
  auto pt505 = cc->MakeCKKSPackedPlaintext(pt505_filled);
  const auto& ct1016 = cc->EvalMult(ct18, pt505);
  std::vector<float> v3024(std::begin(v28) + 506 * 512, std::begin(v28) + 506 * 512 + 1024);
  std::vector<float> v3025(528);
  std::copy(v3024.begin() + 0, v3024.begin() + 0 + 528, v3025.begin());
  std::vector<float> v3026(496);
  std::copy(v3024.begin() + 528, v3024.begin() + 528 + 496, v3026.begin());
  std::copy(v3025.begin(), v3025.end(), v86.begin() + 496);
  std::copy(v3026.begin(), v3026.end(), v86.begin() + 0);
  std::vector<double> v3029(std::begin(v86), std::end(v86));
  auto pt506_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt506_filled = v3029;
  pt506_filled.clear();
  pt506_filled.reserve(pt506_filled_n);
  for (auto i = 0; i < pt506_filled_n; ++i) {
    pt506_filled.push_back(v3029[i % v3029.size()]);
  }
  auto pt506 = cc->MakeCKKSPackedPlaintext(pt506_filled);
  const auto& ct1017 = cc->EvalMult(ct20, pt506);
  std::vector<float> v3030(std::begin(v28) + 507 * 512, std::begin(v28) + 507 * 512 + 1024);
  std::vector<float> v3031(528);
  std::copy(v3030.begin() + 0, v3030.begin() + 0 + 528, v3031.begin());
  std::vector<float> v3032(496);
  std::copy(v3030.begin() + 528, v3030.begin() + 528 + 496, v3032.begin());
  std::copy(v3031.begin(), v3031.end(), v86.begin() + 496);
  std::copy(v3032.begin(), v3032.end(), v86.begin() + 0);
  std::vector<double> v3035(std::begin(v86), std::end(v86));
  auto pt507_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt507_filled = v3035;
  pt507_filled.clear();
  pt507_filled.reserve(pt507_filled_n);
  for (auto i = 0; i < pt507_filled_n; ++i) {
    pt507_filled.push_back(v3035[i % v3035.size()]);
  }
  auto pt507 = cc->MakeCKKSPackedPlaintext(pt507_filled);
  const auto& ct1018 = cc->EvalMult(ct22, pt507);
  std::vector<float> v3036(std::begin(v28) + 508 * 512, std::begin(v28) + 508 * 512 + 1024);
  std::vector<float> v3037(528);
  std::copy(v3036.begin() + 0, v3036.begin() + 0 + 528, v3037.begin());
  std::vector<float> v3038(496);
  std::copy(v3036.begin() + 528, v3036.begin() + 528 + 496, v3038.begin());
  std::copy(v3037.begin(), v3037.end(), v86.begin() + 496);
  std::copy(v3038.begin(), v3038.end(), v86.begin() + 0);
  std::vector<double> v3041(std::begin(v86), std::end(v86));
  auto pt508_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt508_filled = v3041;
  pt508_filled.clear();
  pt508_filled.reserve(pt508_filled_n);
  for (auto i = 0; i < pt508_filled_n; ++i) {
    pt508_filled.push_back(v3041[i % v3041.size()]);
  }
  auto pt508 = cc->MakeCKKSPackedPlaintext(pt508_filled);
  const auto& ct1019 = cc->EvalMult(ct24, pt508);
  std::vector<float> v3042(std::begin(v28) + 509 * 512, std::begin(v28) + 509 * 512 + 1024);
  std::vector<float> v3043(528);
  std::copy(v3042.begin() + 0, v3042.begin() + 0 + 528, v3043.begin());
  std::vector<float> v3044(496);
  std::copy(v3042.begin() + 528, v3042.begin() + 528 + 496, v3044.begin());
  std::copy(v3043.begin(), v3043.end(), v86.begin() + 496);
  std::copy(v3044.begin(), v3044.end(), v86.begin() + 0);
  std::vector<double> v3047(std::begin(v86), std::end(v86));
  auto pt509_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt509_filled = v3047;
  pt509_filled.clear();
  pt509_filled.reserve(pt509_filled_n);
  for (auto i = 0; i < pt509_filled_n; ++i) {
    pt509_filled.push_back(v3047[i % v3047.size()]);
  }
  auto pt509 = cc->MakeCKKSPackedPlaintext(pt509_filled);
  const auto& ct1020 = cc->EvalMult(ct26, pt509);
  std::vector<float> v3048(std::begin(v28) + 510 * 512, std::begin(v28) + 510 * 512 + 1024);
  std::vector<float> v3049(528);
  std::copy(v3048.begin() + 0, v3048.begin() + 0 + 528, v3049.begin());
  std::vector<float> v3050(496);
  std::copy(v3048.begin() + 528, v3048.begin() + 528 + 496, v3050.begin());
  std::copy(v3049.begin(), v3049.end(), v86.begin() + 496);
  std::copy(v3050.begin(), v3050.end(), v86.begin() + 0);
  std::vector<double> v3053(std::begin(v86), std::end(v86));
  auto pt510_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt510_filled = v3053;
  pt510_filled.clear();
  pt510_filled.reserve(pt510_filled_n);
  for (auto i = 0; i < pt510_filled_n; ++i) {
    pt510_filled.push_back(v3053[i % v3053.size()]);
  }
  auto pt510 = cc->MakeCKKSPackedPlaintext(pt510_filled);
  const auto& ct1021 = cc->EvalMult(ct28, pt510);
  std::vector<float> v3054(std::begin(v28) + 511 * 512, std::begin(v28) + 511 * 512 + 1024);
  std::vector<float> v3055(528);
  std::copy(v3054.begin() + 0, v3054.begin() + 0 + 528, v3055.begin());
  std::vector<float> v3056(496);
  std::copy(v3054.begin() + 528, v3054.begin() + 528 + 496, v3056.begin());
  std::copy(v3055.begin(), v3055.end(), v86.begin() + 496);
  std::copy(v3056.begin(), v3056.end(), v86.begin() + 0);
  std::vector<double> v3059(std::begin(v86), std::end(v86));
  auto pt511_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt511_filled = v3059;
  pt511_filled.clear();
  pt511_filled.reserve(pt511_filled_n);
  for (auto i = 0; i < pt511_filled_n; ++i) {
    pt511_filled.push_back(v3059[i % v3059.size()]);
  }
  auto pt511 = cc->MakeCKKSPackedPlaintext(pt511_filled);
  const auto& ct1022 = cc->EvalMult(ct30, pt511);
  const auto& ct1023 = cc->EvalAdd(ct1007, ct1008);
  const auto& ct1024 = cc->EvalAdd(ct1009, ct1010);
  const auto& ct1025 = cc->EvalAdd(ct1023, ct1024);
  const auto& ct1026 = cc->EvalAdd(ct1011, ct1012);
  const auto& ct1027 = cc->EvalAdd(ct1013, ct1014);
  const auto& ct1028 = cc->EvalAdd(ct1026, ct1027);
  const auto& ct1029 = cc->EvalAdd(ct1025, ct1028);
  const auto& ct1030 = cc->EvalAdd(ct1015, ct1016);
  const auto& ct1031 = cc->EvalAdd(ct1017, ct1018);
  const auto& ct1032 = cc->EvalAdd(ct1030, ct1031);
  const auto& ct1033 = cc->EvalAdd(ct1019, ct1020);
  const auto& ct1034 = cc->EvalAdd(ct1021, ct1022);
  const auto& ct1035 = cc->EvalAdd(ct1033, ct1034);
  const auto& ct1036 = cc->EvalAdd(ct1032, ct1035);
  const auto& ct1037 = cc->EvalAdd(ct1029, ct1036);
  const auto& ct1038 = cc->EvalRotate(ct1037, 496);
  const auto& ct1039 = cc->EvalAdd(ct46, ct78);
  const auto& ct1040 = cc->EvalAdd(ct110, ct142);
  const auto& ct1041 = cc->EvalAdd(ct1039, ct1040);
  const auto& ct1042 = cc->EvalAdd(ct174, ct206);
  const auto& ct1043 = cc->EvalAdd(ct238, ct270);
  const auto& ct1044 = cc->EvalAdd(ct1042, ct1043);
  const auto& ct1045 = cc->EvalAdd(ct1041, ct1044);
  const auto& ct1046 = cc->EvalAdd(ct302, ct334);
  const auto& ct1047 = cc->EvalAdd(ct366, ct398);
  const auto& ct1048 = cc->EvalAdd(ct1046, ct1047);
  const auto& ct1049 = cc->EvalAdd(ct430, ct462);
  const auto& ct1050 = cc->EvalAdd(ct494, ct526);
  const auto& ct1051 = cc->EvalAdd(ct1049, ct1050);
  const auto& ct1052 = cc->EvalAdd(ct1048, ct1051);
  const auto& ct1053 = cc->EvalAdd(ct1045, ct1052);
  const auto& ct1054 = cc->EvalAdd(ct558, ct590);
  const auto& ct1055 = cc->EvalAdd(ct622, ct654);
  const auto& ct1056 = cc->EvalAdd(ct1054, ct1055);
  const auto& ct1057 = cc->EvalAdd(ct686, ct718);
  const auto& ct1058 = cc->EvalAdd(ct750, ct782);
  const auto& ct1059 = cc->EvalAdd(ct1057, ct1058);
  const auto& ct1060 = cc->EvalAdd(ct1056, ct1059);
  const auto& ct1061 = cc->EvalAdd(ct814, ct846);
  const auto& ct1062 = cc->EvalAdd(ct878, ct910);
  const auto& ct1063 = cc->EvalAdd(ct1061, ct1062);
  const auto& ct1064 = cc->EvalAdd(ct942, ct974);
  const auto& ct1065 = cc->EvalAdd(ct1006, ct1038);
  std::vector<double> v3061(std::begin(v47), std::end(v47));
  auto pt512_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt512_filled = v3061;
  pt512_filled.clear();
  pt512_filled.reserve(pt512_filled_n);
  for (auto i = 0; i < pt512_filled_n; ++i) {
    pt512_filled.push_back(v3061[i % v3061.size()]);
  }
  auto pt512 = cc->MakeCKKSPackedPlaintext(pt512_filled);
  const auto& ct1066 = cc->EvalAdd(ct1065, pt512);
  const auto& ct1067 = cc->EvalAdd(ct1064, ct1066);
  const auto& ct1068 = cc->EvalAdd(ct1063, ct1067);
  const auto& ct1069 = cc->EvalAdd(ct1060, ct1068);
  const auto& ct1070 = cc->EvalAdd(ct1053, ct1069);
  const auto& ct1071 = cc->EvalRotate(ct1070, 512);
  std::vector<float> v3062 = v20;
  for (auto v3063 = 0; v3063 < 1024; ++v3063) {
    size_t v3065 = v3063 % v24;
    float v3066 = v1[v3065];
    v3062[v3063 + 1024 * (0)] = v3066;
  }
  std::vector<double> v3069(std::begin(v3062), std::end(v3062));
  auto pt513_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt513_filled = v3069;
  pt513_filled.clear();
  pt513_filled.reserve(pt513_filled_n);
  for (auto i = 0; i < pt513_filled_n; ++i) {
    pt513_filled.push_back(v3069[i % v3069.size()]);
  }
  auto pt513 = cc->MakeCKKSPackedPlaintext(pt513_filled);
  const auto& ct1072 = cc->EvalAdd(ct1070, pt513);
  const auto& ct1073 = cc->EvalAdd(ct1072, ct1071);
  std::vector<float> v3070 = v20;
  for (auto v3071 = 0; v3071 < 1024; ++v3071) {
    v3070[v3071 + 1024 * (0)] = v18;
  }
  const auto& ct1074 = cc->ModReduce(ct1073);
  std::vector<double> v3075(std::begin(v3070), std::end(v3070));
  auto pt514_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt514_filled = v3075;
  pt514_filled.clear();
  pt514_filled.reserve(pt514_filled_n);
  for (auto i = 0; i < pt514_filled_n; ++i) {
    pt514_filled.push_back(v3075[i % v3075.size()]);
  }
  auto pt514 = cc->MakeCKKSPackedPlaintext(pt514_filled);
  const auto& ct1075 = cc->EvalMult(ct1074, pt514);
  std::vector<float> v3076 = v20;
  for (auto v3077 = 0; v3077 < 1024; ++v3077) {
    v3076[v3077 + 1024 * (0)] = v17;
  }
  std::vector<double> v3081(std::begin(v3076), std::end(v3076));
  auto pt515_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt515_filled = v3081;
  pt515_filled.clear();
  pt515_filled.reserve(pt515_filled_n);
  for (auto i = 0; i < pt515_filled_n; ++i) {
    pt515_filled.push_back(v3081[i % v3081.size()]);
  }
  auto pt515 = cc->MakeCKKSPackedPlaintext(pt515_filled);
  const auto& ct1076 = cc->EvalAdd(ct1075, pt515);
  const auto& ct1077 = cc->ModReduce(ct1076);
  const auto& ct1078 = cc->LevelReduce(ct1073, nullptr, 1);
  const auto& ct1079 = cc->ModReduce(ct1078);
  const auto& ct1080 = cc->EvalMultNoRelin(ct1077, ct1079);
  std::vector<float> v3082 = v20;
  for (auto v3083 = 0; v3083 < 1024; ++v3083) {
    v3082[v3083 + 1024 * (0)] = v16;
  }
  std::vector<double> v3087(std::begin(v3082), std::end(v3082));
  auto pt516_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt516_filled = v3087;
  pt516_filled.clear();
  pt516_filled.reserve(pt516_filled_n);
  for (auto i = 0; i < pt516_filled_n; ++i) {
    pt516_filled.push_back(v3087[i % v3087.size()]);
  }
  auto pt516 = cc->MakeCKKSPackedPlaintext(pt516_filled);
  const auto& ct1081 = cc->EvalAdd(ct1080, pt516);
  const auto& ct1082 = cc->Relinearize(ct1081);
  const auto& ct1083 = cc->ModReduce(ct1082);
  const auto& ct1084 = cc->LevelReduce(ct1073, nullptr, 2);
  const auto& ct1085 = cc->ModReduce(ct1084);
  const auto& ct1086 = cc->EvalMultNoRelin(ct1083, ct1085);
  std::vector<float> v3088 = v20;
  for (auto v3089 = 0; v3089 < 1024; ++v3089) {
    v3088[v3089 + 1024 * (0)] = v15;
  }
  std::vector<double> v3093(std::begin(v3088), std::end(v3088));
  auto pt517_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt517_filled = v3093;
  pt517_filled.clear();
  pt517_filled.reserve(pt517_filled_n);
  for (auto i = 0; i < pt517_filled_n; ++i) {
    pt517_filled.push_back(v3093[i % v3093.size()]);
  }
  auto pt517 = cc->MakeCKKSPackedPlaintext(pt517_filled);
  const auto& ct1087 = cc->EvalAdd(ct1086, pt517);
  const auto& ct1088 = cc->Relinearize(ct1087);
  const auto& ct1089 = cc->ModReduce(ct1088);
  const auto& ct1090 = cc->LevelReduce(ct1073, nullptr, 3);
  const auto& ct1091 = cc->ModReduce(ct1090);
  const auto& ct1092 = cc->EvalMultNoRelin(ct1089, ct1091);
  std::vector<float> v3094 = v20;
  for (auto v3095 = 0; v3095 < 1024; ++v3095) {
    v3094[v3095 + 1024 * (0)] = v14;
  }
  std::vector<double> v3099(std::begin(v3094), std::end(v3094));
  auto pt518_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt518_filled = v3099;
  pt518_filled.clear();
  pt518_filled.reserve(pt518_filled_n);
  for (auto i = 0; i < pt518_filled_n; ++i) {
    pt518_filled.push_back(v3099[i % v3099.size()]);
  }
  auto pt518 = cc->MakeCKKSPackedPlaintext(pt518_filled);
  const auto& ct1093 = cc->EvalAdd(ct1092, pt518);
  const auto& ct1094 = cc->Relinearize(ct1093);
  const auto& ct1095 = cc->ModReduce(ct1094);
  const auto& ct1096 = cc->LevelReduce(ct1073, nullptr, 4);
  const auto& ct1097 = cc->ModReduce(ct1096);
  const auto& ct1098 = cc->EvalMultNoRelin(ct1095, ct1097);
  std::vector<float> v3100 = v20;
  for (auto v3101 = 0; v3101 < 1024; ++v3101) {
    v3100[v3101 + 1024 * (0)] = v13;
  }
  std::vector<double> v3105(std::begin(v3100), std::end(v3100));
  auto pt519_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt519_filled = v3105;
  pt519_filled.clear();
  pt519_filled.reserve(pt519_filled_n);
  for (auto i = 0; i < pt519_filled_n; ++i) {
    pt519_filled.push_back(v3105[i % v3105.size()]);
  }
  auto pt519 = cc->MakeCKKSPackedPlaintext(pt519_filled);
  const auto& ct1099 = cc->EvalAdd(ct1098, pt519);
  const auto& ct1100 = cc->Relinearize(ct1099);
  std::vector<float> v3106 = v12;
  for (auto v3107 = 0; v3107 < 16; ++v3107) {
    for (auto v3110 = 0; v3110 < 1018; ++v3110) {
      size_t v3112 = v3110 % v11;
      bool v3113 = v3112 <= v9;
      if (v3113) {
        size_t v3115 = v3110 + v8;
        size_t v3116 = v3115 % v11;
        size_t v3117 = v3116 - v8;
        size_t v3118 = v26 - v3107;
        size_t v3119 = v3118 - v3110;
        size_t v3120 = v3119 + v7;
        size_t v3121 = v3120 % v24;
        size_t v3122 = v5 - v3121;
        float v3123 = v2[v3122 + 512 * (v3117)];
        v3106[v3110 + 1024 * (v3107)] = v3123;
      }
    }
  }
  std::vector<float> v3125 = v20;
  for (auto v3126 = 0; v3126 < 1024; ++v3126) {
    size_t v3128 = v3126 + v8;
    size_t v3129 = v3128 % v11;
    bool v3130 = v3129 >= v8;
    if (v3130) {
      v3125[v3126 + 1024 * (0)] = v19;
    }
  }
  std::vector<float> v3133(std::begin(v3106) + 0 * 16, std::begin(v3106) + 0 * 16 + 1024);
  const auto& ct1101 = cc->ModReduce(ct1100);
  std::vector<double> v3134(std::begin(v3133), std::end(v3133));
  auto pt520_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt520_filled = v3134;
  pt520_filled.clear();
  pt520_filled.reserve(pt520_filled_n);
  for (auto i = 0; i < pt520_filled_n; ++i) {
    pt520_filled.push_back(v3134[i % v3134.size()]);
  }
  auto pt520 = cc->MakeCKKSPackedPlaintext(pt520_filled);
  const auto& ct1102 = cc->EvalMult(ct1101, pt520);
  std::vector<float> v3135(std::begin(v3106) + 1 * 16, std::begin(v3106) + 1 * 16 + 1024);
  const auto& ct1103 = cc->EvalRotate(ct1100, 1);
  const auto& ct1104 = cc->ModReduce(ct1103);
  std::vector<double> v3136(std::begin(v3135), std::end(v3135));
  auto pt521_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt521_filled = v3136;
  pt521_filled.clear();
  pt521_filled.reserve(pt521_filled_n);
  for (auto i = 0; i < pt521_filled_n; ++i) {
    pt521_filled.push_back(v3136[i % v3136.size()]);
  }
  auto pt521 = cc->MakeCKKSPackedPlaintext(pt521_filled);
  const auto& ct1105 = cc->EvalMult(ct1104, pt521);
  std::vector<float> v3137(std::begin(v3106) + 2 * 16, std::begin(v3106) + 2 * 16 + 1024);
  const auto& ct1106 = cc->EvalRotate(ct1100, 2);
  const auto& ct1107 = cc->ModReduce(ct1106);
  std::vector<double> v3138(std::begin(v3137), std::end(v3137));
  auto pt522_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt522_filled = v3138;
  pt522_filled.clear();
  pt522_filled.reserve(pt522_filled_n);
  for (auto i = 0; i < pt522_filled_n; ++i) {
    pt522_filled.push_back(v3138[i % v3138.size()]);
  }
  auto pt522 = cc->MakeCKKSPackedPlaintext(pt522_filled);
  const auto& ct1108 = cc->EvalMult(ct1107, pt522);
  std::vector<float> v3139(std::begin(v3106) + 3 * 16, std::begin(v3106) + 3 * 16 + 1024);
  const auto& ct1109 = cc->EvalRotate(ct1100, 3);
  const auto& ct1110 = cc->ModReduce(ct1109);
  std::vector<double> v3140(std::begin(v3139), std::end(v3139));
  auto pt523_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt523_filled = v3140;
  pt523_filled.clear();
  pt523_filled.reserve(pt523_filled_n);
  for (auto i = 0; i < pt523_filled_n; ++i) {
    pt523_filled.push_back(v3140[i % v3140.size()]);
  }
  auto pt523 = cc->MakeCKKSPackedPlaintext(pt523_filled);
  const auto& ct1111 = cc->EvalMult(ct1110, pt523);
  const auto& ct1112 = cc->EvalAdd(ct1102, ct1105);
  const auto& ct1113 = cc->EvalAdd(ct1108, ct1111);
  const auto& ct1114 = cc->EvalAdd(ct1112, ct1113);
  std::vector<float> v3141(std::begin(v3106) + 4 * 16, std::begin(v3106) + 4 * 16 + 1024);
  std::vector<float> v3142(1020);
  std::copy(v3141.begin() + 0, v3141.begin() + 0 + 1020, v3142.begin());
  std::vector<float> v3143(4);
  std::copy(v3141.begin() + 1020, v3141.begin() + 1020 + 4, v3143.begin());
  std::copy(v3142.begin(), v3142.end(), v86.begin() + 4);
  std::copy(v3143.begin(), v3143.end(), v86.begin() + 0);
  std::vector<double> v3146(std::begin(v86), std::end(v86));
  auto pt524_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt524_filled = v3146;
  pt524_filled.clear();
  pt524_filled.reserve(pt524_filled_n);
  for (auto i = 0; i < pt524_filled_n; ++i) {
    pt524_filled.push_back(v3146[i % v3146.size()]);
  }
  auto pt524 = cc->MakeCKKSPackedPlaintext(pt524_filled);
  const auto& ct1115 = cc->EvalMult(ct1101, pt524);
  std::vector<float> v3147(std::begin(v3106) + 5 * 16, std::begin(v3106) + 5 * 16 + 1024);
  std::vector<float> v3148(1020);
  std::copy(v3147.begin() + 0, v3147.begin() + 0 + 1020, v3148.begin());
  std::vector<float> v3149(4);
  std::copy(v3147.begin() + 1020, v3147.begin() + 1020 + 4, v3149.begin());
  std::copy(v3148.begin(), v3148.end(), v86.begin() + 4);
  std::copy(v3149.begin(), v3149.end(), v86.begin() + 0);
  std::vector<double> v3152(std::begin(v86), std::end(v86));
  auto pt525_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt525_filled = v3152;
  pt525_filled.clear();
  pt525_filled.reserve(pt525_filled_n);
  for (auto i = 0; i < pt525_filled_n; ++i) {
    pt525_filled.push_back(v3152[i % v3152.size()]);
  }
  auto pt525 = cc->MakeCKKSPackedPlaintext(pt525_filled);
  const auto& ct1116 = cc->EvalMult(ct1104, pt525);
  std::vector<float> v3153(std::begin(v3106) + 6 * 16, std::begin(v3106) + 6 * 16 + 1024);
  std::vector<float> v3154(1020);
  std::copy(v3153.begin() + 0, v3153.begin() + 0 + 1020, v3154.begin());
  std::vector<float> v3155(4);
  std::copy(v3153.begin() + 1020, v3153.begin() + 1020 + 4, v3155.begin());
  std::copy(v3154.begin(), v3154.end(), v86.begin() + 4);
  std::copy(v3155.begin(), v3155.end(), v86.begin() + 0);
  std::vector<double> v3158(std::begin(v86), std::end(v86));
  auto pt526_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt526_filled = v3158;
  pt526_filled.clear();
  pt526_filled.reserve(pt526_filled_n);
  for (auto i = 0; i < pt526_filled_n; ++i) {
    pt526_filled.push_back(v3158[i % v3158.size()]);
  }
  auto pt526 = cc->MakeCKKSPackedPlaintext(pt526_filled);
  const auto& ct1117 = cc->EvalMult(ct1107, pt526);
  std::vector<float> v3159(std::begin(v3106) + 7 * 16, std::begin(v3106) + 7 * 16 + 1024);
  std::vector<float> v3160(1020);
  std::copy(v3159.begin() + 0, v3159.begin() + 0 + 1020, v3160.begin());
  std::vector<float> v3161(4);
  std::copy(v3159.begin() + 1020, v3159.begin() + 1020 + 4, v3161.begin());
  std::copy(v3160.begin(), v3160.end(), v86.begin() + 4);
  std::copy(v3161.begin(), v3161.end(), v86.begin() + 0);
  std::vector<double> v3164(std::begin(v86), std::end(v86));
  auto pt527_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt527_filled = v3164;
  pt527_filled.clear();
  pt527_filled.reserve(pt527_filled_n);
  for (auto i = 0; i < pt527_filled_n; ++i) {
    pt527_filled.push_back(v3164[i % v3164.size()]);
  }
  auto pt527 = cc->MakeCKKSPackedPlaintext(pt527_filled);
  const auto& ct1118 = cc->EvalMult(ct1110, pt527);
  const auto& ct1119 = cc->EvalAdd(ct1115, ct1116);
  const auto& ct1120 = cc->EvalAdd(ct1117, ct1118);
  const auto& ct1121 = cc->EvalAdd(ct1119, ct1120);
  const auto& ct1122 = cc->EvalRotate(ct1121, 4);
  std::vector<float> v3165(std::begin(v3106) + 8 * 16, std::begin(v3106) + 8 * 16 + 1024);
  std::vector<float> v3166(1016);
  std::copy(v3165.begin() + 0, v3165.begin() + 0 + 1016, v3166.begin());
  std::vector<float> v3167(8);
  std::copy(v3165.begin() + 1016, v3165.begin() + 1016 + 8, v3167.begin());
  std::copy(v3166.begin(), v3166.end(), v86.begin() + 8);
  std::copy(v3167.begin(), v3167.end(), v86.begin() + 0);
  std::vector<double> v3170(std::begin(v86), std::end(v86));
  auto pt528_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt528_filled = v3170;
  pt528_filled.clear();
  pt528_filled.reserve(pt528_filled_n);
  for (auto i = 0; i < pt528_filled_n; ++i) {
    pt528_filled.push_back(v3170[i % v3170.size()]);
  }
  auto pt528 = cc->MakeCKKSPackedPlaintext(pt528_filled);
  const auto& ct1123 = cc->EvalMult(ct1101, pt528);
  std::vector<float> v3171(std::begin(v3106) + 9 * 16, std::begin(v3106) + 9 * 16 + 1024);
  std::vector<float> v3172(1016);
  std::copy(v3171.begin() + 0, v3171.begin() + 0 + 1016, v3172.begin());
  std::vector<float> v3173(8);
  std::copy(v3171.begin() + 1016, v3171.begin() + 1016 + 8, v3173.begin());
  std::copy(v3172.begin(), v3172.end(), v86.begin() + 8);
  std::copy(v3173.begin(), v3173.end(), v86.begin() + 0);
  std::vector<double> v3176(std::begin(v86), std::end(v86));
  auto pt529_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt529_filled = v3176;
  pt529_filled.clear();
  pt529_filled.reserve(pt529_filled_n);
  for (auto i = 0; i < pt529_filled_n; ++i) {
    pt529_filled.push_back(v3176[i % v3176.size()]);
  }
  auto pt529 = cc->MakeCKKSPackedPlaintext(pt529_filled);
  const auto& ct1124 = cc->EvalMult(ct1104, pt529);
  std::vector<float> v3177(std::begin(v3106) + 10 * 16, std::begin(v3106) + 10 * 16 + 1024);
  std::vector<float> v3178(1016);
  std::copy(v3177.begin() + 0, v3177.begin() + 0 + 1016, v3178.begin());
  std::vector<float> v3179(8);
  std::copy(v3177.begin() + 1016, v3177.begin() + 1016 + 8, v3179.begin());
  std::copy(v3178.begin(), v3178.end(), v86.begin() + 8);
  std::copy(v3179.begin(), v3179.end(), v86.begin() + 0);
  std::vector<double> v3182(std::begin(v86), std::end(v86));
  auto pt530_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt530_filled = v3182;
  pt530_filled.clear();
  pt530_filled.reserve(pt530_filled_n);
  for (auto i = 0; i < pt530_filled_n; ++i) {
    pt530_filled.push_back(v3182[i % v3182.size()]);
  }
  auto pt530 = cc->MakeCKKSPackedPlaintext(pt530_filled);
  const auto& ct1125 = cc->EvalMult(ct1107, pt530);
  std::vector<float> v3183(std::begin(v3106) + 11 * 16, std::begin(v3106) + 11 * 16 + 1024);
  std::vector<float> v3184(1016);
  std::copy(v3183.begin() + 0, v3183.begin() + 0 + 1016, v3184.begin());
  std::vector<float> v3185(8);
  std::copy(v3183.begin() + 1016, v3183.begin() + 1016 + 8, v3185.begin());
  std::copy(v3184.begin(), v3184.end(), v86.begin() + 8);
  std::copy(v3185.begin(), v3185.end(), v86.begin() + 0);
  std::vector<double> v3188(std::begin(v86), std::end(v86));
  auto pt531_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt531_filled = v3188;
  pt531_filled.clear();
  pt531_filled.reserve(pt531_filled_n);
  for (auto i = 0; i < pt531_filled_n; ++i) {
    pt531_filled.push_back(v3188[i % v3188.size()]);
  }
  auto pt531 = cc->MakeCKKSPackedPlaintext(pt531_filled);
  const auto& ct1126 = cc->EvalMult(ct1110, pt531);
  const auto& ct1127 = cc->EvalAdd(ct1123, ct1124);
  const auto& ct1128 = cc->EvalAdd(ct1125, ct1126);
  const auto& ct1129 = cc->EvalAdd(ct1127, ct1128);
  const auto& ct1130 = cc->EvalRotate(ct1129, 8);
  std::vector<float> v3189(std::begin(v3106) + 12 * 16, std::begin(v3106) + 12 * 16 + 1024);
  std::vector<float> v3190(1012);
  std::copy(v3189.begin() + 0, v3189.begin() + 0 + 1012, v3190.begin());
  std::vector<float> v3191(12);
  std::copy(v3189.begin() + 1012, v3189.begin() + 1012 + 12, v3191.begin());
  std::copy(v3190.begin(), v3190.end(), v86.begin() + 12);
  std::copy(v3191.begin(), v3191.end(), v86.begin() + 0);
  std::vector<double> v3194(std::begin(v86), std::end(v86));
  auto pt532_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt532_filled = v3194;
  pt532_filled.clear();
  pt532_filled.reserve(pt532_filled_n);
  for (auto i = 0; i < pt532_filled_n; ++i) {
    pt532_filled.push_back(v3194[i % v3194.size()]);
  }
  auto pt532 = cc->MakeCKKSPackedPlaintext(pt532_filled);
  const auto& ct1131 = cc->EvalMult(ct1101, pt532);
  std::vector<float> v3195(std::begin(v3106) + 13 * 16, std::begin(v3106) + 13 * 16 + 1024);
  std::vector<float> v3196(1012);
  std::copy(v3195.begin() + 0, v3195.begin() + 0 + 1012, v3196.begin());
  std::vector<float> v3197(12);
  std::copy(v3195.begin() + 1012, v3195.begin() + 1012 + 12, v3197.begin());
  std::copy(v3196.begin(), v3196.end(), v86.begin() + 12);
  std::copy(v3197.begin(), v3197.end(), v86.begin() + 0);
  std::vector<double> v3200(std::begin(v86), std::end(v86));
  auto pt533_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt533_filled = v3200;
  pt533_filled.clear();
  pt533_filled.reserve(pt533_filled_n);
  for (auto i = 0; i < pt533_filled_n; ++i) {
    pt533_filled.push_back(v3200[i % v3200.size()]);
  }
  auto pt533 = cc->MakeCKKSPackedPlaintext(pt533_filled);
  const auto& ct1132 = cc->EvalMult(ct1104, pt533);
  std::vector<float> v3201(std::begin(v3106) + 14 * 16, std::begin(v3106) + 14 * 16 + 1024);
  std::vector<float> v3202(1012);
  std::copy(v3201.begin() + 0, v3201.begin() + 0 + 1012, v3202.begin());
  std::vector<float> v3203(12);
  std::copy(v3201.begin() + 1012, v3201.begin() + 1012 + 12, v3203.begin());
  std::copy(v3202.begin(), v3202.end(), v86.begin() + 12);
  std::copy(v3203.begin(), v3203.end(), v86.begin() + 0);
  std::vector<double> v3206(std::begin(v86), std::end(v86));
  auto pt534_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt534_filled = v3206;
  pt534_filled.clear();
  pt534_filled.reserve(pt534_filled_n);
  for (auto i = 0; i < pt534_filled_n; ++i) {
    pt534_filled.push_back(v3206[i % v3206.size()]);
  }
  auto pt534 = cc->MakeCKKSPackedPlaintext(pt534_filled);
  const auto& ct1133 = cc->EvalMult(ct1107, pt534);
  std::vector<float> v3207(std::begin(v3106) + 15 * 16, std::begin(v3106) + 15 * 16 + 1024);
  std::vector<float> v3208(1012);
  std::copy(v3207.begin() + 0, v3207.begin() + 0 + 1012, v3208.begin());
  std::vector<float> v3209(12);
  std::copy(v3207.begin() + 1012, v3207.begin() + 1012 + 12, v3209.begin());
  std::copy(v3208.begin(), v3208.end(), v86.begin() + 12);
  std::copy(v3209.begin(), v3209.end(), v86.begin() + 0);
  std::vector<double> v3212(std::begin(v86), std::end(v86));
  auto pt535_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt535_filled = v3212;
  pt535_filled.clear();
  pt535_filled.reserve(pt535_filled_n);
  for (auto i = 0; i < pt535_filled_n; ++i) {
    pt535_filled.push_back(v3212[i % v3212.size()]);
  }
  auto pt535 = cc->MakeCKKSPackedPlaintext(pt535_filled);
  const auto& ct1134 = cc->EvalMult(ct1110, pt535);
  const auto& ct1135 = cc->EvalAdd(ct1131, ct1132);
  const auto& ct1136 = cc->EvalAdd(ct1133, ct1134);
  const auto& ct1137 = cc->EvalAdd(ct1135, ct1136);
  const auto& ct1138 = cc->EvalRotate(ct1137, 12);
  const auto& ct1139 = cc->EvalAdd(ct1114, ct1122);
  const auto& ct1140 = cc->EvalAdd(ct1130, ct1138);
  std::vector<double> v3214(std::begin(v3125), std::end(v3125));
  auto pt536_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt536_filled = v3214;
  pt536_filled.clear();
  pt536_filled.reserve(pt536_filled_n);
  for (auto i = 0; i < pt536_filled_n; ++i) {
    pt536_filled.push_back(v3214[i % v3214.size()]);
  }
  auto pt536 = cc->MakeCKKSPackedPlaintext(pt536_filled);
  const auto& ct1141 = cc->EvalAdd(ct1140, pt536);
  const auto& ct1142 = cc->EvalAdd(ct1139, ct1141);
  const auto& ct1143 = cc->EvalRotate(ct1142, 512);
  const auto& ct1144 = cc->EvalAdd(ct1142, ct1143);
  const auto& ct1145 = cc->EvalRotate(ct1144, 256);
  const auto& ct1146 = cc->EvalAdd(ct1144, ct1145);
  const auto& ct1147 = cc->EvalRotate(ct1146, 128);
  const auto& ct1148 = cc->EvalAdd(ct1146, ct1147);
  const auto& ct1149 = cc->EvalRotate(ct1148, 64);
  const auto& ct1150 = cc->EvalAdd(ct1148, ct1149);
  const auto& ct1151 = cc->EvalRotate(ct1150, 32);
  const auto& ct1152 = cc->EvalAdd(ct1150, ct1151);
  const auto& ct1153 = cc->EvalRotate(ct1152, 16);
  std::vector<float> v3215 = v20;
  for (auto v3216 = 0; v3216 < 1024; ++v3216) {
    size_t v3218 = v3216 + v8;
    size_t v3219 = v3218 % v11;
    bool v3220 = v3219 >= v8;
    if (v3220) {
      size_t v3222 = v3216 % v11;
      float v3223 = v3[v3222];
      v3215[v3216 + 1024 * (0)] = v3223;
    }
  }
  std::vector<double> v3226(std::begin(v3215), std::end(v3215));
  auto pt537_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt537_filled = v3226;
  pt537_filled.clear();
  pt537_filled.reserve(pt537_filled_n);
  for (auto i = 0; i < pt537_filled_n; ++i) {
    pt537_filled.push_back(v3226[i % v3226.size()]);
  }
  auto pt537 = cc->MakeCKKSPackedPlaintext(pt537_filled);
  const auto& ct1154 = cc->EvalAdd(ct1152, pt537);
  const auto& ct1155 = cc->EvalAdd(ct1154, ct1153);
  std::vector<CiphertextT> v3227(1);
  const auto& ct1156 = cc->ModReduce(ct1155);
  v3227[0] = ct1156;
  return v3227;
}
std::vector<CiphertextT> mnist__encrypt__arg4(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk) {
  std::vector<float> v1(1024, 0);
  [[maybe_unused]] size_t v2 = 0;
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 784;
  std::vector<float> v5 = v1;
  for (auto v6 = 0; v6 < 784; ++v6) {
    float v8 = v0[v6 + 784 * (0)];
    v5[v6 + 1024 * (0)] = v8;
  }
  std::vector<float> v10(std::begin(v5) + 0 * 1, std::begin(v5) + 0 * 1 + 1024);
  std::vector<double> v11(std::begin(v10), std::end(v10));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v11;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (auto i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v11[i % v11.size()]);
  }
  auto pt = cc->MakeCKKSPackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  std::vector<CiphertextT> v12{ct};
  return v12;
}
std::vector<float> mnist__decrypt__result0(CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 1024;
  [[maybe_unused]] size_t v2 = 16;
  [[maybe_unused]] size_t v3 = 6;
  [[maybe_unused]] size_t v4 = 1;
  [[maybe_unused]] size_t v5 = 0;
  std::vector<float> v6(10, 0);
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& v7_cast = pt->GetCKKSPackedValue();
  std::vector<float> v7(v7_cast.size());
  std::transform(std::begin(v7_cast), std::end(v7_cast), std::begin(v7), [](const std::complex<double>& c) { return c.real(); });
  std::vector<float> v8 = v6;
  for (auto v9 = 0; v9 < 1024; ++v9) {
    size_t v11 = v9 + v3;
    size_t v12 = v11 % v2;
    bool v13 = v12 >= v3;
    if (v13) {
      size_t v15 = v9 % v2;
      float v16 = v7[v9 + 1024 * (0)];
      v8[v15 + 10 * (0)] = v16;
    }
  }
  return v8;
}
CryptoContextT mnist__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(7);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT mnist__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512});
  return cc;
}
