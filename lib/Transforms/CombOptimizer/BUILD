# CombOptmizer pass

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CombOptimizer",
    srcs = ["CombOptimizer.cpp"],
    hdrs = ["@heir//include/Transforms/CombOptimizer:CombOptimizer.h"],
    data = [
        ":map_lut_to_lut3.v",
        "@at_clifford_yosys//:share_files",
        "@edu_berkeley_abc//:abc",
    ],
    defines = [
        "ABCEXTERNAL=$(location @edu_berkeley_abc//:abc)",
    ],
    includes = ["@heir//include"],
    deps = [
        ":ImportRTLIL",
        "@at_clifford_yosys//:kernel",
        "@at_clifford_yosys//:version",
        "@at_clifford_yosys//:yosys",
        "@bazel_tools//tools/cpp/runfiles",
        "@circt//:CombHWDialect",
        "@edu_berkeley_abc//:abc",
        "@heir//include/Transforms/CombOptimizer:pass_inc_gen",
        "@heir//lib/Target/Verilog:VerilogEmitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "CombOptimizerRegistration",
    srcs = ["PassRegistration.cpp"],
    deps = [
        ":CombOptimizer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ImportRTLIL",
    srcs = ["ImportRTLIL.cpp"],
    hdrs = ["ImportRTLIL.h"],
    deps = [
        "@at_clifford_yosys//:kernel",
        "@at_clifford_yosys//:version",
        "@circt//:CombHWDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)
