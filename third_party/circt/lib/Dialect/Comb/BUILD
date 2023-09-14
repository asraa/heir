package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        "@heir//third_party/circt/include/circt/Dialect/Comb:dialect_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/Comb:enum_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/Comb:headers",
        "@heir//third_party/circt/include/circt/Dialect/Comb:ops_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/Comb:type_inc_gen",
        "@heir//third_party/circt/lib/Dialect/HW:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
