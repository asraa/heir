package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = glob(
        [
            "*.cpp",
        ],
        exclude = [
            "HWReductions.cpp",
        ],
    ),
    hdrs = [
        "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
    ],
    deps = [
        "@heir//third_party/circt/include/circt/Dialect/Comb:headers",
        "@heir//third_party/circt/include/circt/Dialect/HW:attributes_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:dialect_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:enum_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:headers",
        "@heir//third_party/circt/include/circt/Dialect/HW:op_interfaces_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:ops_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:type_interfaces_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:types_inc_gen",
        "@heir//third_party/circt/lib/Support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
