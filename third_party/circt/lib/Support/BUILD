package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Support",
    srcs = [
        "CustomDirectiveImpl.cpp",
        "InstanceGraph.cpp",
        "ValueMapper.cpp",
    ],
    hdrs = [
        "@heir//third_party/circt/include/circt/Support:BackedgeBuilder.h",
        "@heir//third_party/circt/include/circt/Support:BuilderUtils.h",
        "@heir//third_party/circt/include/circt/Support:CustomDirectiveImpl.h",
        "@heir//third_party/circt/include/circt/Support:InstanceGraph.h",
        "@heir//third_party/circt/include/circt/Support:InstanceGraphInterface.h",
        "@heir//third_party/circt/include/circt/Support:LLVM.h",
        "@heir//third_party/circt/include/circt/Support:Namespace.h",
        "@heir//third_party/circt/include/circt/Support:ParsingUtils.h",
        "@heir//third_party/circt/include/circt/Support:SymCache.h",
        "@heir//third_party/circt/include/circt/Support:ValueMapper.h",
    ],
    copts = [
        "-Ithird_party/circt/include",
    ],
    strip_include_prefix = "/third_party/circt/include",
    deps = [
        "@heir//third_party/circt/include/circt/Support:interfaces_inc_gen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
