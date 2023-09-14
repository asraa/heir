load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    glob([
        "*.h",
    ]),
)

td_library(
    name = "td_files",
    srcs = glob([
        "*.td",
    ]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "interfaces_inc_gen",
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-op-interface-decls",
            ],
            "InstanceGraphInterface.h.inc",
        ),
        (
            [
                "-gen-op-interface-defs",
            ],
            "InstanceGraphInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "InstanceGraphInterface.td",
    deps = [
        ":td_files",
    ],
)
