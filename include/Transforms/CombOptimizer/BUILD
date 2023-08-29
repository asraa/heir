# CombOptimizer transform tablegen and headers

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    [
        "CombOptimizer.h",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CombOptimizer",
            ],
            "CombOptimizer.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "CombOptimizer.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CombOptimizer.td",
    deps = [
        "@circt//:CombHWDialect",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
