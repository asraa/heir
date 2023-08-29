load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "support_td_files",
    srcs = glob([
        "include/circt/Support/*.td",
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
    name = "support_interfaces_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-op-interface-decls",
            ],
            "include/circt/Support/InstanceGraphInterface.h.inc",
        ),
        (
            [
                "-gen-op-interface-defs",
            ],
            "include/circt/Support/InstanceGraphInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Support/InstanceGraphInterface.td",
    deps = [
        ":support_td_files",
    ],
)

cc_library(
    name = "support",
    srcs = glob([
        "lib/Support/*.cpp",
    ]),
    hdrs = glob([
        "include/circt/Support/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":support_interfaces_inc_gen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

td_library(
    name = "td_files",
    srcs = glob([
        "include/circt/Dialect/HW/*.td",
    ]),
    includes = ["include"],
    deps = [
        ":support_td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "include/circt/Dialect/HW/HWDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "include/circt/Dialect/HW/HWDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/HW/HWDialect.td",
    deps = [
        ":support_interfaces_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
            ],
            "include/circt/Dialect/HW/HWTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "include/circt/Dialect/HW/HWTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/HW/HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-op-decls",
            ],
            "include/circt/Dialect/HW/HW.h.inc",
        ),
        (
            [
                "-gen-op-defs",
            ],
            "include/circt/Dialect/HW/HW.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/HW/HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
            ],
            "include/circt/Dialect/HW/HWAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
            ],
            "include/circt/Dialect/HW/HWAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/HW/HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "op_interfaces_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-op-interface-decls",
            ],
            "include/circt/Dialect/HW/HWOpInterfaces.h.inc",
        ),
        (
            [
                "-gen-op-interface-defs",
            ],
            "include/circt/Dialect/HW/HWOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/HW/HWOpInterfaces.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "type_interfaces_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-type-interface-decls",
            ],
            "include/circt/Dialect/HW/HWTypeInterfaces.h.inc",
        ),
        (
            [
                "-gen-type-interface-defs",
            ],
            "include/circt/Dialect/HW/HWTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/HW/HWTypeInterfaces.td",
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "enum_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-enum-decls",
            ],
            "include/circt/Dialect/HW/HWEnums.h.inc",
        ),
        (
            [
                "-gen-enum-defs",
            ],
            "include/circt/Dialect/HW/HWEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/HW/HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

cc_library(
    name = "HWDialect",
    srcs = glob(
        [
            "lib/Dialect/HW/*.cpp",
        ],
        exclude = [
            "lib/Dialect/HW/HWReductions.cpp",
        ],
    ),
    hdrs = glob(["include/circt/Dialect/HW/*.h"]) + [
        "include/circt/Dialect/Comb/CombDialect.h",
        "include/circt/Dialect/Comb/CombOps.h",
        "include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enum_inc_gen",
        ":op_interfaces_inc_gen",
        ":ops_inc_gen",
        ":support",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "CombDialect",
    srcs = glob([
        "lib/Dialect/Comb/*.cpp",
    ]),
    hdrs = glob([
        "include/circt/Dialect/Comb/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":HWDialect",
        ":comb_dialect_inc_gen",
        ":comb_enum_inc_gen",
        ":comb_ops_inc_gen",
        ":comb_type_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

td_library(
    name = "comb_td_files",
    srcs = [
        "include/circt/Dialect/Comb/Comb.td",
        "include/circt/Dialect/Comb/Combinational.td",
    ],
    includes = ["include"],
    deps = [
        ":td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "comb_dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=comb",
            ],
            "include/circt/Dialect/Comb/CombDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=comb",
            ],
            "include/circt/Dialect/Comb/CombDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/Comb/Comb.td",
    deps = [
        ":comb_td_files",
    ],
)

gentbl_cc_library(
    name = "comb_ops_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-op-decls",
            ],
            "include/circt/Dialect/Comb/Comb.h.inc",
        ),
        (
            [
                "-gen-op-defs",
            ],
            "include/circt/Dialect/Comb/Comb.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/Comb/Comb.td",
    deps = [
        ":comb_td_files",
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "comb_type_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
            ],
            "include/circt/Dialect/Comb/CombTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "include/circt/Dialect/Comb/CombTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/Comb/Comb.td",
    deps = [
        ":comb_td_files",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "comb_enum_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-enum-decls",
            ],
            "include/circt/Dialect/Comb/CombEnums.h.inc",
        ),
        (
            [
                "-gen-enum-defs",
            ],
            "include/circt/Dialect/Comb/CombEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/circt/Dialect/Comb/Comb.td",
    deps = [
        ":comb_dialect_inc_gen",
        ":comb_td_files",
    ],
)

cc_library(
    name = "CombHWDialect",
    srcs = glob(
        [
            "lib/Dialect/HW/*.cpp",
            "lib/Dialect/Comb/*.cpp",
        ],
        exclude = [
            "lib/Dialect/HW/HWReductions.cpp",
        ],
    ),
    hdrs = glob([
        "include/circt/Dialect/HW/*.h",
        "include/circt/Dialect/Comb/*.h",
    ]) + ["@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h"],
    includes = ["include"],
    deps = [
        ":attributes_inc_gen",
        ":comb_dialect_inc_gen",
        ":comb_enum_inc_gen",
        ":comb_ops_inc_gen",
        ":comb_type_inc_gen",
        ":dialect_inc_gen",
        ":enum_inc_gen",
        ":op_interfaces_inc_gen",
        ":ops_inc_gen",
        ":support",
        ":type_interfaces_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
