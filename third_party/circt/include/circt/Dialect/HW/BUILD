load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    glob(["*.h"]),
)

cc_library(
    name = "headers",
    hdrs = [
        "ConversionPatterns.h",
        "CustomDirectiveImpl.h",
        "HWAttributes.h",
        "HWDialect.h",
        "HWInstanceGraph.h",
        "HWOpInterfaces.h",
        "HWOps.h",
        "HWSymCache.h",
        "HWTypeInterfaces.h",
        "HWTypes.h",
        "HWVisitors.h",
        "InnerSymbolTable.h",
        "InstanceImplementation.h",
        "ModuleImplementation.h",
        "PortConverter.h",
        "@heir//third_party/circt/include/circt/Dialect/Comb:CombDialect.h",
        "@heir//third_party/circt/include/circt/Dialect/Comb:CombOps.h",
    ],
    strip_include_prefix = "/third_party/circt/include",
    deps = [
        "@heir//third_party/circt/include/circt/Dialect/HW:attributes_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:dialect_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:enum_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:op_interfaces_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:ops_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:type_interfaces_inc_gen",
        "@heir//third_party/circt/include/circt/Dialect/HW:types_inc_gen",
        "@heir//third_party/circt/lib/Support",
    ],
)

td_library(
    name = "td_files",
    srcs = glob([
        "*.td",
    ]),
    includes = ["/third_party/circt/include"],
    deps = [
        "@heir//third_party/circt/include/circt/Support:td_files",
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
    includes = ["/third_party/circt/include"],
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
            ],
            "HWDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "HWDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HWDialect.td",
    deps = [
        ":td_files",
        "@heir//third_party/circt/include/circt/Support:interfaces_inc_gen",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    includes = ["/third_party/circt/include"],
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
            ],
            "HWTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
            ],
            "HWTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    includes = ["/third_party/circt/include"],
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-op-decls",
            ],
            "HW.h.inc",
        ),
        (
            [
                "-gen-op-defs",
            ],
            "HW.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "attributes_inc_gen",
    includes = ["/third_party/circt/include"],
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
            ],
            "HWAttributes.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
            ],
            "HWAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "op_interfaces_inc_gen",
    includes = ["/third_party/circt/include"],
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-op-interface-decls",
            ],
            "HWOpInterfaces.h.inc",
        ),
        (
            [
                "-gen-op-interface-defs",
            ],
            "HWOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HWOpInterfaces.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "type_interfaces_inc_gen",
    includes = ["/third_party/circt/include"],
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-type-interface-decls",
            ],
            "HWTypeInterfaces.h.inc",
        ),
        (
            [
                "-gen-type-interface-defs",
            ],
            "HWTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HWTypeInterfaces.td",
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "enum_inc_gen",
    includes = ["/third_party/circt/include"],
    strip_include_prefix = "/third_party/circt/include",
    tbl_outs = [
        (
            [
                "-gen-enum-decls",
            ],
            "HWEnums.h.inc",
        ),
        (
            [
                "-gen-enum-defs",
            ],
            "HWEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HW.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
