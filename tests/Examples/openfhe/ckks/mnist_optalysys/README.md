# MNIST Optalysys

Generate the openfhe dialect IR with

```
bazel run -c opt @heir//tools:heir-opt -- --torch-linalg-to-ckks="scaling-mod-bits=28 first-mod-bits=28" $(pwd)/tests/Examples/common/mnist.mlir --scheme-to-openfhe -o /tmp/mnist.openfhe.mlir
```

Translate to C++ header and source:

```bash
$ bazel run -c opt @//heir/tools:heir-translate -- --emit-openfhe-pke-header /tmp/mnist.openfhe.mlir > $(pwd)/tests/Examples/openfhe/ckks/mnist_optalysys/mnist.h
$ bazel run -c opt @//heir/tools:heir-translate -- --emit-openfhe-pke/tmp/mnist.openfhe.mlir > $(pwd)/tests/Examples/openfhe/ckks/mnist_optalysys/mnist.cpp
```

This generates OpenFHE C++ code with 28-bit moduli and a 28-bit scaling parameter.