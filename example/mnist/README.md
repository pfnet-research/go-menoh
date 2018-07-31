# MNIST Example

This example follows [ONNX/models/MNIST](https://github.com/onnx/models/tree/master/mnist)

## Prepare

Download dataset and extract.

```bash
$ wget https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz -P data
$ tar xfz data/mnist.tar.gz -C data
```

```
examples/
  |- mnist
      |- main.go
      |- README.md (this file)
      |- data/
          |- mnist
              |- test_data_set_0/
              |- test_data_set_1/
              |- test_data_set_2/
              |- model.onnx
              |- ...
```

### Replace model

`model.onnx` uses `auto_pad` attribute ([Issue#85](https://github.com/onnx/models/issues/85)), which is deprecated. Menoh does not support the attribute. Until resolved the issue, this example uses other ONNX model.

```bash
$ wget https://github.com/pfnet-research/menoh-haskell/raw/master/data/mnist.onnx -P data/mnist
```

## Run with test dataset

```
$ go run main.go
```

Return nothing when no error.

**Option**

- `data`: path of dataset. set `data/mnist` on default.
- `model`: path of ONNX model. set `data` + "/mnist.onnx" on default.
  - when fixing [Issue#85](https://github.com/onnx/models/issues/85), will set `data` + "/model.onnx" on default.
- `mode`: running mode, choose from "test" or "server", set "test" on default.
  - "test" mode runs with `test_data_set_*`, compares results
  - current only supports "test" mode.
