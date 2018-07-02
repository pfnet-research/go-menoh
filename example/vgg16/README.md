# Tutorial

This tutorial shows making a basic application to load VGG16 model, input an image and output a classification result.

## Prepare

### Requirements

- [disintegration/imaging](https://github.com/disintegration/imaging)

### ONNX model and dataset

Setup example data, see [gen_test_data.py](https://github.com/pfnet-research/menoh#run-test)

```
data/
  |- Light_sussex_hen.jpg
  |- synset_words.txt
  |- VGG16.onnx
examples/
  |- vgg16
      |- main.go
      |- README.md (this tutorial)
```

## Setup in/out configuration

Make configuration (`menoh.Config`) to set model at first. The configuration has the below information. Menoh model runner (`menoh.Runner`), as described later, uses the configuration.

- backend type, like "MKL-DNN"
- input(s) information
  - node name
  - dimension size
- output(s) information
  - node name

VGG16 (VGG16.onnx) has one input and one output, input expects 1x3x224x224 dimension data. Brief information of VGG16.onnx is here. To see more detail, use [tool/menoh_onnx_viewer](https://github.com/pfnet-research/menoh/blob/master/docs/tutorial.md#setup-model):

- Input node
  - `0:Conv`
    - `input0: 140326425860192`
- Output node
  - `39:Softmax`
    - `output0: 140326200803680`

As code, `menoh.InputConfig`, `menoh.OutputConfig`:

```go
import "github.com/pfnet-research/menoh-go"
```

```go
input0Conv := menoh.InputConfig{
	Name: "140326425860192",
	Dtype: menoh.TypeFloat,
	Dims: []int32{1, 3, 224, 224},
}

output39Softmax := menoh.OutputConfig{
	Name: "140326200803680",
	Dtype: menoh.TypeFloat,
	FromInternal: false,
}
```

In additional, to show feature vectors, setup `menoh.OutputConfig` as same.

- Node
  - `32:FC`
    - `output0: 140326200777584`

```go
output32FC := menoh.OutputConfig{
	Name: "140326200777584",
	Dtype: menoh.TypeFloat,
	FromInternal: true, // mark to output as feature vector
}
```

Make `menoh.Config` object:

```go
vgg16Config := menoh.Config{
	ONNXModelPath: "../../data/VGG16.onnx",
	Backend:       menoh.TypeMKLDNN,
	BackendConfig: "",
	Inputs:        []menoh.InputConfig{input0Conv},
	Outputs:       []menoh.OutputConfig{output32FC, output39Softmax},
}
```

## Setup model

To setup ONNX model, use `menoh.NewRunner`. This `Runner` has Menoh model object setup by profiling information and manage life cycle. Once making a runner, the application can use it constantly unless the configuration is changed, don't have to make a runner every time.

```go
runner, _ := menoh.NewRunner(vgg16Config)
defer runner.Stop()
```

## Input image

VGG16.onnx requires a pre-precessed image to input:

1. Crop and resize to 224x224 size
1. Reorder the dimension as batch size, color channel, height, width.
1. Convert pixel as float of 0-255 range.
1. Reorder color channel as "BGR" (OpenCV default style, not "RGB")

And `menoh.Runner` requires `menoh.Tensor` type on input, which represents a matrix to pass between go code and Menoh library. The blow go code shows loading the image, crop/resize the image size, make `menoh.Tensor` type.

```go
import (
	"os"
	"image"
)
```

```go
// load image
imageFile, _ := os.Open("../../data/Light_sussex_hen.jpg")
defer imageFile.Close()
img, _, _ := image.Decode(imageFile)
// crop/resize to 224x224
resizedImg := cropAndResize(img, width, height)
// get []float32{...} array to make Tensor
oneHotFloats := toOneHotFloats(resizedImg, channel)
// make Tensor to input the runner
resizedImgTensor := &menoh.FloatTensor{
	Dtype: menoh.TypeFloat,
	Dims:  []int32{1, 3, 224, 224},
	Array: oneHotFloats,
}
```

Input the pre-precessed image to the runner.

```go
runner.RunWithTensor("140326425860192", resizedImgTensor)
```

or

```go
runner.Run(
	map[string]menoh.Tensor{
		"140326425860192": resizedImgTensor,
	})
```

## Get output

The runner has already setup 2 output variables, `39:Softmax` and `32:FC`. Calling `GetOutput()` with the target name then the runner returns result variable as `menoh.Tensor` type.

```go
fc6OutTensor, _ := runner.GetOutput("140326200777584")
softmaxOutTensor, _ := runner.GetOutput("140326200803680")
```

To input another image, the runner returns variables calculated by the next image.

```go
// prepare next image
nextImage := &menoh.FloatTensor{...}
// run with the next image
runner.RunWithTensor("140326425860192", nextImage)
// get another result
fc6OutNext, _ := runner.GetOutput("140326200777584")
softmaxOutNext, _ := runner.GetOutput("140326200803680")
```

## Running VGG16 example

```bash
$ go run main.go
vgg16 example
-18.8019 -33.2770 -10.3634 23.3145 -2.2429 -7.4052 -25.6390 -17.8969 -8.7609 15.1024
8 0.93620 n01514859 hen
7 0.06000 n01514668 cock
86 0.00239 n01807496 partridge
82 0.00045 n01797886 ruffed grouse, partridge, Bonasa umbellus
97 0.00010 n01847000 drake
$
```
