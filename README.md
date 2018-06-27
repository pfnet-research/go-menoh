# Menoh Go

Golang binding for [Menoh](https://github.com/pfnet-research/menoh)

## Requirements

- OS
  - Linux
  - Mac
  - ~~Windows~~ (TBD)
- Go 1.10+
- [Menoh](https://github.com/pfnet-research/menoh) 1.0.1+

## Install

After install Menoh, then

```bash
$ go get -u github.com/pfnet-research/menoh-go
```

## Running VGG16 example

### Requirements

- [disintegration/imaging](https://github.com/disintegration/imaging)

Setup example data, see [gen_test_data.py](https://github.com/pfnet-research/menoh#run-test)

```
data/
  |- Light_sussex_hen.jpg
  |- synset_words.txt
  |- VGG16.onnx
```

```bash
$ cd example/vgg16
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