# Menoh Go

Golang binding for [Menoh](https://github.com/pfnet-research/menoh)

## Requirements

- OS
  - Linux
  - Mac
  - Windows
- Go 1.10+
- [Menoh](https://github.com/pfnet-research/menoh) 1.0.1+

## Install

After install Menoh, then

```bash
$ go get -u github.com/pfnet-research/menoh-go
```

### Windows

Add a path to DLLs distributed by Menoh to local Path environment.

```
\path\to\menoh\bin
  |- libiomp5md.dll
  |- menoh.dll
  |- mkldnn.dll
  |- mklml.dll
```

```
set PATH=\path\to\menoh\bin;%PATH%
```


## Usage

See [example/vgg16/tutorial](example/vgg16/README.md)

## License

MIT License (see [LICENSE](/LICENSE) file).
