package onnx

import (
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"math"

	"github.com/golang/protobuf/proto"
	"github.com/pfnet-research/go-menoh"
)

// LoadONNXTensorFromFile returns ONNX's Tensor instance loaded from the path.
func LoadONNXTensorFromFile(path string) (*TensorProto, error) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot load '%s', %v", path, err)
	}
	tensor := &TensorProto{}
	if err := proto.Unmarshal(b, tensor); err != nil {
		return nil, fmt.Errorf("cannot convert to ONNX tensor, %v", err)
	}
	return tensor, nil
}

// ConvertToMenohTensor converts from ONNXS's tensor to Menoh's tensor.
func ConvertToMenohTensor(t *TensorProto) (menoh.Tensor, error) {
	switch dtype := t.GetDataType(); dtype {
	case TensorProto_FLOAT:
		floats := t.GetFloatData()
		if len(floats) == 0 {
			raw := t.GetRawData()
			if len(raw) != 0 {
				floats = convertToFloat32Array(raw)
			}
		}
		dims := t.GetDims()
		dimsInt32 := make([]int32, len(dims))
		for i := 0; i < len(dims); i++ {
			dimsInt32[i] = int32(dims[i])
		}
		return &menoh.FloatTensor{
			Dims:  dimsInt32,
			Array: floats,
		}, nil
	default:
		return nil, fmt.Errorf("type %s is not supported", dtype)
	}
}

func convertToFloat32Array(raw []byte) []float32 {
	bitLength := 4
	length := len(raw) / bitLength
	floats := make([]float32, length)
	for i := 0; i < length; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*bitLength : (i+1)*bitLength])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}
