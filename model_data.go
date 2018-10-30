package menoh

import (
	"errors"

	"github.com/pfnet-research/go-menoh/external"
)

// ModelData is a wrap of menoh.ModelData.
type ModelData struct {
	external.ModelData
}

// NewRawModelData return empty ModelData to be setup outer.
func NewRawModelData() (*ModelData, error) {
	m, err := external.MakeModelData()
	if err != nil {
		return nil, err
	}
	return &ModelData{*m}, nil
}

// NewModelDataFromPath returns ModelData using ONNX file placed on the path.
func NewModelDataFromPath(path string) (*ModelData, error) {
	m, err := external.MakeModelDataFromONNX(path)
	if err != nil {
		return nil, err
	}
	return &ModelData{*m}, nil
}

// NewModelDataFromBytes return ModelData from binary data.
func NewModelDataFromBytes(data []byte) (*ModelData, error) {
	if len(data) == 0 {
		return nil, errors.New("ONNX file data is empty")
	}
	m, err := external.MakeModelDataFromONNXBytes(data)
	if err != nil {
		return nil, err
	}
	return &ModelData{*m}, nil
}

// AddTensorAsParameter adds tensor to named parameter.
func (m *ModelData) AddTensorAsParameter(name string, param Tensor) error {
	menohDtype, err := toMenohDtype(param.dtype())
	if err != nil {
		return err
	}
	variable := external.Variable{
		Dtype:        menohDtype,
		Dims:         param.Shape(),
		BufferHandle: param.ptr(),
	}
	return m.AddParameter(name, variable)
}
