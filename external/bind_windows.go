package external

import "C"
import (
	"errors"
	"syscall"
	"unsafe"
)

// ModelData bind. Required to delete after making, call Delete function.
type ModelData struct {
	h uintptr
}

// Delete object.
func (m *ModelData) Delete() {
	MenohDeleteModelData(m.h)
}

// MakeModelDataFromONNX returns ModelData using ONNX file path.
func MakeModelDataFromONNX(path string) (*ModelData, error) {
	var h uintptr
	if err := checkError(MenohMakeModelDataFromONNX(path, unsafe.Pointer(&h))); err != nil {
		return nil, err
	}
	return &ModelData{h: h}, nil
}

// MakeModelDataFromONNXBytes return ModelData with ONNX file byte data.
func MakeModelDataFromONNXBytes(data []byte) (*ModelData, error) {
	var h uintptr
	if err := checkError(MenohMakeModelDataFromONNXBytes(data, unsafe.Pointer(&h))); err != nil {
		return nil, err
	}
	return &ModelData{h: h}, nil
}

// MakeModelData returns empty ModelData object, to build manually.
func MakeModelData() (*ModelData, error) {
	var h uintptr
	if err := checkError(MenohMakeModelData(unsafe.Pointer(&h))); err != nil {
		return nil, err
	}
	return &ModelData{h: h}, nil
}

// AddParameter adds named parameter.
func (m *ModelData) AddParameter(name string, param Variable) error {
	return checkError(MenohModelDataAddParameter(
		m.h, name, int(param.Dtype), param.Dims, param.BufferHandle))
}

// AddNewNode adds new opType.
func (m *ModelData) AddNewNode(opType string) error {
	return checkError(MenohModelDataAddNewNode(m.h, opType))
}

// AddInputNameToCurrentNode adds input name to current node.
func (m *ModelData) AddInputNameToCurrentNode(inputName string) error {
	return checkError(MenohModelDataAddInputNameToCurrentNode(m.h, inputName))
}

// AddOutputNameToCurrentNode adds output name to current node.
func (m *ModelData) AddOutputNameToCurrentNode(outputName string) error {
	return checkError(MenohModelDataAddOutputNameToCurrentNode(m.h, outputName))
}

// AddAttributeIntToCurrentNode adds integer type attribute to current node.
func (m *ModelData) AddAttributeIntToCurrentNode(attributeName string, value int) error {
	return checkError(MenohModelDataAddAttributeIntToCurrentNode(m.h, attributeName, value))
}

// AddAttributeFloatToCurrentNode adds float type attribute to current node.
func (m *ModelData) AddAttributeFloatToCurrentNode(attributeName string, value float32) error {
	return checkError(MenohModelDataAddAttributeFloatToCurrentNode(m.h, attributeName, value))
}

// AddAttributeIntsToCurrentNode adds int array type attribute to current node.
func (m *ModelData) AddAttributeIntsToCurrentNode(attributeName string, value []int) error {
	return checkError(MenohModelDataAddAttributeIntsToCurrentNode(m.h, attributeName, value))
}

// AddAttributeFloatsToCurrentNode adds float array type attribute to current node.
func (m *ModelData) AddAttributeFloatsToCurrentNode(attributeName string, value []float32) error {
	return checkError(MenohModelDataAddAttributeFloatsToCurrentNode(m.h, attributeName, value))
}

// Optimize ModelData with profiling table.
func (m *ModelData) Optimize(table VariableProfileTable) error {
	return checkError(MenohModelDataOptimize(m.h, table.h))
}

// VariableProfileTableBuilder bind. Required to delete after making, call Delete function.
type VariableProfileTableBuilder struct {
	h uintptr
}

// Delete object.
func (b *VariableProfileTableBuilder) Delete() {
	MenohDeleteVariableProfileTableBuilder(b.h)
}

// MakeVariableProfileTableBuilder returns VariableProfileTableBuilder.
func MakeVariableProfileTableBuilder() (*VariableProfileTableBuilder, error) {
	var h uintptr
	if err := checkError(MenohMakeVariableProfileTableBuilder(unsafe.Pointer(&h))); err != nil {
		return nil, err
	}
	return &VariableProfileTableBuilder{h: h}, nil
}

// AddInputProfile adds input profile with layer name, data type and dimension size.
func (b *VariableProfileTableBuilder) AddInputProfile(name string, dtype TypeMenohDtype, dims ...int32) error {
	return checkError(MenohVariableProfileTableBuilderAddInputProfile(b.h, name, int(dtype), dims))
}

// AddOutputProfile adds output profile with layer name and data type.
func (b *VariableProfileTableBuilder) AddOutputProfile(name string, dtype TypeMenohDtype) error {
	return checkError(
		MenohVariableProfileTableBuilderAddOutputName(b.h, name))
}

// BuildVariableProfileTable returns VariableProfileTable.
func (b *VariableProfileTableBuilder) BuildVariableProfileTable(md ModelData) (
	*VariableProfileTable, error) {

	var h uintptr
	if err := checkError(MenohBuildVariableProfileTable(b.h, md.h, unsafe.Pointer(&h))); err != nil {
		return nil, err
	}
	return &VariableProfileTable{h: h}, nil
}

// VariableProfile represents profile information to make real variable.
type VariableProfile struct {
	Dtype TypeMenohDtype
	Dims  []int32
}

// VariableProfileTable bind. Required to delete after making, call Delete function.
type VariableProfileTable struct {
	h uintptr
}

// Delete object.
func (t *VariableProfileTable) Delete() {
	MenohDeleteVariableProfileTable(t.h)
}

// GetVariableProfile returns profile which setup variable information includes.
func (t *VariableProfileTable) GetVariableProfile(name string) (*VariableProfile, error) {
	var dtype int
	if err := checkError(MenohVariableProfileTableGetDtype(t.h, name, unsafe.Pointer(&dtype))); err != nil {
		return nil, err
	}
	var size int
	if err := checkError(MenohVariableProfileTableGetDimsSize(t.h, name, unsafe.Pointer(&size))); err != nil {
		return nil, err
	}
	dims := make([]int32, size)
	for i := 0; i < int(size); i++ {
		var dim int32
		if err := checkError(MenohVariableProfileTableGetDimsAt(t.h, name, i, unsafe.Pointer(&dim))); err != nil {
			return nil, err
		}
		dims[i] = dim
	}
	return &VariableProfile{
		Dtype: toDtype(dtype),
		Dims:  dims,
	}, nil
}

// ModelBuilder bind. Required to delete after making, call Delete function.
type ModelBuilder struct {
	h uintptr
}

// Delete object.
func (b *ModelBuilder) Delete() {
	MenohDeleteModelBuilder(b.h)
}

// MakeModelBuilder returns ModelBuilder.
func MakeModelBuilder(vpt VariableProfileTable) (*ModelBuilder, error) {
	var h uintptr
	if err := checkError(MenohMakeModelBuilder(vpt.h, unsafe.Pointer(&h))); err != nil {
		return nil, err
	}
	return &ModelBuilder{h: h}, nil
}

// AttachExternalBuffer attaches data buffer to get data. This process must be done
// before building Model.
func (b *ModelBuilder) AttachExternalBuffer(variableName string, bufferPtr unsafe.Pointer) error {
	return checkError(
		MenohModelBuilderAttachExternalBuffer(b.h, variableName, bufferPtr))
}

// BuildModel returns Model.
func (b *ModelBuilder) BuildModel(md ModelData, backend, backendConfig string) (*Model, error) {
	var h uintptr
	if err := checkError(MenohBuildModel(b.h, md.h, backend, backendConfig, unsafe.Pointer(&h))); err != nil {
		return nil, err
	}
	return &Model{h: h}, nil
}

// Variable represents data include data attribution and pointer.
type Variable struct {
	Dtype        TypeMenohDtype
	Dims         []int32
	BufferHandle unsafe.Pointer
}

// Model bind. Required to delete after making, call Delete function.
type Model struct {
	h uintptr
}

// Delete object.
func (m *Model) Delete() {
	MenohDeleteModel(m.h)
}

// GetVariable returns Variable, which set the target data information.
func (m *Model) GetVariable(name string) (*Variable, error) {
	var dtype int
	if err := checkError(MenohModelGetVariableDtype(m.h, name, unsafe.Pointer(&dtype))); err != nil {
		return nil, err
	}
	var size int
	if err := checkError(MenohModelGetVariableDimsSize(m.h, name, unsafe.Pointer(&size))); err != nil {
		return nil, err
	}
	dims := make([]int32, size)
	for i := 0; i < int(size); i++ {
		var dim int32
		if err := checkError(MenohModelGetVariableDimsAt(m.h, name, i, unsafe.Pointer(&dim))); err != nil {
			return nil, err
		}
		dims[i] = int32(dim)
	}
	var ptr unsafe.Pointer
	if err := checkError(MenohModelgetVariableBufferHandle(m.h, name, &ptr)); err != nil {
		return nil, err
	}

	return &Variable{
		Dtype:        toDtype(dtype),
		Dims:         dims,
		BufferHandle: ptr,
	}, nil
}

// Run calculation.
func (m *Model) Run() error {
	return checkError(MenohModelRun(m.h))
}

// TypeMenohDtype binds 'menoh_dtype_constant' enum
type TypeMenohDtype int

// Dtype
const (
	TypeFloat TypeMenohDtype = iota
)

func toDtype(typeCode int) TypeMenohDtype {
	return TypeMenohDtype(int(typeCode))
}

type typeMenohError int

// Menoh Error type
const (
	typeSuccess typeMenohError = iota
	typeSTDError
	typeUnknownError
	typeInvalidFilename
	typeUnsupportedONNXOpsetVersion
	typeONNXParseError
	typeInvalidDtype
	typeInvalidAttributeType
	typeUnsupportedOperatorAttribute
	typeDimensionMismatch
	typeVariableNotFound
	typeIndexOutOfRange
	typeJSONParseError
	typeInvalidBackendName
	typeUnsupportedOperator
	typeFailedToConfigureOperator
	typeBackendError
	typeSameNamedVariableAlreadyExist
	typeUnsupportedInputDims
	typeSameNamedParameterAlreadyExist
	typeSameNamedAttributeAlreadyExist
	typeInvalidBackendConfigError
	typeInputNotFoundError
	typeOutputNotFoundError
)

func (e typeMenohError) String() string {
	messages := []string{
		"success",
		"std error",
		"unknown error",
		"invalid filename",
		"unsupported ONNX opset version",
		"ONNX parse error",
		"invalid dtype",
		"invalid attribute type",
		"unsupported operator attribute",
		"dimension mismatch",
		"variable not found",
		"index out of range",
		"JSON parse error",
		"invalid backend name",
		"unsupported operator",
		"failed to configure operator",
		"backend error",
		"same named variable already exist",
		"unsupported input dims",
		"same named parameter already exist",
		"same named attribute already exist",
		"invalid backend config",
		"input not found",
		"output not found",
	}

	if e < typeSuccess || e > typeOutputNotFoundError {
		return "unknown type error"
	}
	return messages[e]
}

func checkError(err error) error {
	if err == nil {
		return nil
	}
	if err == syscall.EINVAL {
		return err
	}
	errType := typeMenohError(uintptr(err.(syscall.Errno)))
	if errType == typeSuccess {
		return nil
	}
	// MenohGetLastErrorMessage returns uintptr, not convert to string correctly
	// So not get last message and only return error type
	return errors.New(errType.String())
}
