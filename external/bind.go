package external

/*
#cgo LDFLAGS: -lmenoh

#include <stdlib.h>
#include <menoh/menoh.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type ModelData struct {
	h C.menoh_model_data_handle
}

func (m *ModelData) Delete() {
	C.menoh_delete_model_data(m.h)
	m.h = nil
}

func MakeModelDataFromONNX(path string) (*ModelData, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	var h C.menoh_model_data_handle
	if err := checkError(C.menoh_make_model_data_from_onnx(cPath, &h)); err != nil {
		return nil, err
	}
	return &ModelData{h: h}, nil
}

func (m *ModelData) Optimize(table VariableProfileTable) error {
	return checkError(C.menoh_model_data_optimize(m.h, table.h))
}

type VariableProfileTableBuilder struct {
	h C.menoh_variable_profile_table_builder_handle
}

func (b *VariableProfileTableBuilder) Delete() {
	C.menoh_delete_variable_profile_table_builder(b.h)
	b.h = nil
}

func MakeVariableProfileTableBuilder() (*VariableProfileTableBuilder, error) {
	var h C.menoh_variable_profile_table_builder_handle
	if err := checkError(C.menoh_make_variable_profile_table_builder(&h)); err != nil {
		return nil, err
	}
	return &VariableProfileTableBuilder{h: h}, nil
}

func (b *VariableProfileTableBuilder) AddInputProfile(name string, dtype TypeMenohDtype, dims ...int32) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	switch dimLen := len(dims); dimLen {
	case 2:
		return checkError(
			C.menoh_variable_profile_table_builder_add_input_profile_dims_2(
				b.h, cName, C.int(dtype), C.int(dims[0]), C.int(dims[1])))
	case 4:
		return checkError(
			C.menoh_variable_profile_table_builder_add_input_profile_dims_4(
				b.h, cName, C.int(dtype), C.int(dims[0]), C.int(dims[1]), C.int(dims[2]), C.int(dims[3])))
	default:
		return fmt.Errorf("dimension size %d is not supported", dimLen)
	}
}

func (b *VariableProfileTableBuilder) AddOutputProfile(name string, dtype TypeMenohDtype) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkError(
		C.menoh_variable_profile_table_builder_add_output_profile(
			b.h, cName, C.int(dtype)))
}

func (b *VariableProfileTableBuilder) BuildVariableProfileTable(md ModelData) (
	*VariableProfileTable, error) {

	var h C.menoh_variable_profile_table_handle
	if err := checkError(C.menoh_build_variable_profile_table(b.h, md.h, &h)); err != nil {
		return nil, err
	}
	return &VariableProfileTable{h: h}, nil
}

type VariableProfile struct {
	Dtype TypeMenohDtype
	Dims  []int32
}

type VariableProfileTable struct {
	h C.menoh_variable_profile_table_handle
}

func (t *VariableProfileTable) Delete() {
	C.menoh_delete_variable_profile_table(t.h)
	t.h = nil
}

func (t *VariableProfileTable) GetVariableProfile(name string) (*VariableProfile, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var dtype C.int
	if err := checkError(C.menoh_variable_profile_table_get_dtype(t.h, cName, &dtype)); err != nil {
		return nil, err
	}
	var size C.int
	if err := checkError(C.menoh_variable_profile_table_get_dims_size(t.h, cName, &size)); err != nil {
		return nil, err
	}
	dims := make([]int32, size)
	for i := 0; i < int(size); i++ {
		var dim C.int
		if err := checkError(C.menoh_variable_profile_table_get_dims_at(t.h, cName, C.int(i), &dim)); err != nil {
			return nil, err
		}
		dims[i] = int32(dim)
	}
	return &VariableProfile{
		Dtype: toDtype(dtype),
		Dims:  dims,
	}, nil
}

type ModelBuilder struct {
	h C.menoh_model_builder_handle
}

func (b *ModelBuilder) Delete() {
	C.menoh_delete_model_builder(b.h)
	b.h = nil
}

func MakeModelBuilder(vpt VariableProfileTable) (*ModelBuilder, error) {
	var h C.menoh_model_builder_handle
	if err := checkError(C.menoh_make_model_builder(vpt.h, &h)); err != nil {
		return nil, err
	}
	return &ModelBuilder{h: h}, nil
}

func (b *ModelBuilder) AttachExternalBuffer(variableName string, bufferPtr unsafe.Pointer) error {
	cVariableName := C.CString(variableName)
	defer C.free(unsafe.Pointer(cVariableName))
	return checkError(
		C.menoh_model_builder_attach_external_buffer(b.h, cVariableName, bufferPtr))
}

func (b *ModelBuilder) BuildModel(md ModelData, backend, backendConfig string) (*Model, error) {
	cBackend := C.CString(backend)
	defer C.free(unsafe.Pointer(cBackend))
	cBackendConfig := C.CString(backendConfig)
	defer C.free(unsafe.Pointer(cBackendConfig))
	var h C.menoh_model_handle
	if err := checkError(C.menoh_build_model(b.h, md.h, cBackend, cBackendConfig, &h)); err != nil {
		return nil, err
	}
	return &Model{h: h}, nil
}

type Variable struct {
	Dtype  TypeMenohDtype
	Dims   []int32
	BufferHandle unsafe.Pointer
}

type Model struct {
	h C.menoh_model_handle
}

func (m *Model) Delete() {
	C.menoh_delete_model(m.h)
	m.h = nil
}

func (m *Model) GetVariable(name string) (*Variable, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var dtype C.int
	e := C.menoh_model_get_variable_dtype(m.h, cName, &dtype)
	if err := checkError(e); err != nil {
		return nil, err
	}
	var size C.int
	e = C.menoh_model_get_variable_dims_size(m.h, cName, &size)
	if err := checkError(e); err != nil {
		return nil, err
	}
	dims := make([]int32, size)
	for i := 0; i < int(size); i++ {
		var dim C.int
		e := C.menoh_model_get_variable_dims_at(m.h, cName, C.int(i), &dim)
		if err := checkError(e); err != nil {
			return nil, err
		}
		dims[i] = int32(dim)
	}
	var ptr unsafe.Pointer
	e = C.menoh_model_get_variable_buffer_handle(m.h, cName, &ptr)
	if err := checkError(e); err != nil {
		return nil, err
	}

	return &Variable{
		Dtype:  toDtype(dtype),
		Dims:   dims,
		BufferHandle: ptr,
	}, nil
}

func (m *Model) Run() error {
	return checkError(C.menoh_model_run(m.h))
}

// TypeMenohDtype binds 'menoh_dtype_constant' enum
type TypeMenohDtype C.menoh_dtype

// Dtype
const (
	TypeFloat TypeMenohDtype = iota
)

func toDtype(typeCode C.int) TypeMenohDtype {
	return TypeMenohDtype(int(typeCode))
}

type typeMenohError C.menoh_error_code

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
	typeSameNameVariableAlreadyExist
)

func (e typeMenohError) String() string {
	switch e {
	case typeSuccess:
		return "success"
	case typeSTDError:
		return "std error"
	case typeUnknownError:
		return "unknown error"
	case typeInvalidFilename:
		return "invalid filename"
	case typeUnsupportedONNXOpsetVersion:
		return "unsupported ONNX opset version"
	case typeONNXParseError:
		return "ONNX parse error"
	case typeInvalidDtype:
		return "invalid dtype"
	case typeInvalidAttributeType:
		return "invalid attribute type"
	case typeUnsupportedOperatorAttribute:
		return "unsupported operator attribute"
	case typeDimensionMismatch:
		return "dimension mismatch"
	case typeVariableNotFound:
		return "variable not found"
	case typeIndexOutOfRange:
		return "index out of range"
	case typeJSONParseError:
		return "JSON parse error"
	case typeInvalidBackendName:
		return "invalid backend name"
	case typeUnsupportedOperator:
		return "unsupported operator"
	case typeFailedToConfigureOperator:
		return "failed to configure operator"
	case typeBackendError:
		return "backend error"
	case typeSameNameVariableAlreadyExist:
		return "save name variable already exist"
	default:
		return "unknown type error"
	}
}

func checkError(errCode C.int) error {
	errType := typeMenohError(int(errCode))
	if errType == typeSuccess {
		return nil
	}
	lastMessage := C.GoString(C.menoh_get_last_error_message())
	if lastMessage == "" {
		return errors.New(errType.String())
	}
	// last message includes error type
	return errors.New(lastMessage)
}
