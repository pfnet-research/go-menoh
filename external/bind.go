package gomenoh

/*
#cgo LDFLAGS: -lmenoh
#include <menoh/menoh.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const (
	_                             = iota
	Success                       = C.menoh_error_code_success
	StdError                      = C.menoh_error_code_std_error
	UnkownError                   = C.menoh_error_code_unknown_error
	InvalidFilename               = C.menoh_error_code_invalid_filename
	UnsupportedOnnxOpsetVersion   = C.menoh_error_code_unsupported_onnx_opset_version
	OnnxParseError                = C.menoh_error_code_onnx_parse_error
	InvalidDtype                  = C.menoh_error_code_invalid_dtype
	InvalidAttributeType          = C.menoh_error_code_invalid_attribute_type
	UnsupportedOperatorAttribute  = C.menoh_error_code_unsupported_operator_attribute
	DimensionMismatch             = C.menoh_error_code_dimension_mismatch
	VariableNotFound              = C.menoh_error_code_variable_not_found
	IndexOutOfRange               = C.menoh_error_code_index_out_of_range
	UnsupportedOperator           = C.menoh_error_code_unsupported_operator
	FailedToConfigureOperator     = C.menoh_error_code_failed_to_configure_operator
	BackendError                  = C.menoh_error_code_backend_error
	SameNamedVariableAlreadyExist = C.menoh_error_code_same_named_variable_already_exist
	InvalidDimsSize               = iota
)

type dType C.menoh_dtype

const (
	_           = iota
	Float dType = C.menoh_dtype_float
)

func errorCheck(code C.menoh_error_code) error {
	if code != C.menoh_error_code_success {
		return fmt.Errorf("%s", C.GoString(C.menoh_get_last_error_message()))
	}
	return nil
}

type ModelData struct {
	h C.menoh_model_data_handle
}

func MakeModelDataFromOnnx(onnxFileName string) (*ModelData, error) {
	var h C.menoh_model_data_handle
	if err := errorCheck(C.menoh_make_model_data_from_onnx(C.CString(onnxFileName), &h)); err != nil {
		return nil, err
	}
	return &ModelData{h: h}, nil
}

func (m *ModelData) Delete() {
	C.menoh_delete_model_data(m.h)
}

func (m *ModelData) Optimiza(vpt VariableProfileTable) error {
	return errorCheck(C.menoh_model_data_optimize(m.h, vpt.h))
}

type VariableProfileTableBuilder struct {
	h C.menoh_variable_profile_table_builder_handle
}

func MakeVariableProfileTableBuilder() (*VariableProfileTableBuilder, error) {
	var h C.menoh_variable_profile_table_builder_handle
	if err := errorCheck(C.menoh_make_variable_profile_table_builder(&h)); err != nil {
		return nil, err
	}
	return &VariableProfileTableBuilder{h: h}, nil
}

func (v *VariableProfileTableBuilder) Delete() {
	C.menoh_delete_variable_profile_table_builder(v.h)
}

func (v *VariableProfileTableBuilder) AddInputProfile(variableName string, dtype dType, dims []int32) error {
	name := C.CString(variableName)
	d := C.menoh_dtype(dtype)
	var err error
	switch len(dims) {
	case 2:
		err = errorCheck(C.menoh_variable_profile_table_builder_add_input_profile_dims_2(v.h, name, d, C.int32_t(dims[0]), C.int32_t(dims[1])))
	case 4:
		err = errorCheck(C.menoh_variable_profile_table_builder_add_input_profile_dims_4(v.h, name, d, C.int32_t(dims[0]), C.int32_t(dims[1]), C.int32_t(dims[2]), C.int32_t(dims[3])))
	default:
		err = fmt.Errorf("menoh invalid dims size error (2 or 4 is valid): dims size of %s is specified %d", name, len(dims))
	}
	return err
}

func (v *VariableProfileTableBuilder) AddOutputProfile(name string, dtype dType) error {
	return errorCheck(C.menoh_variable_profile_table_builder_add_output_profile(v.h, C.CString(name), C.menoh_dtype(dtype)))
}

func (v *VariableProfileTableBuilder) BuildVariableProfileTable(modelData ModelData) (*VariableProfileTable, error) {
	var h C.menoh_variable_profile_table_handle
	if err := errorCheck(C.menoh_build_variable_profile_table(v.h, modelData.h, &h)); err != nil {
		return nil, err
	}
	return &VariableProfileTable{h: h}, nil
}

type VariableProfile struct {
	Dtype dType
	Dims  []int32
}

type VariableProfileTable struct {
	h C.menoh_variable_profile_table_handle
}

func (v *VariableProfileTable) GetVariableProfile(variableName string) (*VariableProfile, error) {
	name := C.CString(variableName)
	var dtype C.menoh_dtype
	if err := errorCheck(C.menoh_variable_profile_table_get_dtype(v.h, name, &dtype)); err != nil {
		return nil, err
	}
	var dimsSize C.int32_t
	if err := errorCheck(C.menoh_variable_profile_table_get_dims_size(v.h, name, &dimsSize)); err != nil {
		return nil, err
	}
	dims := make([]int32, dimsSize)
	for i := C.int32_t(0); i < dimsSize; i++ {
		var dim C.int32_t
		if err := errorCheck(C.menoh_variable_profile_table_get_dims_at(v.h, name, i, &dim)); err != nil {
			return nil, err
		}
		dims[i] = int32(dim)
	}
	return &VariableProfile{Dtype: dType(dtype), Dims: dims}, nil
}

type ModelBuilder struct {
	h C.menoh_model_builder_handle
}

func MakeModelBuilder(v VariableProfileTable) (*ModelBuilder, error) {
	var h C.menoh_model_builder_handle
	if err := errorCheck(C.menoh_make_model_builder(v.h, &h)); err != nil {
		return nil, err
	}
	return &ModelBuilder{h: h}, nil
}

func (m *ModelBuilder) Delete() {
	C.menoh_delete_model_builder(m.h)
}

func (m *ModelBuilder) AttachExternalBuffer(name string, buffer unsafe.Pointer) error {
	return errorCheck(C.menoh_model_builder_attach_external_buffer(m.h, C.CString(name), buffer))
}

func (m *ModelBuilder) BuildModel(d ModelData, backendName, backendConfig string) (*Model, error) {
	var h C.menoh_model_handle
	if err := errorCheck(C.menoh_build_model(m.h, d.h, C.CString(backendName), C.CString(backendConfig), &h)); err != nil {
		return nil, err
	}
	return &Model{h: h}, nil
}

type Model struct {
	h C.menoh_model_handle
}

type Variable struct {
	Dtype        dType
	Dims         []int32
	BufferHandle unsafe.Pointer
}

func (m *Model) Delete() {
	C.menoh_delete_model(m.h)
}

func (m *Model) GetVariable(variableName string) (*Variable, error) {
	name := C.CString(variableName)
	var buff unsafe.Pointer
	if err := errorCheck(C.menoh_model_get_variable_buffer_handle(m.h, name, &buff)); err != nil {
		return nil, err
	}

	var dtype C.menoh_dtype
	if err := errorCheck(C.menoh_model_get_variable_dtype(m.h, name, &dtype)); err != nil {
		return nil, err
	}

	var dimsSize C.int32_t
	if err := errorCheck(C.menoh_model_get_variable_dims_size(m.h, name, &dimsSize)); err != nil {
		return nil, err
	}

	dims := make([]int32, dimsSize)
	for i := C.int32_t(0); i < dimsSize; i++ {
		var dim C.int32_t
		if err := errorCheck(C.menoh_model_get_variable_dims_at(m.h, name, i, &dim)); err != nil {
			return nil, err
		}
		dims[i] = int32(dim)
	}

	return &Variable{Dtype: dType(dtype), Dims: dims, BufferHandle: buff}, nil
}

func (m *Model) Run() error {
	return errorCheck(C.menoh_model_run(m.h))
}
