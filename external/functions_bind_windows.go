package external

//go:generate go run $GOROOT/src/syscall/mksyscall_windows.go -output functions_windows.go functions_bind_windows.go

//sys MenohDeleteModelData(mdHandle uintptr) = menoh.menoh_delete_model_data
//sys MenohMakeModelDataFromONNX(path string, mdHandle unsafe.Pointer) (err error) = menoh.menoh_make_model_data_from_onnx
//sys MenohModelDataOptimize(mdHandle uintptr, vptHandle uintptr) (err error) = menoh.menoh_model_data_optimize
//sys MenohDeleteVariableProfileTableBuilder(vptHandle uintptr) = menoh.menoh_delete_variable_profile_table_builder
//sys MenohMakeVariableProfileTableBuilder(vptbHandle unsafe.Pointer) (err error) = menoh.menoh_make_variable_profile_table_builder
//sys MenohVariableProfileTableBuilderAddInputProfileDims2(vptbHandle uintptr, name string, dtype int, dim1 int32, dim2 int32) (err error) = menoh.menoh_variable_profile_table_builder_add_input_profile_dims_2
//sys MenohVariableProfileTableBuilderAddInputProfileDims4(vptbHandle uintptr, name string, dtype int, dim1 int32, dim2 int32, dim3 int32, dim4 int32) (err error) = menoh.menoh_variable_profile_table_builder_add_input_profile_dims_4
//sys MenohVariableProfileTableBuilderAddoutputProfile(vptbHandle uintptr, name string, dtype int) (err error) = menoh.menoh_variable_profile_table_builder_add_output_profile
//sys MenohBuildVariableProfileTable(vptbHandle uintptr, mdHandle uintptr, vptHandle unsafe.Pointer) (err error) = menoh.menoh_build_variable_profile_table
//sys MenohDeleteVariableProfileTable(vptHandle uintptr) = menoh.menoh_delete_variable_profile_table
//sys MenohVariableProfileTableGetDtype(vptHandle uintptr, name string, dtypeHandle unsafe.Pointer) (err error) = menoh.menoh_variable_profile_table_get_dtype
//sys MenohVariableProfileTableGetDimsSize(vptHandle uintptr, name string, sizeHandle unsafe.Pointer) (err error) = menoh.menoh_variable_profile_table_get_dims_size
//sys MenohVariableProfileTableGetDimsAt(vptHandle uintptr, name string, pos int, dimHandle unsafe.Pointer) (err error) = menoh.menoh_variable_profile_table_get_dims_at
//sys MenohDeleteModelBuilder(mbHandle uintptr) = menoh.menoh_delete_model_builder
//sys MenohMakeModelBuilder(vptHandle uintptr, mbHandle unsafe.Pointer) (err error) = menoh.menoh_make_model_builder
//sys MenohModelBuilderAttachExternalBuffer(mbHandle uintptr, name string, buffer unsafe.Pointer) (err error) = menoh.menoh_model_builder_attach_external_buffer
//sys MenohBuildModel(mbHandle uintptr, mdHandle uintptr, backend string, backendConfig string, modelHandle unsafe.Pointer) (err error) = menoh.menoh_build_model
//sys MenohDeleteModel(mHandle uintptr) = menoh.menoh_delete_model
//sys MenohModelGetVariableDtype(mHandle uintptr, name string, dtypeHandle unsafe.Pointer) (err error) = menoh.menoh_model_get_variable_dtype
//sys MenohModelGetVariableDimsSize(mHandle uintptr, name string, sizeHandle unsafe.Pointer) (err error) = menoh.menoh_model_get_variable_dims_size
//sys MenohModelGetVariableDimsAt(mHandle uintptr, name string, pos int, dimHandle unsafe.Pointer) (err error) = menoh.menoh_model_get_variable_dims_at
//sys MenohModelgetVariableBufferHandle(mHandle uintptr, name string, buffer *unsafe.Pointer) (err error) = menoh.menoh_model_get_variable_buffer_handle
//sys MenohModelRun(mHandle uintptr) (err error) = menoh.menoh_model_run
//sys MenohGetLastErrorMessage() (msg string) = menoh.menoh_get_last_error_message
