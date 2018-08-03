/*
Package menoh provides a wrapper "runner" for Menoh library to execute ONNX
model. It consits the following structures.

Runner is a wrapper to manage an ONNX model and run with input variables.

Config is information about ONNX model, required for making a runner.

Tensor represents number array for in/out variable, similar to ONNX's Tensor.
*/
package menoh

import (
	"errors"
	"fmt"

	"github.com/pfnet-research/go-menoh/external"
)

// Runner setups Menoh model with profiling and executes with input variables.
// A runner supports to call Run (or RunWithTensor) method repeatedly until
// stopping.
type Runner struct {
	modelData    *external.ModelData
	vptBuilder   *external.VariableProfileTableBuilder
	vpTable      *external.VariableProfileTable
	modelBuilder *external.ModelBuilder
	model        *external.Model

	conf Config

	inputs  map[string]Tensor
	outputs map[string]Tensor
}

// NewRunner returns Runner using configuration, the runner setup Menoh model
// and ready for execution. Require to call Stop function after the process is done.
func NewRunner(conf Config) (runner *Runner, err error) {
	runner = &Runner{
		conf:    conf,
		inputs:  map[string]Tensor{},
		outputs: map[string]Tensor{},
	}
	defer func() {
		if err != nil {
			runner.Stop()
			runner = nil
		}
	}()

	modelData, err := external.MakeModelDataFromONNX(conf.ONNXModelPath)
	if err != nil {
		return
	}
	runner.modelData = modelData

	vptBuilder, err := external.MakeVariableProfileTableBuilder()
	if err != nil {
		return
	}
	runner.vptBuilder = vptBuilder
	for _, c := range conf.Inputs {
		menohDtype, _ := toMenohDtype(c.Dtype)
		if err = vptBuilder.AddInputProfile(c.Name, menohDtype, c.Dims...); err != nil {
			return
		}
	}
	for _, c := range conf.Outputs {
		menohDtype, _ := toMenohDtype(c.Dtype)
		if err = vptBuilder.AddOutputProfile(c.Name, menohDtype); err != nil {
			return
		}
	}

	vpt, err := vptBuilder.BuildVariableProfileTable(*modelData)
	if err != nil {
		return
	}
	runner.vpTable = vpt
	for _, c := range conf.Outputs {
		if !c.FromInternal {
			continue
		}
		vp, lerr := vpt.GetVariableProfile(c.Name)
		if lerr != nil {
			err = lerr
			return
		}
		dtype, _ := toDtype(vp.Dtype)
		tensor := newTensorHandle(dtype, vp.Dims...)
		runner.outputs[c.Name] = tensor
	}

	modelBuilder, err := external.MakeModelBuilder(*vpt)
	if err != nil {
		return
	}
	runner.modelBuilder = modelBuilder
	for _, c := range conf.Inputs {
		tensor := newTensorHandle(c.Dtype, c.Dims...)
		if err = modelBuilder.AttachExternalBuffer(c.Name, tensor.ptr()); err != nil {
			return
		}
		runner.inputs[c.Name] = tensor
	}
	for _, c := range conf.Outputs {
		if !c.FromInternal {
			continue
		}
		tensor := runner.outputs[c.Name]
		if err = modelBuilder.AttachExternalBuffer(c.Name, tensor.ptr()); err != nil {
			return
		}
	}

	model, err := modelBuilder.BuildModel(*modelData, conf.Backend.String(), conf.BackendConfig)
	if err != nil {
		return
	}
	runner.model = model
	for _, c := range conf.Outputs {
		if c.FromInternal {
			continue
		}
		out, lerr := model.GetVariable(c.Name)
		if lerr != nil {
			err = lerr
			return
		}
		dtype, _ := toDtype(out.Dtype)
		tensor := newTensorHandleByPtr(dtype, out.BufferHandle, out.Dims...)
		runner.outputs[c.Name] = tensor
	}

	return
}

// GetInput returns a Tensor attached to the target model.
func (r *Runner) GetInput(name string) (Tensor, error) {
	tensor, ok := r.inputs[name]
	if !ok {
		return nil, fmt.Errorf("%s is not attached", name)
	}
	return tensor, nil
}

// RunWithTensor inputs the tenser with tne name, and runs.
func (r *Runner) RunWithTensor(name string, t Tensor) error {
	return r.Run(map[string]Tensor{
		name: t,
	})
}

// Run with the inputs which are set name and tensor as key-value.
// If nothing to input, set nil.
func (r *Runner) Run(inputs map[string]Tensor) error {
	for n, t := range inputs {
		tensor, ok := r.inputs[n]
		if !ok {
			return fmt.Errorf("%s is not attached", n)
		}
		if err := updateArray(t, tensor); err != nil {
			return fmt.Errorf("cannot update array, %v", err)
		}
	}
	return r.model.Run()
}

// Outputs all variables set by the configuration.
func (r *Runner) Outputs() map[string]Tensor {
	return r.outputs
}

// GetOutput returns the target variable.
func (r *Runner) GetOutput(name string) (Tensor, error) {
	t, ok := r.outputs[name]
	if ok {
		return t, nil
	}
	return nil, fmt.Errorf("%s is not found", name)
}

// Stop the runner.
func (r *Runner) Stop() {
	if r.model != nil {
		r.model.Delete()
	}
	if r.modelBuilder != nil {
		r.modelBuilder.Delete()
	}
	if r.vpTable != nil {
		r.vpTable.Delete()
	}
	if r.vptBuilder != nil {
		r.vptBuilder.Delete()
	}
	if r.modelData != nil {
		r.modelData.Delete()
	}
}

func toMenohDtype(dtype TypeDtype) (external.TypeMenohDtype, error) {
	switch dtype {
	case TypeFloat:
		return external.TypeFloat, nil
	default:
		return -1, errors.New("not supported dtype")
	}
}

func toDtype(mdtype external.TypeMenohDtype) (TypeDtype, error) {
	switch mdtype {
	case external.TypeFloat:
		return TypeFloat, nil
	default:
		return -1, errors.New("not supported menoh dtype")
	}
}
