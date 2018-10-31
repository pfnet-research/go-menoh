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
func NewRunner(conf Config) (*Runner, error) {
	modelData, err := external.MakeModelDataFromONNX(conf.ONNXModelPath)
	if err != nil {
		return nil, err
	}
	return buildRunner(modelData, conf)
}

// NewRunnerWithModelData returns Runner using configuration and ONNX model.
// The ONNX model is passed on memory, not use conf.ONNXModelPath.
// Spec of a returned runner is same as NewRunner, see docs of the function.
func NewRunnerWithModelData(modelData *ModelData, conf Config) (*Runner, error) {
	return buildRunner(&modelData.ModelData, conf)
}

func (r *Runner) makeVariableProfileTableBuilder() error {
	vptBuilder, err := external.MakeVariableProfileTableBuilder()
	if err != nil {
		return err
	}
	r.vptBuilder = vptBuilder
	for _, c := range r.conf.Inputs {
		menohDtype, _ := toMenohDtype(c.Dtype)
		if err = vptBuilder.AddInputProfile(c.Name, menohDtype, c.Dims...); err != nil {
			return err
		}
	}
	for _, c := range r.conf.Outputs {
		menohDtype, _ := toMenohDtype(c.Dtype)
		if err = vptBuilder.AddOutputProfile(c.Name, menohDtype); err != nil {
			return err
		}
	}
	return nil
}

func (r *Runner) buildVariableProfileTable() error {
	vpt, err := r.vptBuilder.BuildVariableProfileTable(*r.modelData)
	if err != nil {
		return err
	}
	r.vpTable = vpt
	for _, c := range r.conf.Outputs {
		if !c.FromInternal {
			continue
		}
		vp, err := vpt.GetVariableProfile(c.Name)
		if err != nil {
			return err
		}
		dtype, _ := toDtype(vp.Dtype)
		tensor := newTensorHandle(dtype, vp.Dims...)
		r.outputs[c.Name] = tensor
	}
	return nil
}

func (r *Runner) makeModelBuilder() error {
	modelBuilder, err := external.MakeModelBuilder(*r.vpTable)
	if err != nil {
		return err
	}
	r.modelBuilder = modelBuilder
	for _, c := range r.conf.Inputs {
		tensor := newTensorHandle(c.Dtype, c.Dims...)
		if err := modelBuilder.AttachExternalBuffer(c.Name, tensor.ptr()); err != nil {
			return err
		}
		r.inputs[c.Name] = tensor
	}
	for _, c := range r.conf.Outputs {
		if !c.FromInternal {
			continue
		}
		tensor := r.outputs[c.Name]
		if err := modelBuilder.AttachExternalBuffer(c.Name, tensor.ptr()); err != nil {
			return err
		}
	}
	return nil
}

func (r *Runner) buildModel() error {
	model, err := r.modelBuilder.BuildModel(
		*r.modelData, r.conf.Backend.String(), r.conf.BackendConfig)
	if err != nil {
		return err
	}
	r.model = model
	for _, c := range r.conf.Outputs {
		if c.FromInternal {
			continue
		}
		out, err := model.GetVariable(c.Name)
		if err != nil {
			return err
		}
		dtype, _ := toDtype(out.Dtype)
		tensor := newTensorHandleByPtr(dtype, out.BufferHandle, out.Dims...)
		r.outputs[c.Name] = tensor
	}
	return nil
}

func buildRunner(modelData *external.ModelData, conf Config) (runner *Runner, err error) {
	runner = &Runner{
		modelData: modelData,
		conf:      conf,
		inputs:    map[string]Tensor{},
		outputs:   map[string]Tensor{},
	}
	defer func() {
		if err != nil {
			runner.Stop()
			runner = nil
		}
	}()

	if err = runner.makeVariableProfileTableBuilder(); err != nil {
		return
	}
	if err = runner.buildVariableProfileTable(); err != nil {
		return
	}
	if err = runner.makeModelBuilder(); err != nil {
		return
	}
	if err = runner.buildModel(); err != nil {
		return
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
