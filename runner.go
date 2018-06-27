package menoh

import (
	"errors"
	"fmt"

	"github.com/pfnet-research/menoh-go/external"
)

// Runner setups Menoh model with profiling and executes.
type Runner struct {
	modelData    *external.ModelData
	vptBuilder   *external.VariableProfileTableBuilder
	vpTable      *external.VariableProfileTable
	modelBuilder *external.ModelBuilder
	model        *external.Model

	conf Config

	vps map[string]Tensor
}

// NewRunner returns Runner.
func NewRunner(conf Config) (runner *Runner, rootErr error) {
	runner = &Runner{conf: conf}
	defer func() {
		if rootErr != nil {
			runner.Stop()
		}
	}()

	modelData, err := external.MakeModelDataFromONNX(conf.ONNXModelPath)
	if err != nil {
		rootErr = err
		return
	}
	runner.modelData = modelData

	vptBuilder, err := external.MakeVariableProfileTableBuilder()
	if err != nil {
		rootErr = err
		return
	}
	runner.vptBuilder = vptBuilder
	for _, c := range conf.Inputs {
		menohDtype, _ := toMenohDtype(c.Dtype)
		if err := vptBuilder.AddInputProfile(c.Name, menohDtype, c.Dims...); err != nil {
			rootErr = err
			return
		}
	}
	for _, c := range conf.Outputs {
		menohDtype, _ := toMenohDtype(c.Dtype)
		if err := vptBuilder.AddOutputProfile(c.Name, menohDtype); err != nil {
			rootErr = err
			return
		}
	}

	vpt, err := vptBuilder.BuildVariableProfileTable(*modelData)
	if err != nil {
		rootErr = err
		return
	}
	runner.vpTable = vpt
	runner.vps = map[string]Tensor{}
	for _, c := range conf.Outputs {
		if !c.FromProfile {
			continue
		}
		vp, err := vpt.GetVariableProfile(c.Name)
		if err != nil {
			rootErr = err
			return
		}
		dtype, _ := toDtype(vp.Dtype)
		tensor := newTensorHandle(dtype, vp.Dims...)
		runner.vps[c.Name] = tensor
	}

	modelBuilder, err := external.MakeModelBuilder(*vpt)
	if err != nil {
		rootErr = err
		return
	}
	runner.modelBuilder = modelBuilder
	for _, c := range conf.Inputs {
		tensor := newTensorHandle(c.Dtype, c.Dims...)
		if err := modelBuilder.AttachExternalBuffer(c.Name, tensor.ptr()); err != nil {
			rootErr = err
			return
		}
		runner.vps[c.Name] = tensor
	}
	for _, c := range conf.Outputs {
		if !c.FromProfile {
			continue
		}
		tensor := runner.vps[c.Name]
		if err := modelBuilder.AttachExternalBuffer(c.Name, tensor.ptr()); err != nil {
			rootErr = err
			return
		}
	}

	model, err := modelBuilder.BuildModel(*modelData, conf.Backend.String(), conf.BackendConfig)
	if err != nil {
		rootErr = err
		return
	}
	runner.model = model
	for _, c := range conf.Outputs {
		if c.FromProfile {
			continue
		}
		out, err := model.GetVariable(c.Name)
		if err != nil {
			rootErr = err
			return
		}
		dtype, _ := toDtype(out.Dtype)
		tensor := newTensorHandleByPtr(dtype, out.BufferHandle, out.Dims...)
		runner.vps[c.Name] = tensor
	}

	return
}

func (r *Runner) RunWithTensor(name string, t Tensor) error {
	return r.Run(map[string]Tensor{
		name: t,
	})
}

func (r *Runner) Run(inputs map[string]Tensor) error {
	for n, t := range inputs {
		tensor, ok := r.vps[n]
		if !ok {
			return fmt.Errorf("%s is not attached", n)
		}
		if err := updateArray(t, tensor); err != nil {
			return err
		}
	}
	return r.model.Run()
}

func (r *Runner) Outputs() map[string]Tensor {
	return r.vps
}

func (r *Runner) GetOutput(name string) (Tensor, error) {
	t, ok := r.vps[name]
	if ok {
		return t, nil
	}
	return nil, fmt.Errorf("%s is not found", name)
}

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
