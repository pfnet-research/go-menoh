package menoh

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func getTestONNXDataset() (string, InputConfig, OutputConfig, error) {
	onnxPath := filepath.Join("test_data", "MLP.onnx")
	if _, err := os.Stat(onnxPath); err != nil {
		return "", InputConfig{}, OutputConfig{}, fmt.Errorf(
			"ONNX file is not found, please put the file to %v", onnxPath)
	}
	inputConfig := InputConfig{
		Name:  "input",
		Dtype: TypeFloat,
		Dims:  []int32{1, 3},
	}
	outputConfig := OutputConfig{
		Name:  "fc2",
		Dtype: TypeFloat,
	}
	return onnxPath, inputConfig, outputConfig, nil
}

func getRunner() (*Runner, error) {
	onnxPath, _, _, err := getTestONNXDataset()
	if err != nil {
		return nil, err
	}
	conf := Config{
		ONNXModelPath: onnxPath,
		Backend:       TypeMKLDNN,
		Inputs: []InputConfig{
			{
				Name:  "input",
				Dtype: TypeFloat,
				Dims:  []int32{1, 3},
			},
		},
		Outputs: []OutputConfig{
			{
				Name:         "fc1",
				Dtype:        TypeFloat,
				FromInternal: true,
			},
			{
				Name:  "fc2",
				Dtype: TypeFloat,
			},
		},
	}
	return NewRunner(conf)
}

func TestNewRunnerSuccess(t *testing.T) {
	onnxPath, inputConfig, outputConfig, err := getTestONNXDataset()
	if err != nil {
		t.Fatal(err)
	}
	// success
	t.Run("load valid ONNX model without output", func(t *testing.T) {
		conf := Config{
			ONNXModelPath: onnxPath,
			Backend:       TypeMKLDNN,
			Inputs:        []InputConfig{inputConfig},
		}
		runner, err := NewRunner(conf)
		if err != nil {
			t.Errorf("runner should be created without error, %v", err)
		}
		if runner == nil {
			t.Fatal("runner should be created")
		}
		defer runner.Stop()
	})
	t.Run("load valid ONNX model", func(t *testing.T) {
		conf := Config{
			ONNXModelPath: onnxPath,
			Backend:       TypeMKLDNN,
			Inputs:        []InputConfig{inputConfig},
			Outputs:       []OutputConfig{outputConfig},
		}
		runner, err := NewRunner(conf)
		if err != nil {
			t.Errorf("runner should be created without error, %v", err)
		}
		if runner == nil {
			t.Fatal("runner should be created")
		}
		defer runner.Stop()
	})
	t.Run("load valid ONNX model", func(t *testing.T) {
		onnxData, err := ioutil.ReadFile(onnxPath)
		if err != nil {
			t.Fatal(err)
		}
		modelData, err := NewModelDataFromBytes(onnxData)
		if err != nil {
			t.Fatal(err)
		}
		conf := Config{
			Backend: TypeMKLDNN,
			Inputs:  []InputConfig{inputConfig},
		}
		runner, err := NewRunnerWithModelData(modelData, conf)
		if err != nil {
			t.Errorf("runner should be created without error, %v", err)
		}
		if runner == nil {
			t.Fatal("runner should be created")
		}
		defer runner.Stop()
	})
}

func TestNewRunnerFail(t *testing.T) {
	onnxPath, inputConfig, outputConfig, err := getTestONNXDataset()
	if err != nil {
		t.Fatal(err)
	}
	// fail
	type testConfig struct {
		name     string
		config   Config
		expected string
	}
	testSet := []testConfig{
		{
			name:     "setup with empty config",
			config:   Config{},
			expected: "invalid filename",
		},
		{
			name:     "load invalid ONNX file name",
			config:   Config{ONNXModelPath: ""},
			expected: "invalid filename",
		},
		{
			name:     "attach no input profile",
			config:   Config{ONNXModelPath: onnxPath},
			expected: "variable not found",
		},
		{
			name: "attach invalid input profile",
			config: Config{
				ONNXModelPath: onnxPath,
				Inputs: []InputConfig{
					{
						Name:  "input",
						Dtype: TypeFloat,
						Dims:  []int32{1, 4},
					},
				},
			},
			expected: "dimension mismatch",
		},
		{
			name: "attach invalid output name",
			config: Config{
				ONNXModelPath: onnxPath,
				Inputs:        []InputConfig{inputConfig},
				Outputs: []OutputConfig{
					{
						Name:  "dummy_output",
						Dtype: TypeFloat,
					},
				},
			},
			expected: "output not found",
		},
		{
			name: "invalid backend",
			config: Config{
				ONNXModelPath: onnxPath,
				Inputs:        []InputConfig{inputConfig},
				Outputs:       []OutputConfig{outputConfig},
			},
			expected: "backend",
		},
	}
	for _, ts := range testSet {
		name, config, expected := ts.name, ts.config, ts.expected
		t.Run(name, func(t *testing.T) {
			runner, err := NewRunner(config)
			if err != nil {
				if !strings.Contains(fmt.Sprintf("%v", err), expected) {
					t.Errorf(`error message should contain expected phrase
   expected: %s
   actual  : %v`, expected, err)
				}
			} else {
				t.Error("an error should be occurred")
			}
			func() {
				if runner != nil {
					t.Error("runner should not be created")
					defer runner.Stop()
				}
			}()
		})
	}
}

func TestGetInput(t *testing.T) {
	runner, err := getRunner()
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Stop()

	t.Run("get input", func(t *testing.T) {
		actual, err := runner.GetInput("input")
		if err != nil {
			t.Fatalf("input variable should be get, %v", err)
		}
		expected := &FloatTensor{
			Dims:  []int32{1, 3},
			Array: []float32{0., 0., 0.},
		}
		if !tensorEquals(actual, expected) {
			t.Errorf(`input variable should equal to expected array
   expected: %v
   actual  : %v`, expected, actual)
		}
	})
	t.Run("get no-existed input", func(t *testing.T) {
		actual, err := runner.GetInput("dummy_input")
		if err == nil {
			t.Error("an error should be occurred")
		}
		if actual != nil {
			t.Errorf("runner should return nothing, but return %v", actual)
		}
	})
}

func TestRunWithTensorAndGetOutput(t *testing.T) {
	runner, err := getRunner()
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Stop()

	t.Run("run with intput variable", func(t *testing.T) {
		input := &FloatTensor{
			Dims:  []int32{1, 3},
			Array: []float32{0., 1., 2.},
		}
		if err := runner.RunWithTensor("input", input); err != nil {
			t.Fatalf("the runner should run without error, %v", err)
		}

		t.Run("get output", func(t *testing.T) {
			actual, err := runner.GetOutput("fc2")
			if err != nil {
				t.Fatalf("the runner should return the output, %v", err)
			}
			expected := &FloatTensor{
				Dims:  []int32{1, 5},
				Array: []float32{0., 0., 15., 96., 177},
			}
			if !tensorEquals(actual, expected) {
				t.Fatalf(`output variable should equal to expected array
   expected: %v
   actual  : %v`, expected, actual)
			}

			t.Run("run next input", func(t *testing.T) {
				input2 := &FloatTensor{
					Dims:  []int32{1, 3},
					Array: []float32{0., 0.5, 1.},
				}
				if err := runner.RunWithTensor("input", input2); err != nil {
					t.Fatalf("the runner should run without error, %v", err)
				}

				t.Run("get 2nd. output", func(t *testing.T) {
					actual, err := runner.GetOutput("fc2")
					if err != nil {
						t.Fatalf("the runner should return the output, %v", err)
					}
					expected := &FloatTensor{
						Dims:  []int32{1, 5},
						Array: []float32{0., 0., 8., 51., 94},
					}
					if !tensorEquals(actual, expected) {
						t.Fatalf(`output variable should equal to expected array
   expected: %v
   actual  : %v`, expected, actual)
					}
				})
			})
		})

		t.Run("get no-existed output", func(t *testing.T) {
			output, err := runner.GetOutput("fc3")
			if err == nil {
				t.Error("an error should be occurred")
			}
			if output != nil {
				t.Error("the runner should return nothing")
			}
		})
	})

	// fail
	t.Run("run with invalid intput", func(t *testing.T) {
		input := &FloatTensor{
			Dims:  []int32{1, 2},
			Array: []float32{0., 1.},
		}
		if err := runner.RunWithTensor("input", input); err == nil {
			t.Error("an error should be occurred with non profiled input")
		}
	})
	t.Run("put no-existed input", func(t *testing.T) {
		input := &FloatTensor{}
		if err := runner.RunWithTensor("dummy_input", input); err == nil {
			t.Error("an error should be occurred with non profiled input")
		}
	})
}

func TestRunAndOutputs(t *testing.T) {
	runner, err := getRunner()
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Stop()

	t.Run("run with map input", func(t *testing.T) {
		inputs := map[string]Tensor{
			"input": &FloatTensor{
				Dims:  []int32{1, 3},
				Array: []float32{0., 1., 2.},
			},
		}
		if err := runner.Run(inputs); err != nil {
			t.Fatalf("the runner should run without error, %v", err)
		}

		t.Run("get output", func(t *testing.T) {
			outputs := runner.Outputs()
			actual, ok := outputs["fc2"]
			if !ok {
				t.Fatal("the runner should return the output")
			}
			expected := &FloatTensor{
				Dims:  []int32{1, 5},
				Array: []float32{0., 0., 15., 96., 177},
			}
			if !tensorEquals(actual, expected) {
				t.Fatalf(`output variable should equal to expected array
   expected: %v
   actual  : %v`, expected, actual)
			}

			t.Run("run next input using non-copy update", func(t *testing.T) {
				input, err := runner.GetInput("input")
				if err != nil {
					t.Fatalf("the runner should return the input, %v", err)
				}
				input.WriteFloat(1, 0.5)
				input.WriteFloat(2, 1)
				if err := runner.Run(nil); err != nil {
					t.Fatalf("the runner should run without error, %v", err)
				}

				t.Run("get 2nd. output", func(t *testing.T) {
					outputs := runner.Outputs()
					actual, ok := outputs["fc2"]
					if !ok {
						t.Fatal("the runner should return the output")
					}
					expected := &FloatTensor{
						Dims:  []int32{1, 5},
						Array: []float32{0., 0., 8., 51., 94},
					}
					if !tensorEquals(actual, expected) {
						t.Fatalf(`output variable should equal to expected array
   expected: %v
   actual  : %v`, expected, actual)
					}
				})
			})
		})
	})
}

func tensorEquals(t1, t2 Tensor) bool {
	if t1.dtype() != t2.dtype() {
		return false
	}
	if len(t1.Shape()) != len(t2.Shape()) {
		return false
	}
	for i := 0; i < len(t1.Shape()); i++ {
		if t1.Shape()[i] != t2.Shape()[i] {
			return false
		}
	}
	if t1.Size() != t2.Size() {
		return false
	}
	switch t1.dtype() {
	case TypeFloat:
		t1f, _ := t1.FloatArray()
		t2f, _ := t2.FloatArray()
		for i, f := range t1f {
			if t2f[i] != f {
				return false
			}
		}
	default:
		return false
	}
	return true
}
