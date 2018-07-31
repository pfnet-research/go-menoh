package main

import (
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"

	"github.com/golang/protobuf/proto"

	"github.com/pfnet-research/go-menoh"
	"github.com/pfnet-research/go-menoh/tools/onnx"
)

const (
	batch      = 1
	channel    = 1
	width      = 28
	height     = 28
	inputName  = "139900320569040"
	outputName = "139898462888656"
)

func main() {
	var (
		dataPath      = flag.String("data", "data/mnist", "dataset path")
		onnxModelPath = flag.String("model", "", "ONNX model path")
		mode          = flag.String("mode", "test", "execution mode")
	)
	flag.Parse()

	modelPath := *onnxModelPath
	if modelPath == "" {
		modelPath = filepath.Join(*dataPath, "mnist.onnx")
	}
	if _, err := os.Stat(modelPath); err != nil {
		panic(fmt.Errorf("ONNX model is not found on '%s'", modelPath))
	}

	switch *mode {
	case "test":
		if err := runAllTest(modelPath, *dataPath); err != nil {
			panic(err)
		}
	case "server":
		fmt.Println("'server' mode is not implemented")
	default:
		panic(fmt.Errorf("'%s' mode is not supported", *mode))
	}
}

func makeRunner(modelPath string) (*menoh.Runner, error) {
	return menoh.NewRunner(menoh.Config{
		ONNXModelPath: modelPath,
		Backend:       menoh.TypeMKLDNN,
		Inputs: []menoh.InputConfig{
			menoh.InputConfig{
				Name:  inputName,
				Dtype: menoh.TypeFloat,
				Dims:  []int32{batch, channel, height, width},
			},
		},
		Outputs: []menoh.OutputConfig{
			menoh.OutputConfig{
				Name:         outputName,
				Dtype:        menoh.TypeFloat,
				FromInternal: false,
			},
		},
	})
}

func runAllTest(modelPath, dataPath string) error {
	runner, err := makeRunner(modelPath)
	if err != nil {
		return fmt.Errorf("cannot build runner, %v", err)
	}
	defer runner.Stop()

	var testErr error
	for i := 0; i < 3; i++ {
		dirPath := filepath.Join(dataPath, fmt.Sprintf("test_data_set_%d", i))
		if _, testErr = os.Stat(dirPath); testErr != nil {
			fmt.Printf("'%s' is not found\n", dirPath)
			continue
		}
		inputDatasetName := fmt.Sprintf("input_%d.pb", i)
		outputDatasetname := fmt.Sprintf("output_%d.pb", i)
		if testErr = runTest(runner,
			filepath.Join(dirPath, inputDatasetName),
			filepath.Join(dirPath, outputDatasetname)); testErr != nil {
			fmt.Printf("fail to test on dataset %d, %v\n", i, testErr)
		}
	}
	if testErr != nil {
		return errors.New("fail to test")
	}
	return nil
}

func runTest(runner *menoh.Runner, inputPath, outputPath string) error {
	input, err := loadONNXTensor(inputPath)
	if err != nil {
		return err
	}

	// classify input array
	dims := input.GetDims()
	inputTensor := &menoh.FloatTensor{
		Dtype: menoh.TypeFloat,
		Dims:  []int32{int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])},
		Array: convertToFloat32Array(input.GetRawData()),
	}
	if err := runner.RunWithTensor(inputName, inputTensor); err != nil {
		return err
	}

	// get output
	actual, err := runner.GetOutput(outputName)
	if err != nil {
		return err
	}
	actualArray, err := actual.FloatArray()
	if err != nil {
		return fmt.Errorf("cannot get float array, %v", err)
	}
	actualNum := argmax(actualArray)

	// compare with expected output
	expected, err := loadONNXTensor(outputPath)
	if err != nil {
		return err
	}
	expectedArray := convertToFloat32Array(expected.GetRawData())
	expectedNum := argmax(expectedArray)
	if actualNum != expectedNum {
		return fmt.Errorf("expected is %d but actual is %d", expectedNum, actualNum)
	}
	return nil
}

func loadONNXTensor(path string) (*onnx.TensorProto, error) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot load '%s', %v", path, err)
	}
	tensor := &onnx.TensorProto{}
	if err := proto.Unmarshal(b, tensor); err != nil {
		return nil, fmt.Errorf("cannot convert to ONNX tensor, %v", err)
	}
	return tensor, nil
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

func argmax(array []float32) int {
	if len(array) == 0 {
		return -1
	}
	if len(array) == 1 {
		return 0
	}
	ret := 0
	temp := array[0]
	for i := 1; i < len(array); i++ {
		if temp < array[i] {
			temp = array[i]
			ret = i
		}
	}
	return ret
}
