package onnx

import (
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/golang/protobuf/proto"
)

func TestLoadONNXTensorFromFile(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "go-menoh-test-")
	if err != nil {
		t.Fatal("cannot make temporary directory")
	}
	defer os.Remove(tempDir)

	t.Run("valid path", func(t *testing.T) {
		// prepare proto file
		testType := TensorProto_FLOAT
		expected := &TensorProto{
			DataType:  &testType,
			Dims:      []int64{1, 2, 3},
			FloatData: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		}
		expectedDataPath := filepath.Join(tempDir, "test_tensor.pb")
		if err := saveTensorProto(expected, expectedDataPath, false); err != nil {
			t.Fatal(err)
		}

		actual, err := LoadONNXTensorFromFile(expectedDataPath)
		if err != nil {
			t.Fatal(err)
		}
		if actual.GetDataType() != expected.GetDataType() {
			t.Errorf("loaded datatype should be %v, but %v", expected.GetDataType(), actual.GetDataType())
		}
		if !checkFloats(actual.GetFloatData(), expected.GetFloatData()) {
			t.Errorf("loaded array should be %v, but %v", expected.GetFloatData(), actual.GetFloatData())
		}
		if !checkInt64s(actual.GetDims(), expected.GetDims()) {
			t.Errorf("loaded dims should be %v, but %v", expected.GetDims(), actual.GetDims())
		}
	})

	t.Run("empty path", func(t *testing.T) {
		actual, err := LoadONNXTensorFromFile("")
		if err == nil {
			t.Error("loading tensor should be failed with empty path")
		}
		if actual != nil {
			t.Errorf("loading tensor should be nil but get %v", actual)
		}
	})

	t.Run("invalid binary file", func(t *testing.T) {
		// prepare proto file
		testType := TensorProto_FLOAT
		expected := &TensorProto{
			DataType:  &testType,
			Dims:      []int64{1, 2, 3},
			FloatData: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		}
		expectedDataPath := filepath.Join(tempDir, "test_tensor.pb")
		if err := saveTensorProto(expected, expectedDataPath, true); err != nil {
			t.Fatal(err)
		}

		actual, err := LoadONNXTensorFromFile(expectedDataPath)
		if err == nil {
			t.Error("loading tensor should be failed with invalid binary")
		}
		if actual != nil {
			t.Errorf("loading tensor should be nil but get %v", actual)
		}
	})
}

func saveTensorProto(tensor *TensorProto, out string, broken bool) error {
	tensorData, err := proto.Marshal(tensor)
	if err != nil {
		return fmt.Errorf("cannot make test tensor binary, %v", err)
	}
	if broken {
		tensorData = tensorData[:len(tensorData)-1]
	}
	if err := ioutil.WriteFile(out, tensorData, 0644); err != nil {
		return fmt.Errorf("cannot save test tensor proto, %v", err)
	}
	return nil
}

func checkFloats(f1, f2 []float32) bool {
	if len(f1) != len(f2) {
		return false
	}
	for i, f := range f1 {
		if f != f2[i] {
			return false
		}
	}
	return true
}

func checkInt64s(i1, i2 []int64) bool {
	if len(i1) != len(i2) {
		return false
	}
	for i, d := range i1 {
		if d != i2[i] {
			return false
		}
	}
	return true
}

func TestConvertToMenohTensor(t *testing.T) {
	t.Run("float type", func(t *testing.T) {
		testType := TensorProto_FLOAT
		expectedFloats := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
		input := &TensorProto{
			DataType:  &testType,
			Dims:      []int64{1, 2, 3},
			FloatData: expectedFloats,
		}
		actual, err := ConvertToMenohTensor(input)
		if err != nil {
			t.Fatalf("converting should success, but %v", err)
		}
		if actual.Size() != len(expectedFloats) {
			t.Errorf("converted tensor size should be %v, but %v", len(expectedFloats), actual.Size())
		}
		floats, err := actual.FloatArray()
		if err != nil {
			t.Errorf("converted tensor should have float array, but %v", err)
		}
		if !checkFloats(floats, expectedFloats) {
			t.Errorf("converted array should be %v, but %v", expectedFloats, floats)
		}
	})

	t.Run("float32 raw type", func(t *testing.T) {
		testType := TensorProto_FLOAT
		expectedFloats := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
		expectedRaw := convertFloat32ArrayToRaw(expectedFloats)
		input := &TensorProto{
			DataType: &testType,
			Dims:     []int64{1, 2, 3},
			RawData:  expectedRaw,
		}
		actual, err := ConvertToMenohTensor(input)
		if err != nil {
			t.Fatalf("converting should success, but %v", err)
		}
		if actual.Size() != len(expectedFloats) {
			t.Errorf("converted tensor size should be %v, but %v", len(expectedFloats), actual.Size())
		}
		floats, err := actual.FloatArray()
		if err != nil {
			t.Errorf("converted tensor should have float array, but %v", err)
		}
		if !checkFloats(floats, expectedFloats) {
			t.Errorf("converted array should be %v, but %v", expectedFloats, floats)
		}
	})

	t.Run("unsupported type", func(t *testing.T) {
		testType := TensorProto_DOUBLE
		expectedDoubles := []float64{float64(0.1), float64(0.2), float64(0.3)}
		input := &TensorProto{
			DataType:   &testType,
			Dims:       []int64{1, 3},
			DoubleData: expectedDoubles,
		}
		actual, err := ConvertToMenohTensor(input)
		if err == nil {
			t.Error("unsupported type should be failed to convert")
		}
		if actual != nil {
			t.Errorf("unsupported type should not be converted, but %v", actual)
		}
	})
}

func convertFloat32ArrayToRaw(floats []float32) []byte {
	bitLength := 4
	raw := make([]byte, len(floats)*bitLength)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(raw[i*bitLength:(i+1)*bitLength], math.Float32bits(f))
	}
	return raw
}
