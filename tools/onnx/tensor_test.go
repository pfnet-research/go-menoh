package onnx

import (
	"encoding/binary"
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
		expectedData, err := proto.Marshal(expected)
		if err != nil {
			t.Fatalf("cannot make test tensor binary, %v", err)
		}
		expectedDataPath := filepath.Join(tempDir, "test_tensor.pb")
		if err := ioutil.WriteFile(expectedDataPath, expectedData, 0644); err != nil {
			t.Fatalf("cannot save test tensor proto, %v", err)
		}

		actual, err := LoadONNXTensorFromFile(expectedDataPath)
		if err != nil {
			t.Fatal(err)
		}
		if actual.GetDataType() != expected.GetDataType() {
			t.Errorf("loaded datatype should be %v, but %v", expected.GetDataType(), actual.GetDataType())
		}
		arrayValueTest := true
		if len(actual.GetFloatData()) == len(expected.GetFloatData()) {
			for i, f := range actual.GetFloatData() {
				if f != expected.GetFloatData()[i] {
					arrayValueTest = false
					break
				}
			}
		} else {
			arrayValueTest = false
		}
		if !arrayValueTest {
			t.Errorf("loaded array should be %v, but %v", expected.GetFloatData(), actual.GetFloatData())
		}
		dimsTest := true
		if len(actual.GetDims()) == len(expected.GetDims()) {
			for i, d := range actual.GetDims() {
				if d != expected.GetDims()[i] {
					dimsTest = false
					break
				}
			}
		} else {
			dimsTest = false
		}
		if !dimsTest {
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
		expectedData, err := proto.Marshal(expected)
		if err != nil {
			t.Fatalf("cannot make test tensor binary, %v", err)
		}
		expectedDataPath := filepath.Join(tempDir, "test_tensor_error.pb")
		if err := ioutil.WriteFile(expectedDataPath, expectedData[:len(expectedData)-1], 0644); err != nil {
			t.Fatalf("cannot save test tensor proto, %v", err)
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

		arrayValueTest := true
		for i, f := range floats {
			if f != expectedFloats[i] {
				arrayValueTest = false
				break
			}
		}
		if !arrayValueTest {
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

		arrayValueTest := true
		for i, f := range floats {
			if f != expectedFloats[i] {
				arrayValueTest = false
				break
			}
		}
		if !arrayValueTest {
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
