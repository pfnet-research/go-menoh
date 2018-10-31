package menoh

import (
	"fmt"
	"io/ioutil"
	"strings"
	"testing"
)

func TestNewRawModelData(t *testing.T) {
	actual, err := NewRawModelData()
	if err != nil {
		t.Errorf("new model data should be returned without error, %v", err)
	}
	if actual == nil {
		t.Fatal("new model data should be returned but returned nil")
	}
	actual.Delete()
}

func TestNewModelDataFromPath(t *testing.T) {
	t.Run("set valid path", func(t *testing.T) {
		path, _, _, err := getTestONNXDataset()
		if err != nil {
			t.Fatal(err)
		}
		actual, err := NewModelDataFromPath(path)
		if err != nil {
			t.Errorf("new model data should be returned without error, %v", err)
		}
		if actual == nil {
			t.Fatal("new model data should be returned with valid path")
		}
		defer actual.Delete()
	})

	// fail
	t.Run("set not existed path", func(t *testing.T) {
		actual, err := NewModelDataFromPath("invalid/path")
		if err == nil {
			t.Errorf("model data should not be returned with not exsted path, %v", err)
		}
		if actual != nil {
			t.Error("model data should be nil, but returned")
			defer actual.Delete()
		}
	})
}

func TestNewModelDataFromBytes(t *testing.T) {
	t.Run("set valid binary", func(t *testing.T) {
		path, _, _, err := getTestONNXDataset()
		if err != nil {
			t.Fatal(err)
		}
		data, err := ioutil.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		actual, err := NewModelDataFromBytes(data)
		if err != nil {
			t.Errorf("new model data should be returned without error, %v", err)
		}
		if actual == nil {
			t.Fatal("new model data should be returned with binary data")
		}
		defer actual.Delete()
	})

	// fail
	type testCase struct {
		name string
		data []byte
		msg  string
	}
	testSet := []testCase{
		{
			name: "empty byte",
			data: []byte{},
			msg:  "data is empty",
		},
		{
			name: "invalid binary",
			data: []byte("dummy data"),
			msg:  "parse error",
		},
	}
	for _, ts := range testSet {
		t.Run(ts.name, func(t *testing.T) {
			actual, err := NewModelDataFromBytes(ts.data)
			if err != nil {
				if !strings.Contains(fmt.Sprintf("%v", err), ts.msg) {
					t.Errorf(`error message should contain expected phrase
   expected: %s
   actual  : %v`, ts.msg, err)
				}
			} else {
				t.Error("an error should be occurred")
			}
			func() {
				if actual != nil {
					t.Error("model data should be nil, but returned")
					defer actual.Delete()
				}
			}()
		})
	}
}

func TestAddTensorParameter(t *testing.T) {
	md, err := NewRawModelData()
	if err != nil {
		t.Fatal(err)
	}
	param := &FloatTensor{
		Dims:  []int32{3},
		Array: []float32{0., 0.},
	}
	if err := md.AddTensorParameter("param", param); err != nil {
		t.Errorf("parameter should be added, %v", err)
	}
}

func TestAddNodeAndSetup(t *testing.T) {
	md, err := NewRawModelData()
	if err != nil {
		t.Fatal(err)
	}
	defer md.Delete()
	// Gemm
	if err := md.AddNewNode("Gemm"); err != nil {
		t.Fatalf("new Gemm node should be added, %v", err)
	}
	if err := md.AddInputNameToCurrentNode("A"); err != nil {
		t.Fatalf("input name should be added, %v", err)
	}
	if err := md.AddOutputNameToCurrentNode("Y"); err != nil {
		t.Fatalf("output name should be added, %v", err)
	}
	if err := md.AddAttributeIntToCurrentNode("transA", 1); err != nil {
		t.Fatalf("int attribute should be added, %v", err)
	}
	if err := md.AddAttributeFloatToCurrentNode("alpha", 1.0); err != nil {
		t.Fatalf("float attribute should be added, %v", err)
	}
	// Pad
	if err := md.AddNewNode("Pad"); err != nil {
		t.Fatalf("new Pad node should be added, %v", err)
	}
	if err := md.AddAttributeIntsToCurrentNode("pads", []int{5, 5}); err != nil {
		t.Fatalf("int array attribute should be added, %v", err)
	}
	// GRU
	if err := md.AddNewNode("GRU"); err != nil {
		t.Fatalf("new GRU node should be added, %v", err)
	}
	if err := md.AddAttributeFloatsToCurrentNode("activation_alpha", []float32{0.1, 0.1}); err != nil {
		t.Fatalf("floats attribute should be added, %v", err)
	}
}
