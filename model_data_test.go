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
