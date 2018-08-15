package menoh

import (
	"errors"
	"testing"
	"unsafe"
)

func TestUpdateArray(t *testing.T) {
	t.Run("update array", func(t *testing.T) {
		src := &FloatTensor{
			Dims:  []int32{5},
			Array: []float32{1., 2., 3., 4., 5.},
		}
		dst := &FloatTensor{
			Dims:  []int32{5},
			Array: []float32{0., 0., 0., 0., 0.},
		}
		if err := updateArray(src, dst); err != nil {
			t.Fatalf("updating should succeed, %v", err)
		}
		for i := 0; i < 5; i++ {
			if dst.Array[i] != src.Array[i] {
				t.Error("value should be copied")
			}
		}
	})
	t.Run("update not same dtype", func(t *testing.T) {
		src := &unknownDtypeTensor{}
		dst := &FloatTensor{}
		if err := updateArray(src, dst); err == nil {
			t.Error("an error should be occurred")
		}
	})
	t.Run("update not same size", func(t *testing.T) {
		src := &FloatTensor{
			Dims:  []int32{5},
			Array: make([]float32, 5),
		}
		dst := &FloatTensor{
			Dims:  []int32{4},
			Array: make([]float32, 4),
		}
		if err := updateArray(src, dst); err == nil {
			t.Error("an error should be occurred")
		}
	})
	t.Run("update not supported dtype", func(t *testing.T) {
		src := &unknownDtypeTensor{}
		dst := &unknownDtypeTensor{}
		if err := updateArray(src, dst); err == nil {
			t.Error("an error should be occured")
		}
	})
}

type unknownDtypeTensor struct{}

func (t *unknownDtypeTensor) ptr() unsafe.Pointer {
	return nil
}

func (t *unknownDtypeTensor) dtype() TypeDtype {
	return typeUnknownDtype
}

func (t *unknownDtypeTensor) Size() int {
	return -1
}

func (t *unknownDtypeTensor) Shape() []int32 {
	return []int32{}
}

func (t *unknownDtypeTensor) FloatArray() ([]float32, error) {
	return []float32{}, errors.New("not implemented")
}

func (t *unknownDtypeTensor) WriteFloat(i int, f float32) error {
	return errors.New("not implemented")
}

func TestFloatTensorWriteFloat(t *testing.T) {
	t.Run("write a value", func(t *testing.T) {
		tensor := FloatTensor{
			Dims:  []int32{5},
			Array: []float32{0., 0., 0., 0., 0.},
		}
		targetIdx := 2
		expected := float32(1.0)
		if err := tensor.WriteFloat(targetIdx, expected); err != nil {
			t.Fatalf("updating should succeed, %v", err)
		}
		if tensor.Array[targetIdx] != expected {
			t.Errorf("%d-th array should be updated to %.1f, but %.1f",
				targetIdx, expected, tensor.Array[2])
		}
	})
	t.Run("write to out of range", func(t *testing.T) {
		tensor := FloatTensor{
			Dims:  []int32{5},
			Array: []float32{0., 0., 0., 0., 0.},
		}
		if err := tensor.WriteFloat(5, 1.); err == nil {
			t.Error("an error should be occurred")
		}
	})
}
