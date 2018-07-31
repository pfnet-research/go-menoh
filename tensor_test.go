package menoh

import (
	"testing"
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
			t.Errorf("%d-th array is updated to %.1f, but %.1f",
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
