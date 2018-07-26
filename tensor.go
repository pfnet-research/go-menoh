package menoh

import (
	"errors"
	"fmt"
	"unsafe"
)

// Tensor is base unit of matrix data to pass Menoh model.
type Tensor interface {
	ptr() unsafe.Pointer
	dtype() TypeDtype

	// Size returns array size.
	Size() int

	// FloatArray returns float32 array. Returns an error when the array
	// cannot cast to the type. It is possible that returned array is copied
	// or split-off from attached array in Menoh model. When updating values,
	// WriteFloat method.
	FloatArray() ([]float32, error)

	// WriteFloat puts float value to i-th index of array.
	WriteFloat(int, float32) error
}

func newTensorHandle(dtype TypeDtype, dims ...int32) Tensor {
	// should be switched by data type but Menoh supports only float
	len := 1
	for _, d := range dims {
		len *= int(d)
	}
	return &FloatTensor{
		Dtype: dtype,
		Dims:  dims,
		Array: make([]float32, len),
	}
}

func newTensorHandleByPtr(dtype TypeDtype, ptr unsafe.Pointer, dims ...int32) Tensor {
	// should be switched by data type but Menoh supports only float
	len := 1
	for _, d := range dims {
		len *= int(d)
	}
	return &FloatTensor{
		Dtype: dtype,
		Dims:  dims,
		Array: (*[1 << 31]float32)(ptr)[:len],
	}
}

func updateArray(src, dst Tensor) error {
	if src.dtype() != dst.dtype() {
		return errors.New("the target tensors must be same dtype")
	}
	if src.Size() != dst.Size() {
		return errors.New("array size must be same")
	}
	switch dtype := src.dtype(); dtype {
	case TypeFloat:
		srcf := src.(*FloatTensor)
		dstf := dst.(*FloatTensor)
		for i := range srcf.Array {
			dstf.Array[i] = srcf.Array[i]
		}
	default:
		return errors.New("not supported dtype on replacing")
	}
	return nil
}

// FloatTensor represents float32 Tessor.
type FloatTensor struct {
	Dtype TypeDtype
	Dims  []int32
	Array []float32
}

func (t *FloatTensor) ptr() unsafe.Pointer {
	return unsafe.Pointer(&t.Array[0])
}

func (t *FloatTensor) dtype() TypeDtype {
	return t.Dtype
}

// Size returns array size.
func (t *FloatTensor) Size() int {
	return len(t.Array)
}

// FloatArray returns float32 array.
func (t *FloatTensor) FloatArray() ([]float32, error) {
	return t.Array, nil
}

// WriteFloat puts float value to i-th index of array.
func (t *FloatTensor) WriteFloat(i int, f float32) error {
	if i >= t.Size() {
		return fmt.Errorf("index %d is greater than array size(%d)", i, t.Size())
	}
	t.Array[i] = f
	return nil
}
