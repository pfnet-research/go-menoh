package menoh

import (
	"errors"
	"unsafe"
)

// Tensor is base unit of matrix data to pass Menoh model.
type Tensor interface {
	ptr() unsafe.Pointer
	dtype() TypeDtype

	// FloatArray returns float32 array. Returns an error when the array
	// cannot cast to the type.
	FloatArray() ([]float32, error)
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
	switch dtype := src.dtype(); dtype {
	case TypeFloat:
		srcf := src.(*FloatTensor)
		dstf := dst.(*FloatTensor)
		if len(dstf.Array) != len(srcf.Array) {
			return errors.New("array size must be same")
		}
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

// FloatArray returns float32 array.
func (t *FloatTensor) FloatArray() ([]float32, error) {
	return t.Array, nil
}
