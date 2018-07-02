package menoh

// Config is setup information to build Menoh model.
type Config struct {
	ONNXModelPath string         // path of ONNX file
	Backend       TypeBackend    // backend type, like menoh.TypeMKLDNN
	BackendConfig string         // backend configuration
	Inputs        []InputConfig  // list of input configuration
	Outputs       []OutputConfig // list of output configuration
}

// InputConfig is input variable information to pass to the model.
type InputConfig struct {
	Name  string    // layer name
	Dtype TypeDtype // data type
	Dims  []int32   // list of dimension size
}

// OutputConfig is output variable information to get from the model.
type OutputConfig struct {
	Name         string    // layer name
	Dtype        TypeDtype // data type
	FromInternal bool      // if the output comes from operator variable such as weight, set true
}

// TypeDtype is a type of data.
type TypeDtype int

const (
	typeUnknownDtype TypeDtype = iota
	// TypeFloat is a TypeDtype of Float.
	TypeFloat
)

// TypeBackend is a type of backend, like MKL-DNN.
type TypeBackend int

const (
	typeUnknownBackend TypeBackend = iota
	// TypeMKLDNN is a TypeBackend of MKL-DNN
	TypeMKLDNN
)

func (t TypeBackend) String() string {
	switch t {
	case TypeMKLDNN:
		return "mkldnn"
	default:
		return "unknown"
	}
}
