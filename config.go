package menoh

type Config struct {
	ONNXModelPath string
	Backend       TypeBackend
	BackendConfig string
	Inputs        []InputConfig
	Outputs       []OutputConfig
}

type InputConfig struct {
	Name  string
	Dtype TypeDtype
	Dims  []int32
}

type OutputConfig struct {
	Name        string
	Dtype       TypeDtype
	FromProfile bool
}

type TypeDtype int

const (
	typeUnknownDtype TypeDtype = iota
	TypeFloat
)

type TypeBackend int

const (
	typeUnknownBackend TypeBackend = iota
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
