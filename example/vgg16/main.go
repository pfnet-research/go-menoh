package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	"os"
	"sort"
	"strings"

	"github.com/disintegration/imaging"
	"github.com/pfnet-research/menoh-go"
)

func main() {
	const (
		batch   = 1
		channel = 3
		width   = 224
		height  = 224

		conv1_1InName  = "140326425860192"
		fc6OutName     = "140326200777584"
		softmaxOutName = "140326200803680"
	)
	var (
		inputImagePath  = flag.String("input-image", "../../data/Light_sussex_hen.jpg", "input image path")
		onnxModelPath   = flag.String("model", "../../data/VGG16.onnx", "ONNX model path")
		synsetWordsPath = flag.String("synset-words", "../../data/synset_words.txt", "synset words file path")
	)
	flag.Parse()
	fmt.Println("vgg16 example")

	// prepare input data
	imageFile, err := os.Open(*inputImagePath)
	if err != nil {
		panic(err)
	}
	defer imageFile.Close()
	img, _, err := image.Decode(imageFile)
	if err != nil {
		panic(err)
	}
	resizedImg := cropAndResize(img, width, height)
	resizedImgTensor := &menoh.FloatTensor{
		Dtype: menoh.TypeFloat,
		Dims:  []int32{batch, channel, height, width},
		Array: toOneHotFloats(resizedImg, channel),
	}

	// build model runner
	runner, err := menoh.NewRunner(menoh.Config{
		ONNXModelPath: *onnxModelPath,
		Backend:       menoh.TypeMKLDNN,
		BackendConfig: "",
		Inputs: []menoh.InputConfig{
			menoh.InputConfig{
				Name:  conv1_1InName,
				Dtype: menoh.TypeFloat,
				Dims:  []int32{batch, channel, height, width},
			},
		},
		Outputs: []menoh.OutputConfig{
			menoh.OutputConfig{
				Name:        fc6OutName,
				Dtype:       menoh.TypeFloat,
				FromProfile: true,
			},
			menoh.OutputConfig{
				Name:        softmaxOutName,
				Dtype:       menoh.TypeFloat,
				FromProfile: false,
			},
		},
	})
	if err != nil {
		panic(err)
	}
	defer runner.Stop()

	// run ONNX model with input and get output result
	if err := runner.RunWithTensor(conv1_1InName, resizedImgTensor); err != nil {
		panic(err)
	}
	fc6OutTensor, err := runner.GetOutput(fc6OutName)
	if err != nil {
		panic(err)
	}
	fc6OutData, _ := fc6OutTensor.FloatArray()
	softmaxOutTensor, err := runner.GetOutput(softmaxOutName)
	if err != nil {
		panic(err)
	}
	softmaxOutData, _ := softmaxOutTensor.FloatArray()

	// evalute image detection
	fc6OutLog := make([]string, 10)
	for i, f := range fc6OutData[:10] {
		fc6OutLog[i] = fmt.Sprintf("%.4f", f)
	}
	fmt.Println(strings.Join(fc6OutLog, " "))
	categories, err := loadCategoryList(*synsetWordsPath)
	if err != nil {
		panic(err)
	}
	topKIndices := extractTopKIndexList(softmaxOutData, 5)
	for _, idx := range topKIndices {
		fmt.Printf("%d %.5f %s\n", idx, softmaxOutData[idx], categories[idx])
	}
}

func cropAndResize(img image.Image, width, height int) image.Image {
	return imaging.Fill(img, width, height, imaging.Center, imaging.Linear)
}

func toOneHotFloats(img image.Image, channel int) []float32 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	floats := make([]float32, channel*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			floats[0*(w*h)+y*w+x] = float32(b / 257)
			floats[1*(w*h)+y*w+x] = float32(g / 257)
			floats[2*(w*h)+y*w+x] = float32(r / 257)
		}
	}
	return floats
}

func loadCategoryList(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return []string{}, err
	}
	defer file.Close()

	categories := []string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		categories = append(categories, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return []string{}, err
	}
	return categories, nil
}

func extractTopKIndexList(values []float32, k int) []int {
	type pair struct {
		index int
		value float32
	}
	pairs := make([]pair, len(values))
	for i, f := range values {
		pairs[i] = pair{
			index: i,
			value: f,
		}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})
	topKIndices := make([]int, k)
	for i := 0; i < k; i++ {
		topKIndices[i] = pairs[i].index
	}
	return topKIndices
}
