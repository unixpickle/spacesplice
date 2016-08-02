package spacesplice

import (
	"bytes"
	"encoding/gob"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"path/filepath"
	"strings"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/boosting"
)

const (
	boostMaxSamples = 50000
	boostSteps      = 20
	boostBacktrack  = 15
	boostLookahead  = 5
)

func init() {
	gob.Register(&boostClassifier{})
}

type BoostStumps struct {
	classifier *boosting.SumClassifier
}

// TrainBoostStumps trains a boosted classifier on a
// directory full of sample text files.
func TrainBoostStumps(corpusDir string) (*BoostStumps, error) {
	contents, err := ioutil.ReadDir(corpusDir)
	if err != nil {
		return nil, err
	}

	log.Println("Building samples...")

	var samples boostSampleList
	for _, fileInfo := range contents {
		if strings.HasPrefix(fileInfo.Name(), ".") {
			continue
		}
		path := filepath.Join(corpusDir, fileInfo.Name())
		sampleBody, err := ioutil.ReadFile(path)
		if err != nil {
			return nil, err
		}
		fields := strings.Fields(string(sampleBody))
		boundaries := map[int]bool{}
		var joined bytes.Buffer
		var idx int
		for _, f := range fields {
			for i, x := range []byte(f) {
				if i == len(f)-1 {
					boundaries[idx] = true
				}
				joined.WriteByte(x)
				idx++
			}
		}
		for i := 0; i < joined.Len(); i++ {
			samples = append(samples, boostSample{
				document: joined.Bytes(),
				idx:      i,
				space:    boundaries[i],
			})
		}
	}

	if len(samples) > boostMaxSamples {
		for i := 0; i < boostMaxSamples; i++ {
			idx := rand.Intn(len(samples)-i) + i
			samples[i], samples[idx] = samples[idx], samples[i]
		}
		samples = samples[:boostMaxSamples]
	}

	log.Println("Creating classifier...")
	gradienter := boosting.Gradient{
		Loss:    boosting.ExpLoss{},
		Desired: make(linalg.Vector, samples.Len()),
		List:    samples,
		Pool:    boostPool{},
	}
	for i, sample := range samples {
		if !sample.space {
			gradienter.Desired[i] = -1
		} else {
			gradienter.Desired[i] = 1
		}
	}
	for i := 0; i < boostSteps; i++ {
		cost := gradienter.Step()
		log.Println("Step", i, "cost", cost)
	}

	return &BoostStumps{classifier: &gradienter.Sum}, nil
}

// DeserializeBoostStumps deserializes a BoostStumps
// which was serialized with BoostStumps.Serialize().
func DeserializeBoostStumps(d []byte) (*BoostStumps, error) {
	buf := bytes.NewBuffer(d)
	dec := gob.NewDecoder(buf)
	var res boosting.SumClassifier
	if err := dec.Decode(&res); err != nil {
		return nil, err
	}
	return &BoostStumps{classifier: &res}, nil
}

// Fields uses the classifier to split the spaceless
// text into fields (i.e. words).
func (b *BoostStumps) Fields(text string) []string {
	parts := strings.Fields(text)
	var res []string
	for _, partStr := range parts {
		part := []byte(partStr)
		var field bytes.Buffer
		for i := 0; i < len(part); i++ {
			sample := boostSampleList{boostSample{
				document: part,
				idx:      i,
			}}
			field.WriteByte(part[i])
			classification := b.classifier.Classify(sample)[0]
			if classification > 0 {
				res = append(res, field.String())
				field.Reset()
			}
		}
		if field.Len() > 0 {
			res = append(res, field.String())
		}
	}
	return res
}

// SerializerType returns the unique ID used to
// serialize the BoostStumps type with the serializer
// package.
func (b *BoostStumps) SerializerType() string {
	return serializerTypeBoostStumps
}

// Serialize serializes the BoostStumps.
func (b *BoostStumps) Serialize() ([]byte, error) {
	var res bytes.Buffer
	enc := gob.NewEncoder(&res)
	if err := enc.Encode(b.classifier); err != nil {
		return nil, err
	}
	return res.Bytes(), nil
}

type boostSample struct {
	document []byte
	idx      int
	space    bool
}

func (b *boostSample) ValueAt(idx int) byte {
	if idx >= 0 {
		if idx+b.idx >= len(b.document) {
			return 0
		}
		return b.document[idx+b.idx]
	} else {
		if idx+b.idx < 0 {
			return 0
		}
		return b.document[idx+b.idx]
	}
}

type boostSampleList []boostSample

func (b boostSampleList) Len() int {
	return len(b)
}

type boostClassifier struct {
	RelIdx int
	Value  byte
}

func (b *boostClassifier) Classify(s boosting.SampleList) linalg.Vector {
	res := make(linalg.Vector, s.Len())
	for i, x := range s.(boostSampleList) {
		if x.ValueAt(i) == b.Value {
			res[i] = 1
		} else {
			res[i] = -1
		}
	}
	return res
}

type boostPool struct{}

func (_ boostPool) BestClassifier(s boosting.SampleList, w linalg.Vector) boosting.Classifier {
	list := s.(boostSampleList)

	var bestDot float64
	var bestClassifier boosting.Classifier
	for idx := -boostBacktrack; idx <= boostLookahead; idx++ {
		for value := 0; value < 0x100; value++ {
			var dot float64
			for i, sample := range list {
				realVal := sample.ValueAt(i)
				if byte(value) == realVal {
					dot += w[i]
				} else {
					dot -= w[i]
				}
			}
			dot = math.Abs(dot)
			if dot > bestDot || bestClassifier == nil {
				bestDot = dot
				bestClassifier = &boostClassifier{
					RelIdx: idx,
					Value:  byte(value),
				}
			}
		}
	}
	return bestClassifier
}
