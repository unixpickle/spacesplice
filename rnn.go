package spacesplice

import (
	"bytes"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	rnnTrainingSeqSize = 128
	rnnFeatureCount    = 256
	rnnStateSize       = 128
	rnnOutputHidden    = 128
	rnnStepSize        = 0.001
	rnnBatchSize       = 20
	rnnMaxSamples      = 1 << 13
)

// RNN splits a string into fields using a bidirectional
// recurrent neural network.
type RNN struct {
	Net *rnn.Bidirectional
}

func DeserializeRNN(d []byte) (*RNN, error) {
	net, err := rnn.DeserializeBidirectional(d)
	if err != nil {
		return nil, err
	}
	return &RNN{Net: net}, nil
}

func TrainRNN(corpusDir string) (*RNN, error) {
	res := &RNN{Net: createRNN()}

	log.Println("Loading samples...")
	samples, err := createRNNSamples(corpusDir)
	if err != nil {
		return nil, err
	}

	rand.Seed(time.Now().UnixNano())
	sgd.ShuffleSampleSet(samples)
	if samples.Len() > rnnMaxSamples {
		samples = samples.Subset(0, rnnMaxSamples)
	}

	log.Printf("Training on %d samples (Ctrl+C to end)...", samples.Len())
	cost := neuralnet.SigmoidCECost{}
	grad := &sgd.Adam{
		Gradienter: &seqtoseq.SeqFuncGradienter{
			Learner:  res.Net,
			SeqFunc:  res.Net,
			CostFunc: cost,
		},
	}

	var epoch int
	sgd.SGDInteractive(grad, samples, rnnStepSize, rnnBatchSize, func() bool {
		tc := seqtoseq.TotalCostSeqFunc(res.Net, 20, samples, cost)
		log.Printf("Epoch %d: cost=%f", epoch, tc)
		epoch++
		return true
	})

	return res, nil
}

func (r *RNN) Fields(text string) []string {
	var res []string
	parts := strings.Fields(text)
	for _, part := range parts {
		partBytes := []byte(part)
		inSeq := make([]autofunc.Result, len(partBytes))
		for i, b := range partBytes {
			v := make(linalg.Vector, rnnFeatureCount)
			v[int(b)] = 1
			inSeq[i] = &autofunc.Variable{Vector: v}
		}
		out := r.Net.BatchSeqs([][]autofunc.Result{inSeq}).OutputSeqs()[0]

		var buf bytes.Buffer
		for i, b := range partBytes {
			buf.WriteByte(b)
			if out[i][0] > 0 {
				res = append(res, buf.String())
				buf.Reset()
			}
		}
		if buf.Len() > 0 {
			res = append(res, buf.String())
		}
	}
	return res
}

func (r *RNN) SerializerType() string {
	return serializerTypeRNN
}

func (r *RNN) Serialize() ([]byte, error) {
	return r.Net.Serialize()
}

func createRNN() *rnn.Bidirectional {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  rnnStateSize * 2,
			OutputCount: rnnOutputHidden,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  rnnOutputHidden,
			OutputCount: 1,
		},
	}
	outNet.Randomize()
	return &rnn.Bidirectional{
		Forward:  &rnn.RNNSeqFunc{Block: rnn.NewGRU(rnnFeatureCount, rnnStateSize)},
		Backward: &rnn.RNNSeqFunc{Block: rnn.NewGRU(rnnFeatureCount, rnnStateSize)},
		Output:   &rnn.NetworkSeqFunc{Network: outNet},
	}
}

func createRNNSamples(corpusDir string) (sgd.SampleSet, error) {
	var res rnnSampleSet
	err := ReadSamples(corpusDir, func(sampleBody []byte) {
		data, bounds := rnnBoundedSample(string(sampleBody))
		for i := 0; i+rnnTrainingSeqSize <= len(data); i += rnnTrainingSeqSize {
			res.samples = append(res.samples, data[i:i+rnnTrainingSeqSize])
			res.endFlags = append(res.endFlags, bounds[i:i+rnnTrainingSeqSize])
		}
	})
	if err != nil {
		return nil, err
	}
	return &res, nil
}

type rnnSampleSet struct {
	samples  [][]byte
	endFlags [][]bool
}

func (r *rnnSampleSet) Len() int {
	return len(r.samples)
}

func (r *rnnSampleSet) Copy() sgd.SampleSet {
	res := &rnnSampleSet{
		samples:  make([][]byte, len(r.samples)),
		endFlags: make([][]bool, len(r.endFlags)),
	}
	copy(res.samples, r.samples)
	copy(res.endFlags, r.endFlags)
	return res
}

func (r *rnnSampleSet) Swap(i, j int) {
	r.samples[i], r.samples[j] = r.samples[j], r.samples[i]
	r.endFlags[i], r.endFlags[j] = r.endFlags[j], r.endFlags[i]
}

func (r *rnnSampleSet) GetSample(idx int) interface{} {
	data := r.samples[idx]
	res := seqtoseq.Sample{
		Inputs:  make([]linalg.Vector, len(data)),
		Outputs: make([]linalg.Vector, len(data)),
	}
	for i, b := range data {
		inVec := make(linalg.Vector, rnnFeatureCount)
		inVec[int(b)] = 1
		res.Inputs[i] = inVec
		if r.endFlags[idx][i] {
			res.Outputs[i] = []float64{1}
		} else {
			res.Outputs[i] = []float64{0}
		}
	}
	return res
}

func (r *rnnSampleSet) Subset(start, end int) sgd.SampleSet {
	return &rnnSampleSet{
		samples:  r.samples[start:end],
		endFlags: r.endFlags[start:end],
	}
}

type rnnBatchLearner struct {
	*rnn.SeqFuncFunc
	Params []*autofunc.Variable
}

func (r *rnnBatchLearner) Parameters() []*autofunc.Variable {
	return r.Params
}

func rnnBoundedSample(data string) (sample []byte, bounds []bool) {
	fields := strings.Fields(data)
	for _, field := range fields {
		byteField := []byte(field)
		for i, x := range byteField {
			sample = append(sample, x)
			bounds = append(bounds, i == len(byteField)-1)
		}
	}
	return
}
