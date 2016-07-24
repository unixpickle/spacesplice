package spacesplice

import (
	"bytes"
	"encoding/gob"
	"errors"
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/unixpickle/weakai/idtrees"
)

const (
	forestSize         = 100
	forestFeatureCount = 10
	forestSampleCount  = 1000
)

func init() {
	gob.Register(&forestSample{})
}

// Forest is a Splicer that uses a random forest to
// insert spaces into a piece of text.
type Forest struct {
	forest idtrees.Forest
}

// TrainForest trains a forest on a directory full
// sample text files.
func TrainForest(corpusDir string) (*Forest, error) {
	contents, err := ioutil.ReadDir(corpusDir)
	if err != nil {
		return nil, err
	}

	var samples []idtrees.Sample
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
		joinedStr := joined.String()
		for i := 0; i < joined.Len(); i++ {
			samples = append(samples, &forestSample{
				textDoc:    joinedStr,
				index:      i,
				endOfField: boundaries[i],
			})
		}
	}

	var allAttrs []idtrees.Attr
	for i := -20; i <= 20; i++ {
		allAttrs = append(allAttrs, i)
	}
	forest := idtrees.BuildForest(forestSize, samples, allAttrs, forestSampleCount,
		forestFeatureCount, func(s []idtrees.Sample, a []idtrees.Attr) *idtrees.Tree {
			return idtrees.ID3(s, a, 0)
		})
	return &Forest{forest: forest}, nil
}

// DeserializeForest deserializes a Forest which
// was serialized with Forest.Serialize().
func DeserializeForest(d []byte) (*Forest, error) {
	// TODO: this.
	return nil, errors.New("not yet implemented")
}

// Fields uses the forest to split the spaceless text
// into fields (i.e. words).
func (f *Forest) Fields(text string) []string {
	// parts := strings.Fields(text)
	var res []string
	// TODO: this.
	return res
}

// SerializerType returns the unique ID used to
// serialize the Forest type with the serializer
// package.
func (f *Forest) SerializerType() string {
	return serializerTypeForest
}

// Serialize serializes the Markov model.
func (f *Forest) Serialize() ([]byte, error) {
	var res bytes.Buffer
	enc := gob.NewEncoder(&res)
	if err := enc.Encode(f.forest); err != nil {
		return nil, err
	}
	return res.Bytes(), nil
}

type forestSample struct {
	textDoc    string
	index      int
	endOfField bool
}

func (f *forestSample) Attr(a idtrees.Attr) idtrees.Val {
	idx := a.(int) + f.index
	if idx < 0 || idx >= len(f.textDoc) {
		return 0
	}
	return f.textDoc[idx]
}

func (f *forestSample) Class() idtrees.Class {
	return f.endOfField
}
