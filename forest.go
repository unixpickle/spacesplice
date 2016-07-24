package spacesplice

import (
	"bytes"
	"encoding/gob"
	"io/ioutil"
	"log"
	"path/filepath"
	"strings"

	"github.com/unixpickle/weakai/idtrees"
)

const (
	forestSize         = 500
	forestFeatureCount = 6
	forestSampleCount  = 7000
	forestFracCutoff   = 0.22
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

	log.Println("Building samples...")

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

	log.Println("Creating forest...")

	var allAttrs []idtrees.Attr
	for i := -7; i <= 3; i++ {
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
	buf := bytes.NewBuffer(d)
	dec := gob.NewDecoder(buf)
	var res idtrees.Forest
	if err := dec.Decode(&res); err != nil {
		return nil, err
	}
	return &Forest{forest: res}, nil
}

// Fields uses the forest to split the spaceless text
// into fields (i.e. words).
func (f *Forest) Fields(text string) []string {
	parts := strings.Fields(text)
	var res []string
	for _, part := range parts {
		var field bytes.Buffer
		for i := 0; i < len(part); i++ {
			sample := &forestSample{
				textDoc: part,
				index:   i,
			}
			field.WriteByte(part[i])
			probs := f.forest.Classify(sample)
			if probs[true] >= probs[false]*forestFracCutoff {
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
