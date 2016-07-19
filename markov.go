package spacesplice

import (
	"encoding/json"
	"io/ioutil"
	"path/filepath"
	"strings"
)

const maxWordLen = 20

// Markov is a Splicer that uses a simple Markov chain
// to insert spaces into a piece of text.
type Markov struct {
	// RawCounts stores the number of occurrences of each
	// word in the training corpus. It can be thought of
	// as a frequency map of underlying words.
	RawCounts map[string]int

	// Table maps words to tables indicating the likelihood
	// of each possible following word. It can be thought of
	// as the frequency map of words given that a different
	// word was just seen.
	Table map[string]map[string]int
}

// TrainMarkov trains a Markov model on a directory full
// of sample text files.
func TrainMarkov(corpusDir string) (*Markov, error) {
	res := &Markov{
		RawCounts: map[string]int{},
		Table:     map[string]map[string]int{},
	}
	contents, err := ioutil.ReadDir(corpusDir)
	if err != nil {
		return nil, err
	}
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
		trainMarkovSample(res, fields)
	}
	return res, nil
}

// DeserializeMarkov deserializes a Markov model which
// was serialized with Markov.Serialize().
func DeserializeMarkov(d []byte) (*Markov, error) {
	var res Markov
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// MostLikely returns the most likely word to be
// observed, given the previous word. If there was
// no previous word, it should be the empty string.
func (m *Markov) MostLikely(previous string, options []string) string {
	var bestCount, bestIdx int
	for i, option := range options {
		if nextMap := m.Table[previous]; nextMap != nil {
			count := m.Table[previous][option]
			if count > bestCount {
				bestCount = count
				bestIdx = i
			}
		}
	}
	if bestCount > 0 {
		return options[bestIdx]
	}

	for i, option := range options {
		count := m.RawCounts[option]
		if count > bestCount {
			bestCount = count
			bestIdx = i
		}
	}
	return options[bestIdx]
}

// Fields uses the Markov model to split the spaceless
// text into fields (i.e. words)
func (m *Markov) Fields(text string) []string {
	parts := strings.Fields(text)
	var res []string
	for _, part := range parts {
		var lastWord string
		for i := 0; i < len(part); i += len(lastWord) {
			followingWords := make([]string, maxWordLen)
			for l := 1; l <= maxWordLen && l+i <= len(part); l++ {
				followingWords[l-1] = part[i : i+l]
			}
			lastWord = m.MostLikely(lastWord, followingWords)
			res = append(res, lastWord)

		}
	}
	return res
}

// SerializerType returns the unique ID used to
// serialize the Markov type with the serializer
// package.
func (m *Markov) SerializerType() string {
	return serializerTypeMarkov
}

// Serialize serializes the Markov model.
func (m *Markov) Serialize() ([]byte, error) {
	return json.Marshal(m)
}

func trainMarkovSample(m *Markov, sampleFields []string) {
	for i, field := range sampleFields {
		m.RawCounts[field]++
		var lastWord string
		if i > 0 {
			lastWord = sampleFields[i-1]
		}
		subTable := m.Table[lastWord]
		if subTable == nil {
			subTable = map[string]int{}
			m.Table[lastWord] = subTable
		}
		subTable[field]++
	}
}
