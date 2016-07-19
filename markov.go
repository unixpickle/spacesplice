package spacesplice

import (
	"encoding/json"
	"io/ioutil"
	"path/filepath"
	"strings"
)

var markovSingleLetterWords = []string{"a", "I"}

const maxWordLen = 20

// Markov is a Splicer that uses a simple Markov chain
// to insert spaces into a piece of text.
type Markov struct {
	TotalCount int

	// RawCounts stores the number of occurrences of each
	// word in the training corpus. It can be thought of
	// as a frequency map of underlying words.
	RawCounts map[string]int

	// Table maps words to tables indicating the likelihood
	// of each possible following word. It can be thought of
	// as the frequency map of words given that a different
	// word was just seen.
	Table map[string]map[string]int

	// TableCounts stores the total number of table entries
	// for each word (i.e. it is the sum of the values for
	// each key in each table). This may be different than
	// RawCounts, since RawCounts includes occurrences at
	// the very ends of training documents.
	TableCounts map[string]int
}

// TrainMarkov trains a Markov model on a directory full
// of sample text files.
func TrainMarkov(corpusDir string) (*Markov, error) {
	res := &Markov{
		RawCounts:   map[string]int{},
		Table:       map[string]map[string]int{},
		TableCounts: map[string]int{},
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

// CondProb returns the conditional probability of a
// word given the previous word. It returns 0 if the
// word has never been seen after the previous word.
// It returns 1 if the word is "".
func (m *Markov) CondProb(previous, word string) float64 {
	if word == "" {
		return 1
	}
	prevMap := m.Table[previous]
	if prevMap == nil {
		return 0
	}
	nextCount := m.TableCounts[previous]
	return float64(prevMap[word]) / float64(nextCount)
}

// Prob returns the unconditional probability of a word.
// It returns 0 if the word has never been seen before.
// It returns 1 if the word is "".
func (m *Markov) Prob(word string) float64 {
	if word == "" {
		return 1
	}
	return float64(m.RawCounts[word]) / float64(m.TotalCount)
}

// BestField returns the most likely next field in the
// string given the previous field.
//
// The previous field "" means this is the first field.
func (m *Markov) BestField(previous string, str string) string {
	var bestCond float64
	var bestCondStr string
	var bestUncond float64
	var bestUncondStr string

	followingPairs(str, func(w1, w2 string) {
		condProb := m.CondProb(previous, w1) * m.CondProb(w1, w2)
		if condProb > bestCond {
			bestCond = condProb
			bestCondStr = w1
		}
		uncondProb := m.Prob(w1) * m.Prob(w2)
		if uncondProb > bestUncond {
			bestUncond = uncondProb
			bestUncondStr = w1
		}
	})

	if bestCond > 0 {
		return bestCondStr
	} else if bestUncond > 0 {
		return bestUncondStr
	}

	followingWords(str, func(w1 string) {
		p := m.Prob(w1)
		if p >= bestUncond {
			bestUncond = p
			bestUncondStr = w1
		}
	})
	return bestUncondStr
}

// Fields uses the Markov model to split the spaceless
// text into fields (i.e. words)
func (m *Markov) Fields(text string) []string {
	parts := strings.Fields(text)
	var res []string
	for _, part := range parts {
		var lastWord string
		for i := 0; i < len(part); i += len(lastWord) {
			lastWord = m.BestField(lastWord, part[i:])
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
	var lastWord string
	for _, field := range sampleFields {
		if len(field) == 1 {
			var allowed bool
			for _, x := range markovSingleLetterWords {
				if x == field {
					allowed = true
				}
			}
			if !allowed {
				continue
			}
		}
		m.TotalCount++
		m.RawCounts[field]++
		m.TableCounts[lastWord]++
		subTable := m.Table[lastWord]
		if subTable == nil {
			subTable = map[string]int{}
			m.Table[lastWord] = subTable
		}
		subTable[field]++
		lastWord = field
	}
}

func followingPairs(str string, f func(w1, w2 string)) {
	followingWords(str, func(w1 string) {
		if len(w1) == len(str) {
			f(w1, "")
		} else {
			followingWords(str[len(w1):], func(w2 string) {
				f(w1, w2)
			})
		}
	})
}

func followingWords(str string, f func(string)) {
	for l := 1; l <= maxWordLen && l <= len(str); l++ {
		f(str[:l])
	}
}
