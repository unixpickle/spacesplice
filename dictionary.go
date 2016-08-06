package spacesplice

import (
	"sort"
	"strings"
)

const dictionaryMaxWord = 20

// A Dictionary is a Fielder which operates by picking
// the longest possible dictionary word.
type Dictionary struct {
	// Words is an alphabetically-sorted list of words.
	Words []string
}

// TrainDictionary trains a Dictionary by reading all
// of the files in the given directory and extracting
// their words.
func TrainDictionary(corpusDir string) (*Dictionary, error) {
	wordMap := map[string]bool{}
	err := ReadSamples(corpusDir, func(sampleBody []byte) {
		for _, field := range strings.Fields(string(sampleBody)) {
			wordMap[field] = true
		}
	})
	if err != nil {
		return nil, err
	}
	words := make([]string, 0, len(wordMap))
	for word := range wordMap {
		words = append(words, word)
	}
	sort.Strings(words)
	return &Dictionary{Words: words}, nil
}

// DeserializeDictionary deserializes a dictionary that
// was serialized with Dictionary.Serialize().
func DeserializeDictionary(d []byte) (*Dictionary, error) {
	return &Dictionary{Words: strings.Fields(string(d))}, nil
}

// Contains returns true if x is in the dictionary.
func (d *Dictionary) Contains(x string) bool {
	idx := sort.SearchStrings(d.Words, x)
	if idx == len(d.Words) {
		return false
	}
	return d.Words[idx] == x
}

// Fields uses the dictionary to split the text into
// fields (i.e. words).
func (d *Dictionary) Fields(text string) []string {
	var res []string
	for _, part := range strings.Fields(text) {
		for len(part) > 0 {
			longestWord := part[:1]
			for l := 2; l <= maxWordLen && l <= len(part); l++ {
				if d.Contains(part[:l]) {
					longestWord = part[:l]
				}
			}
			res = append(res, longestWord)
			part = part[len(longestWord):]
		}
	}
	return res
}

// SerializerType returns the unique ID used to
// serialize the Dictionary type with the serializer
// package.
func (d *Dictionary) SerializerType() string {
	return serializerTypeDictionary
}

// Serialize serializes the Dictionary.
func (d *Dictionary) Serialize() ([]byte, error) {
	return []byte(strings.Join(d.Words, "\n")), nil
}
