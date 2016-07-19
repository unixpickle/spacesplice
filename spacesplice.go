// Package spacesplice provides APIs for adding spaces
// to text whose whitespace has been removed.
//
// Consider the text "helloworld". An English speaker
// can easily tell that this says "hello world". The
// question is, how do you make a computer do the same
// thing reliably?
package spacesplice

import "github.com/unixpickle/serializer"

// A Fielder is anything capable of splitting a piece
// of text into "fields", which are words which would
// normally be separated by spaces.
type Fielder interface {
	serializer.Serializer

	// Fields attempts to intelligently split the body
	// of text into fields (i.e. words) even though the
	// body of text has no whitespace.
	//
	// This is named after the function strings.Fields,
	// which splits a string into blocks via whitespace.
	Fields(text string) []string
}

// TrainFunc is any function which trains a Fielder on
// a directory of text samples.
type TrainFunc func(corpusDir string) (Fielder, error)

// Trainers maps the names of various text prediction
// models to TrainFuncs for those models.
var Trainers = map[string]TrainFunc{
	"markov": func(corpusDir string) (Fielder, error) {
		return TrainMarkov(corpusDir)
	},
	"dict": func(corpusDir string) (Fielder, error) {
		return TrainDictionary(corpusDir)
	},
}
