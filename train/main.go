package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"strings"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/spacesplice"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage: train <model> <corpus dir> <output file>")
		printModels()
		os.Exit(1)
	}

	trainer, ok := spacesplice.Trainers[os.Args[1]]
	if !ok {
		fmt.Fprintln(os.Stderr, "Unknown model:", os.Args[1])
		os.Exit(1)
	}

	res, err := trainer(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error training model:", err)
		os.Exit(1)
	}

	serialized, err := serializer.SerializeWithType(res)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to serialize:", err)
		os.Exit(1)
	}

	if err := ioutil.WriteFile(os.Args[3], serialized, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}
}

func printModels() {
	var names []string
	for name := range spacesplice.Trainers {
		names = append(names, name)
	}
	sort.Strings(names)
	fmt.Fprintln(os.Stderr, "Available models:", strings.Join(names, ", "))
}
