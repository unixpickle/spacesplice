package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/spacesplice"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: addspaces <model file> <phrase>")
		os.Exit(1)
	}
	modelData, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read model:", err)
		os.Exit(1)
	}
	model, err := serializer.DeserializeWithType(modelData)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to deserialize model:", err)
		os.Exit(1)
	}
	fielder, ok := model.(spacesplice.Fielder)
	if !ok {
		fmt.Fprintf(os.Stderr, "Unexpected deserialized type: %T\n", model)
		os.Exit(1)
	}
	fields := fielder.Fields(os.Args[2])
	fmt.Println(strings.Join(fields, " "))
}
