package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/spacesplice"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: addspaces <model file>")
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
	for str := range inputStream() {
		fields := fielder.Fields(str)
		fmt.Println(strings.Join(fields, " "))
	}
}

func inputStream() <-chan string {
	res := make(chan string)
	go func() {
		var buf [1]byte
		var stringBuf bytes.Buffer
		for {
			if n, err := os.Stdin.Read(buf[:]); err != nil {
				if stringBuf.Len() > 0 {
					res <- stringBuf.String()
				}
				close(res)
				return
			} else if n > 0 {
				if buf[0] == '\n' {
					res <- stringBuf.String()
					stringBuf.Reset()
				} else {
					stringBuf.WriteByte(buf[0])
				}
			}
		}
	}()
	return res
}
