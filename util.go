package spacesplice

import (
	"io/ioutil"
	"path/filepath"
	"strings"
)

func ReadSamples(dir string, f func(d []byte)) error {
	contents, err := ioutil.ReadDir(dir)
	if err != nil {
		return err
	}

	for _, fileInfo := range contents {
		if strings.HasPrefix(fileInfo.Name(), ".") {
			continue
		}
		path := filepath.Join(dir, fileInfo.Name())
		sampleBody, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}
		f(sampleBody)
	}
	return nil
}
