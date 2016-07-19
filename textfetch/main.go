// Command textfetch fetches Wikipedia articles as
// plain text and saves them to a file.
package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/yhat/scrape"
	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage: textfetch <dest dir> <article count> <start URL>")
		os.Exit(1)
	}

	count, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid article count:", os.Args[2])
		os.Exit(1)
	}

	startURL := os.Args[3]
	samples, err := FetchSamples(startURL, count)

	if err != nil {
		fmt.Fprintln(os.Stderr, "Error fetching articles:", err)
		os.Exit(1)
	}

	outputPath := os.Args[1]
	if _, err := os.Stat(outputPath); err != nil && os.IsNotExist(err) {
		if err := os.Mkdir(outputPath, 0755); err != nil {
			fmt.Fprintln(os.Stderr, "Error making output directory:", err)
			os.Exit(1)
		}
	}

	for url, body := range samples {
		lastIdx := strings.LastIndex(url, "/")
		pageName := url[lastIdx+1:]
		destPath := filepath.Join(outputPath, pageName+".txt")
		if err := ioutil.WriteFile(destPath, []byte(body), 0755); err != nil {
			fmt.Fprintln(os.Stderr, "Error writing output file:", err)
			os.Exit(1)
		}
	}
}

// FetchSamples fetches Wikipedia samples by starting
// from the given page and randomly following links.
// The resulting map maps article URLs to their bodies.
func FetchSamples(startPage string, numPages int) (map[string]string, error) {
	res := map[string]string{}
	err := fetchSamplesRecursive(startPage, &numPages, res)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func fetchSamplesRecursive(page string, remaining *int, output map[string]string) error {
	if _, ok := output[page]; ok {
		return nil
	}

	log.Printf("Fetching: %s (%d remaining)", page, *remaining)

	res, err := http.Get(page)
	if err != nil {
		return err
	}
	root, err := html.Parse(res.Body)
	res.Body.Close()
	if err != nil {
		return err
	}

	body := articleBody(root)
	output[page] = body

	*remaining--
	if *remaining == 0 {
		return nil
	}

	links, err := articleLinks(page, root)
	if err != nil {
		return err
	}
	randomizeLinkOrder(links)

	for _, link := range links {
		err := fetchSamplesRecursive(link, remaining, output)
		if err != nil {
			return err
		}
		if *remaining == 0 {
			return nil
		}
	}

	return nil
}

func articleBody(root *html.Node) string {
	paragraphs := scrape.FindAll(root, scrape.ByTag(atom.P))
	var blocks []string
	for _, p := range paragraphs {
		blocks = append(blocks, elementString(p))
	}
	return strings.Join(blocks, "\n\n")
}

func articleLinks(page string, root *html.Node) ([]string, error) {
	wikiIdx := strings.Index(page, "/wiki/")
	if wikiIdx < 0 {
		return nil, errors.New("invalid page URL: " + page)
	}
	wikiRoot := page[:wikiIdx]
	links := scrape.FindAll(root, scrape.ByTag(atom.A))

	var res []string
	for _, link := range links {
		href := scrape.Attr(link, "href")
		if strings.HasPrefix(href, "/wiki") {
			res = append(res, wikiRoot+href)
		}
	}
	return res, nil
}

func randomizeLinkOrder(links []string) {
	for i := 0; i < len(links)-1; i++ {
		idx := rand.Intn(len(links)-i) + i
		links[i], links[idx] = links[idx], links[i]
	}
}

func elementString(p *html.Node) string {
	if p.DataAtom == atom.Sup {
		return ""
	}
	if p.Type == html.TextNode {
		return p.Data
	} else {
		var res []string
		child := p.FirstChild
		for child != nil {
			res = append(res, elementString(child))
			child = child.NextSibling
		}
		return strings.Join(res, "")
	}
}
