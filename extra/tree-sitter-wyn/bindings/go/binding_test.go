package tree_sitter_wyn_test

import (
	"testing"

	tree_sitter "github.com/smacker/go-tree-sitter"
	"github.com/tree-sitter/tree-sitter-wyn"
)

func TestCanLoadGrammar(t *testing.T) {
	language := tree_sitter.NewLanguage(tree_sitter_wyn.Language())
	if language == nil {
		t.Errorf("Error loading Wyn grammar")
	}
}
