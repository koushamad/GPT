package cmd

import (
	"fmt"
	"github.com/koushamad/gpt/lib"
	"gorgonia.org/gorgonia"
	"log"

	"github.com/spf13/cobra"
)

// initCmd represents the init command
var initCmd = &cobra.Command{
	Use:   "init",
	Short: "GPT Transformer",
	Long:  `GPT Transformer with Gorgonia.`,
	Run: func(cmd *cobra.Command, args []string) {
		g := gorgonia.NewGraph()
		nHeads := 8
		dModel := 512
		seqLen := 128

		// Create MultiHeadAttention
		mha, err := lib.NewMultiHeadAttention(g, nHeads, dModel)
		if err != nil {
			log.Fatalf("Failed to create MultiHeadAttention: %v", err)
		}

		// Create PositionalEncoding
		posEnc, err := lib.PositionalEncoding(g, seqLen, dModel)
		if err != nil {
			log.Fatalf("Failed to create positional encoding: %v", err)
		}

		// Create LayerNormalization
		layerNorm := lib.NewLayerNormalization(g, dModel)

		// Create FeedForward
		ffn, err := lib.NewFeedForward(g, dModel)
		if err != nil {
			log.Fatalf("Failed to create feed-forward layer: %v", err)
		}

		// Print model components
		fmt.Printf("MultiHeadAttention created with %d heads and dModel %d\n", nHeads, dModel)
		fmt.Printf("Positional encoding created with seqLen %d and dModel %d\n", seqLen, dModel)
		fmt.Println("Layer normalization created")
		fmt.Println("Feed-forward layer created")

		fmt.Println(mha)
		fmt.Println(posEnc)
		fmt.Println(layerNorm)
		fmt.Println(ffn)
	},
}

func init() {
	rootCmd.AddCommand(initCmd)
}
