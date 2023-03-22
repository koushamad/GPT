package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "gpt",
	Short: "GPT is an implementation of the GPT model",
	Long: `A Golang implementation of the GPT model
		based on Gorgonia and Cobra.`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Welcome to the GPT implementation")
		fmt.Println("Use the 'gpt train' command to train the model")
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
