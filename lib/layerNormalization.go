package lib

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type LayerNormalization struct {
	gamma, beta *gorgonia.Node
}

func NewLayerNormalization(g *gorgonia.ExprGraph, dModel int) *LayerNormalization {
	gamma := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, dModel), gorgonia.WithName("gamma"), gorgonia.WithInit(gorgonia.Ones()))
	beta := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, dModel), gorgonia.WithName("beta"), gorgonia.WithInit(gorgonia.Zeroes()))

	return &LayerNormalization{
		gamma: gamma,
		beta:  beta,
	}
}
