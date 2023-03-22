package lib

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MultiHeadAttention struct {
	NHeads         int
	dK             int
	wq, wk, wv, wo *gorgonia.Node
}

func NewMultiHeadAttention(g *gorgonia.ExprGraph, nHeads, dModel int) (*MultiHeadAttention, error) {
	dK := dModel / nHeads
	wq := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(dModel, dK), gorgonia.WithName("WQ"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	wk := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(dModel, dK), gorgonia.WithName("WK"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	wv := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(dModel, dK), gorgonia.WithName("WV"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	wo := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(dK*nHeads, dModel), gorgonia.WithName("WO"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	return &MultiHeadAttention{
		NHeads: nHeads,
		dK:     dK,
		wq:     wq,
		wk:     wk,
		wv:     wv,
		wo:     wo,
	}, nil
}
