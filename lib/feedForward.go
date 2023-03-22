package lib

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type FeedForward struct {
	w1, w2, b1, b2 *gorgonia.Node
}

func NewFeedForward(g *gorgonia.ExprGraph, dModel int) (*FeedForward, error) {
	dff := 4 * dModel
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(dModel, dff), gorgonia.WithName("W1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(dff, dModel), gorgonia.WithName("W2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	b1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, dff), gorgonia.WithName("b1"), gorgonia.WithInit(gorgonia.Zeroes()))
	b2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, dModel), gorgonia.WithName("b2"), gorgonia.WithInit(gorgonia.Zeroes()))

	return &FeedForward{
		w1: w1,
		w2: w2,
		b1: b1,
		b2: b2,
	}, nil
}

func (ff *FeedForward) Forward(g *gorgonia.ExprGraph, x *gorgonia.Node) (output *gorgonia.Node, err error) {
	// Linear layer 1
	z1, err := gorgonia.Mul(x, ff.w1)
	if err != nil {
		return nil, err
	}

	z1, err = gorgonia.BroadcastAdd(z1, ff.b1, nil, []byte{0})
	if err != nil {
		return nil, err
	}

	// Activation function (ReLU)
	a1, err := gorgonia.Rectify(z1)
	if err != nil {
		return nil, err
	}

	// Linear layer 2
	z2, err := gorgonia.Mul(a1, ff.w2)
	if err != nil {
		return nil, err
	}

	z2, err = gorgonia.BroadcastAdd(z2, ff.b2, nil, []byte{0})
	if err != nil {
		return nil, err
	}

	return z2, nil
}
