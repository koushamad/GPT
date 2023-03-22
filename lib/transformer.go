package lib

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math"
)

func PositionalEncoding(g *gorgonia.ExprGraph, seqLen, dModel int) (posEnc *gorgonia.Node, err error) {
	posEncTensor := tensor.New(tensor.WithShape(seqLen, dModel), tensor.WithBacking(make([]float32, seqLen*dModel)))

	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < dModel; i += 2 {
			// Calculate the sine and cosine values
			angle := float32(pos) / float32(math.Pow(10000, float64(i)/float64(dModel)))
			sine := float32(math.Sin(float64(angle)))
			cosine := float32(math.Cos(float64(angle)))

			// Assign the sine and cosine values to the tensor
			posEncTensor.SetAt(sine, pos, i)
			posEncTensor.SetAt(cosine, pos, i+1)
		}
	}

	posEnc = gorgonia.NodeFromAny(g, posEncTensor, gorgonia.WithName("PositionalEncoding"))
	return
}
