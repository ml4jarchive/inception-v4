package org.ml4j.nn.models.inceptionv4.impl;

import java.io.IOException;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.models.inceptionv4.InceptionV4WeightsLoader;

public class DefaultUntrainedInceptionV4WeightsLoader implements InceptionV4WeightsLoader {

	@Override
	public Matrix getDenseLayerWeights(MatrixFactory matrixFactory, String name, int rows, int columns)
			throws IOException {
		return null;
	}

	@Override
	public Matrix getConvolutionalLayerWeights(MatrixFactory matrixFactory, String name, int width, int height,
			int inputDepth, int outputDepth) throws IOException {
		return null;
	}

	@Override
	public Matrix getBatchNormLayerWeights(MatrixFactory matrixFactory, String name,
			int inputDepth) throws IOException {
		return null;
	}

}
