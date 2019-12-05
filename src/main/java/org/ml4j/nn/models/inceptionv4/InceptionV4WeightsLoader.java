package org.ml4j.nn.models.inceptionv4;

import java.io.IOException;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

public interface InceptionV4WeightsLoader {
	
	Matrix getDenseLayerWeights(MatrixFactory matrixFactory, String name, int rows, int columns) throws IOException;

	Matrix getConvolutionalLayerWeights(MatrixFactory matrixFactory, String name, int width, int height, int inputDepth, int outputDepth) throws IOException;
	
	Matrix getBatchNormLayerWeights(MatrixFactory matrixFactory, String name, int inputDepth) throws IOException;
}
