package org.ml4j.nn.models.inceptionv4;

import java.io.IOException;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

/**
 * Helper to aid the loading of pretrained Inception V4 Weights.
 * 
 * @author Michael Lavelle
 *
 */
public class InceptionV4WeightsLoader {
	
	private InceptionV4WeightsLoader() {}
	
	public static Matrix getDenseLayerWeights(MatrixFactory matrixFactory, String name, int rows, int columns) throws IOException {
		return null;
	}

	public static Matrix getConvolutionalLayerWeights(MatrixFactory matrixFactory, String name, int width, int height, int inputDepth, int outputDepth) throws IOException {
		return null;
	}
	
	public static Matrix getBatchNormLayerWeights(MatrixFactory matrixFactory, String name, int width, int height, int inputDepth, int outputDepth) throws IOException {
		return null;
	}
}
