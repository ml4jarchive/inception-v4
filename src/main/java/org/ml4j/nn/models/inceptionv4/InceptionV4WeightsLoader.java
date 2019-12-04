package org.ml4j.nn.models.inceptionv4;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.util.SerializationHelper;

public class InceptionV4WeightsLoader {
	
	private static final InceptionV4WeightsLoader INSTANCE = new InceptionV4WeightsLoader();
	
	private SerializationHelper serializationHelper;
	
	public InceptionV4WeightsLoader() {
		this.serializationHelper = new SerializationHelper(InceptionV4WeightsLoader.class.getClassLoader(), "inceptionv4javaweights");
	}
	
	private float[] deserializeWeights(String name) {
		return serializationHelper.deserialize(float[].class, name);
	}
	
	public static Matrix getDenseLayerWeights(MatrixFactory matrixFactory, String name, int rows, int columns)  {
		float[] weights =  INSTANCE.deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(rows, columns, weights);
	}

	public static Matrix getConvolutionalLayerWeights(MatrixFactory matrixFactory, String name, int width, int height, int inputDepth, int outputDepth) {
		float[] weights =  INSTANCE.deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(outputDepth, width * height * inputDepth, weights);
	}
	
	public static Matrix getBatchNormLayerWeights(MatrixFactory matrixFactory, String name, int width, int height, int inputDepth, int outputDepth) {
		float[] weights =  INSTANCE.deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth * width * height, 1, weights);
	}
}
