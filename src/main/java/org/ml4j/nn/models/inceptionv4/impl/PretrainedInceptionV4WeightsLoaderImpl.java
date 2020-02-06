/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.models.inceptionv4.impl;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;
import java.io.Serializable;
import java.util.Arrays;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.axons.BiasFormatImpl;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.BiasMatrixImpl;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Michael Lavelle
 */
public class PretrainedInceptionV4WeightsLoaderImpl implements InceptionV4WeightsLoader {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultInceptionV4Factory.class);

	private MatrixFactory matrixFactory;
	private ClassLoader classLoader;
	private long uid;

	public PretrainedInceptionV4WeightsLoaderImpl(ClassLoader classLoader, MatrixFactory matrixFactory) {
		this.uid = ObjectStreamClass.lookup(float[].class).getSerialVersionUID();
		this.classLoader = classLoader;
		this.matrixFactory = matrixFactory;
	}

	public static PretrainedInceptionV4WeightsLoaderImpl getLoader(MatrixFactory matrixFactory,
			ClassLoader classLoader) {
		return new PretrainedInceptionV4WeightsLoaderImpl(classLoader, matrixFactory);
	}

	private float[] deserializeWeights(String name) {
		LOGGER.debug("Derializing weights:" + name);
		try {
			return deserialize(float[].class, "inceptionv4javaweights", uid, name);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public WeightsMatrix getDenseLayerWeights(String name, int rows, int columns) {
		float[] weights = deserializeWeights(name);
		return new WeightsMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(rows, columns, weights),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), 
						Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
	}

	public WeightsMatrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth) {
		float[] weights = deserializeWeights(name);
		boolean oneByOneConvolution = width == 1 && height == 1;
		if (oneByOneConvolution) {
			return new WeightsMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(outputDepth, width * height * inputDepth, weights),
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		} else {
			return new WeightsMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(outputDepth, width * height * inputDepth, weights),
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		}
	}

	public WeightsMatrix getBatchNormLayerWeights(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return new WeightsMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights),
				new WeightsFormatImpl(Arrays.asList(
						Dimension.INPUT_DEPTH), 
						Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
	}

	@SuppressWarnings("unchecked")
	public <S extends Serializable> S deserialize(Class<S> clazz, String path, long uid, String id)
			throws IOException, ClassNotFoundException {

		if (classLoader == null) {
			try (InputStream is = new FileInputStream(path + "/" + clazz.getName() + "/" + uid + "/" + id + ".ser")) {
				try (ObjectInputStream ois = new ObjectInputStream(is)) {
					return (S) ois.readObject();
				}
			}
		} else {
			try (InputStream is = classLoader
					.getResourceAsStream(path + "/" + clazz.getName() + "/" + uid + "/" + id + ".ser")) {
				try (ObjectInputStream ois = new ObjectInputStream(is)) {
					return (S) ois.readObject();
				}
			}

		}
	}

	@Override
	public BiasMatrix getDenseLayerBiases(String name, int rows, int columns) {
		float[] weights = deserializeWeights(name);
		return new BiasMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(rows, columns, weights));
	}

	@Override
	public BiasMatrix getBatchNormLayerBiases(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return new BiasMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights),
				new BiasFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH), Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
	}

	@Override
	public Matrix getBatchNormLayerMean(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights);
	}

	@Override
	public Matrix getBatchNormLayerVariance(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights);
	}
}
