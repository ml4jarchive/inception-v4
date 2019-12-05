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

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.models.inceptionv4.InceptionV4WeightsLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Michael Lavelle
 */
public class PretrainedInceptionV4WeightsLoaderImpl implements InceptionV4WeightsLoader {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultInceptionV4Factory.class);
		
	private ClassLoader classLoader;
	private long uid;
	
	public PretrainedInceptionV4WeightsLoaderImpl(ClassLoader classLoader) {
		this.uid = ObjectStreamClass.lookup(float[].class).getSerialVersionUID();
		this.classLoader = classLoader;
	}
	
	public static PretrainedInceptionV4WeightsLoaderImpl getLoader(ClassLoader classLoader) {
		return new PretrainedInceptionV4WeightsLoaderImpl(classLoader);
	}
	
	private float[] deserializeWeights(String name) throws IOException {
		LOGGER.debug("Derializing weights:" + name);
		try {
			return deserialize(float[].class, "inceptionv4javaweights", uid, name);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
	
	public Matrix getDenseLayerWeights(MatrixFactory matrixFactory, String name, int rows, int columns) throws IOException  {
		float[] weights =  deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(rows, columns, weights);
	}

	public Matrix getConvolutionalLayerWeights(MatrixFactory matrixFactory, String name, int width, int height, int inputDepth, int outputDepth) throws IOException {
		float[] weights =  deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(outputDepth, width * height * inputDepth, weights);
	}
	
	public Matrix getBatchNormLayerWeights(MatrixFactory matrixFactory, String name, int inputDepth) throws IOException {
		float[] weights =  deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights);
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
}
