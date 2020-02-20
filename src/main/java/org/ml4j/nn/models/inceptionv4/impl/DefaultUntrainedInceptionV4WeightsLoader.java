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

import java.util.Arrays;

import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.BiasVectorImpl;
import org.ml4j.nn.axons.FeaturesVector;
import org.ml4j.nn.axons.FeaturesVectorFormatImpl;
import org.ml4j.nn.axons.FeaturesVectorImpl;
import org.ml4j.nn.axons.FeaturesVectorOrientation;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;

/**
 * @author Michael Lavelle
 */
public class DefaultUntrainedInceptionV4WeightsLoader implements InceptionV4WeightsLoader {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public WeightsMatrix getDenseLayerWeights(String name, int rows, int columns) {
		return new WeightsMatrixImpl(null,
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), 
						Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
	}

	public WeightsMatrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth) {
		if (width == 1 && height == 1) {
			return new WeightsMatrixImpl(null,
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		} else {
			return new WeightsMatrixImpl(null,
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		}
	}

	public WeightsMatrix getBatchNormLayerWeights(String name, int outputDepth) {
		return new WeightsMatrixImpl(null,
				new WeightsFormatImpl(Arrays.asList(
						Dimension.INPUT_DEPTH), 
						Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
	}

	@Override
	public BiasVector getDenseLayerBiases(String name, int rows, int columns) {
		return null;
	}

	@Override
	public BiasVector getBatchNormLayerBiases(String name, int outputDepth) {
		return new BiasVectorImpl(null, new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
				FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

	@Override
	public FeaturesVector getBatchNormLayerMean(String name, int outputDepth) {
		return new FeaturesVectorImpl(null, new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
				FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

	@Override
	public FeaturesVector getBatchNormLayerVariance(String name, int outputDepth) {
		return new FeaturesVectorImpl(null, new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
				FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

}
