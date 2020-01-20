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

import org.ml4j.Matrix;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;

/**
 * @author Michael Lavelle
 */
public class DefaultUntrainedInceptionV4WeightsLoader implements InceptionV4WeightsLoader {

	@Override
	public Matrix getDenseLayerWeights(String name, int rows, int columns) {
		return null;
	}

	@Override
	public Matrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth) {
		return null;
	}

	@Override
	public Matrix getBatchNormLayerWeights(String name, int inputDepth) {
		return null;
	}

}
