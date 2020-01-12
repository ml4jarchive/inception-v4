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

import java.io.IOException;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4Definition;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilderFactory;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.models.inceptionv4.InceptionV4Factory;
import org.ml4j.nn.models.inceptionv4.InceptionV4Labels;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default factory for an Inception V4 Network.
 * 
 */
public class DefaultInceptionV4Factory implements InceptionV4Factory {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultInceptionV4Factory.class);
		
	private Components3DGraphBuilderFactory<DefaultChainableDirectedComponent<?, ?>> components3DGraphBuilderFactory;
	
	private SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory;
	
	private InceptionV4WeightsLoader weightsLoader;
		
	private InceptionV4Labels labels;
	
	/**
	 * Creates the default pre-trained InceptionV4 Networks
	 * 
	 * @param components3DGraphBuilderFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param classLoader
	 * @throws IOException
	 */
	public DefaultInceptionV4Factory(
			Components3DGraphBuilderFactory<DefaultChainableDirectedComponent<?, ?>> components3DGraphBuilderFactory, 
			MatrixFactory matrixFactory, 
			SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory, ClassLoader classLoader) throws IOException {
		this(components3DGraphBuilderFactory, supervisedFeedForwardNeuralNetworkFactory, 
				new PretrainedInceptionV4WeightsLoaderImpl(classLoader, matrixFactory), new DefaultInceptionV4Labels(classLoader));
	}
	
	/**
	 * Creates InceptionV4 Networks with custom weights and labels
	 * 
	 * @param components3DGraphBuilderFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param weightsLoader
	 * @param labels
	 */
	public DefaultInceptionV4Factory(
			Components3DGraphBuilderFactory<DefaultChainableDirectedComponent<?, ?>> components3DGraphBuilderFactory,
			SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory, InceptionV4WeightsLoader weightsLoader, InceptionV4Labels labels) {
		this.components3DGraphBuilderFactory = components3DGraphBuilderFactory;
		this.supervisedFeedForwardNeuralNetworkFactory = supervisedFeedForwardNeuralNetworkFactory;
		this.weightsLoader = weightsLoader;
		this.labels = labels;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork createInceptionV4(FeedForwardNeuralNetworkContext trainingContext)
			throws IOException {
		
		LOGGER.info("Creating Inception V4 Network...");

		// Obtain the InceptionV4Definition from neural-network-architectures
		InceptionV4Definition inceptionV4Definition = new InceptionV4Definition(weightsLoader);
		
		// Create a graph builder, starting with the input neurons for the InceptionV4Definition.
		InitialComponents3DGraphBuilder<DefaultChainableDirectedComponent<?, ?>> graphBuilder = 
				components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(inceptionV4Definition.getInputNeurons(), 
						trainingContext.getDirectedComponentsContext());
		
		// Create the component graph from the definition and graph builder, and wrap with a supervised feed forward neural network.
		return supervisedFeedForwardNeuralNetworkFactory
				.createSupervisedFeedForwardNeuralNetwork(inceptionV4Definition.createComponentGraph(graphBuilder).getComponents());
	}

	@Override
	public InceptionV4Labels createInceptionV4Labels() throws IOException {
		return labels;
	}
}
