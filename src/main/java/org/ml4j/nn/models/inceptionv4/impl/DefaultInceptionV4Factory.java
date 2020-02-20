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
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WithoutTailDefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.UntrainedTailInceptionV4Definition;
import org.ml4j.nn.architectures.inception.inceptionv4.modules.InceptionV4CustomTailDefinition;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.models.inceptionv4.InceptionV4Factory;
import org.ml4j.nn.models.inceptionv4.InceptionV4Labels;
import org.ml4j.nn.sessions.factories.DefaultSessionFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Default factory for an Inception V4 Network.
 * 
 */
public class DefaultInceptionV4Factory implements InceptionV4Factory {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultInceptionV4Factory.class);

	private DefaultSessionFactory sessionFactory;
	
	private InceptionV4WeightsLoader weightsLoader;

	private InceptionV4Labels labels;

	/**
	 * Creates the default pre-trained InceptionV4 Networks
	 * 
	 * @param sessionFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param classLoader
	 * @throws IOException
	 */
	public DefaultInceptionV4Factory(DefaultSessionFactory sessionFactory,
			MatrixFactory matrixFactory,
			ClassLoader classLoader) throws IOException {
		this(sessionFactory,
				new PretrainedInceptionV4WeightsLoaderImpl(classLoader, matrixFactory),
				new DefaultInceptionV4Labels(classLoader));
	}

	/**
	 * Creates InceptionV4 Networks with custom weights and labels
	 * 
	 * @param sessionFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param weightsLoader
	 * @param labels
	 */
	public DefaultInceptionV4Factory(DefaultSessionFactory sessionFactory,
			InceptionV4WeightsLoader weightsLoader, InceptionV4Labels labels) {
		this.sessionFactory = sessionFactory;
		this.weightsLoader = weightsLoader;
		this.labels = labels;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork createInceptionV4(FeedForwardNeuralNetworkContext trainingContext)
			throws IOException {

		LOGGER.info("Creating Inception V4 Network...");

		// Obtain the InceptionV4Definition from neural-network-architectures
		InceptionV4Definition inceptionV4Definition = new InceptionV4Definition(weightsLoader);
		
		return sessionFactory
			.createSession(trainingContext.getDirectedComponentsContext())
			.buildSupervised3DNeuralNetwork("inceptionV4", inceptionV4Definition.getInputNeurons())
			.withComponentGraphDefinition(inceptionV4Definition)
			.build();
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetwork createInceptionV4(FeedForwardNeuralNetworkContext trainingContext,
			float regularisationLambda, float dropoutKeepProbability)
			throws IOException {

		LOGGER.info("Creating Inception V4 Network...");

		// Obtain the InceptionV4Definition from neural-network-architectures
		InceptionV4Definition inceptionV4Definition = new InceptionV4Definition(weightsLoader);
		inceptionV4Definition.setFinalDenseLayerInputDropoutKeepProbability(dropoutKeepProbability);
		inceptionV4Definition.setFinalDenseLayerRegularisationLambda(regularisationLambda);

		
		return sessionFactory
				.createSession(trainingContext.getDirectedComponentsContext())
				.buildSupervised3DNeuralNetwork("inceptionV4WithRegularisation", inceptionV4Definition.getInputNeurons())
				.withComponentGraphDefinition(inceptionV4Definition)
				.build();
		
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetwork createInceptionV4WithCustomTail(FeedForwardNeuralNetworkContext trainingContext, int outputNeurons,
			WeightsMatrix weights, BiasVector bias, float regularisationLambda, float dropoutKeepProbability)
			throws IOException {

		LOGGER.info("Creating Inception V4 Network...");

		// Obtain the InceptionV4Definition from neural-network-architectures
		UntrainedTailInceptionV4Definition inceptionV4Definition = new UntrainedTailInceptionV4Definition(
				new DefaultUntrainedInceptionV4WeightsLoader(), weights, bias, outputNeurons, regularisationLambda, dropoutKeepProbability);


		return sessionFactory
				.createSession(trainingContext.getDirectedComponentsContext())
				.buildSupervised3DNeuralNetwork("inceptionV4WithCustomTail", inceptionV4Definition.getInputNeurons())
				.withComponentGraphDefinition(inceptionV4Definition)
				.build();
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetwork createInceptionV4Tail(FeedForwardNeuralNetworkContext trainingContext,
			int outputNeuronCount, WeightsMatrix weights, BiasVector biases, float regularisationLambda, float dropoutKeepProbability)
			throws IOException {

		LOGGER.info("Creating Inception V4 Network...");

		// Obtain the InceptionV4Definition from neural-network-architectures
		InceptionV4CustomTailDefinition inceptionV4Definition = new InceptionV4CustomTailDefinition(outputNeuronCount, weights, biases, regularisationLambda, dropoutKeepProbability);

		return sessionFactory
				.createSession(trainingContext.getDirectedComponentsContext())
				.buildSupervised3DNeuralNetwork("inceptionV4CustomTail", inceptionV4Definition.getInputNeurons())
				.withComponentGraphDefinition(inceptionV4Definition)
				.build();
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetwork createInceptionV4WithoutTail(FeedForwardNeuralNetworkContext trainingContext)
			throws IOException {

		LOGGER.info("Creating Inception V4 Network...");

		// Obtain the InceptionV4Definition from neural-network-architectures
		InceptionV4WithoutTailDefinition inceptionV4Definition = new InceptionV4WithoutTailDefinition(weightsLoader);

		return sessionFactory
				.createSession(trainingContext.getDirectedComponentsContext())
				.buildSupervised3DNeuralNetwork("inceptionV4CustomTail", inceptionV4Definition.getInputNeurons())
				.withComponentGraphDefinition(inceptionV4Definition)
				.build();
	}

	@Override
	public InceptionV4Labels createInceptionV4Labels() throws IOException {
		return labels;
	}
}
