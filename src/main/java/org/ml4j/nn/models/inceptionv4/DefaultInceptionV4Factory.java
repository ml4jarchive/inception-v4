package org.ml4j.nn.models.inceptionv4;

import static org.ml4j.nn.models.inceptionv4.InceptionV4WeightsLoader.getDenseLayerWeights;
import static org.ml4j.nn.models.inceptionv4.InceptionV4WeightsLoader.getConvolutionalLayerWeights;
import static org.ml4j.nn.models.inceptionv4.InceptionV4WeightsLoader.getBatchNormLayerWeights;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilderFactory;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;

/**
 * Default factory for an Inception V4 Network.
 * 
 */
public class DefaultInceptionV4Factory implements InceptionV4Factory {

	private DifferentiableActivationFunctionFactory activationFunctionFactory;
	
	private Components3DGraphBuilderFactory components3DGraphBuilderFactory;
	
	private SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory;

	private float regularisationLambda = 0;

	private float batchNormRegularisationLambda = 0;

	private boolean withFreezeOut = false;
	
	public DefaultInceptionV4Factory(
			Components3DGraphBuilderFactory components3DGraphBuilderFactory, 
			DifferentiableActivationFunctionFactory activationFunctionFactory,
			SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory) {
		this.components3DGraphBuilderFactory = components3DGraphBuilderFactory;
		this.activationFunctionFactory = activationFunctionFactory;
		this.supervisedFeedForwardNeuralNetworkFactory = supervisedFeedForwardNeuralNetworkFactory;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork createInceptionV4(FeedForwardNeuralNetworkContext trainingContext)
			throws IOException {

		// Build the list of components in the Inception V4 Network - this will configure the trainingContext with 
		// any component-specific configuration (eg. dropout, freezout, regularisation)
		List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> 
			allComponents = new ArrayList<>();
		
		// Add Stem components
		allComponents.addAll(createStem(trainingContext.getDirectedComponentsContext()).getComponents());
		
		// Add 4 Inception A Blocks
		for (int inceptionAIndex = 0; inceptionAIndex < 4; inceptionAIndex++) {
			allComponents.addAll(createInceptionA(trainingContext.getDirectedComponentsContext(), inceptionAIndex).getComponents());
		}

		// Add Reduction A Block
		allComponents.addAll(createReductionA(trainingContext.getDirectedComponentsContext()).getComponents());

		// Add 7 Inception B Blocks
		for (int inceptionBIndex = 0; inceptionBIndex < 7; inceptionBIndex++) {
			allComponents.addAll(createInceptionB(trainingContext.getDirectedComponentsContext(), inceptionBIndex).getComponents());
		}
		
		// Add Reduction B Block
		allComponents.addAll(createReductionB(trainingContext.getDirectedComponentsContext()).getComponents());

		// Add 3 Inception C Blocks
		for (int inceptionCIndex = 0; inceptionCIndex < 3; inceptionCIndex++) {
			allComponents.addAll(createInceptionC(trainingContext.getDirectedComponentsContext(), inceptionCIndex).getComponents());
		}
		
		// Add Tail components
		allComponents.addAll(createTailComponents(trainingContext.getDirectedComponentsContext()));

		// Create a supervised feed forward neural network for this chain of components
		return supervisedFeedForwardNeuralNetworkFactory.createSupervisedFeedForwardNeuralNetwork(allComponents);
	}

	public InitialComponents3DGraphBuilder createStem(DirectedComponentsContext directedComponentsContext) throws IOException {

		return components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(new Neurons3D(299, 299, 3, false))
				.withConvolutionalAxons()
					.withConnectionWeights(
							getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_1_kernel0", 3, 3, 3, 32))
					.withStride(2, 2).withFilterSize(3, 3).withFilterCount(32).withValidPadding()
					.withAxonsContext(directedComponentsContext,
							c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(149, 149, 32, false))
				.withBatchNormAxons()
					.withBiasUnit()
					.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_1_beta0", 149,
							149, 32, 1))
					.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
							"batch_normalization_1_moving_mean0", 149, 149, 32, 1))
					.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
							"batch_normalization_1_moving_variance0", 149, 149, 32, 1))
					.withAxonsContext(directedComponentsContext, c -> c.withFreezeOut(withFreezeOut))
					// c.withRegularisationLambda(batchNormRegularisationLambda))
				.withConnectionToNeurons(new Neurons3D(149, 149, 32, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
					.withConnectionWeights(
							getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_2_kernel0", 3, 3, 32, 32))
					.withFilterSize(3, 3).withValidPadding()
					.withAxonsContext(directedComponentsContext, c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(147, 147, 32, false))
				.withBatchNormAxons()
					.withBiasUnit()
					.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_2_beta0", 147,
							147, 32, 1))
					.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
							"batch_normalization_2_moving_mean0", 147, 147, 32, 1))
					.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
							"batch_normalization_2_moving_variance0", 147, 147, 32, 1))
					.withAxonsContext(directedComponentsContext, c -> c.withFreezeOut(withFreezeOut))
					// .withAxonsContext(directedComponentsContext, c ->
					// c.withRegularisationLambda(batchNormRegularisationLambda))
				.withConnectionToNeurons(new Neurons3D(147, 147, 32, false))
					.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
					.withConvolutionalAxons()
						.withConnectionWeights(
								getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_3_kernel0", 3, 3, 32, 64))
						.withFilterSize(3, 3).withFilterCount(64).withSamePadding()
						.withAxonsContext(directedComponentsContext,
								c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(147, 147, 64, false))
				.withBatchNormAxons()
				.withBiasUnit()
					.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_3_beta0", 147,
							147, 64, 1))
					.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
							"batch_normalization_3_moving_mean0", 147, 147, 64, 1))
					.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
							"batch_normalization_3_moving_variance0", 147, 147, 64, 1))
					.withAxonsContext(directedComponentsContext,
							c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(147, 147, 64, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withParallelPaths()
					.withPath()
						.withMaxPoolingAxons()
							.withStride(2, 2)
							.withFilterSize(3, 3)
							.withValidPadding()
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.endPath()
					.withPath()
						.withConvolutionalAxons()
							.withConnectionWeights(
									getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_4_kernel0", 3, 3, 64, 96))
							.withStride(2, 2).withFilterSize(3, 3).withFilterCount(96).withValidPadding()
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 96, false))
						.withBatchNormAxons()
							.withBiasUnit()
							.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_4_beta0", 73,
									73, 96, 1))
							.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_4_moving_mean0", 73, 73, 96, 1))
							.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_4_moving_variance0", 73, 73, 96, 1))
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(73, 73, 96, false))
							.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
						.endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT)
				.withParallelPaths()
					.withPath()
						.withConvolutionalAxons()
							.withConnectionWeights(
									getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_5_kernel0", 1, 1, 160, 64))
							.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons().withBiasUnit()
							.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_5_beta0", 73,
									73, 64, 1))
							.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_5_moving_mean0", 73, 73, 64, 1))
							.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_5_moving_variance0", 73, 73, 64, 1))
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
						.withConvolutionalAxons()
							.withConnectionWeights(
									getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_6_kernel0", 3, 3, 64, 96))
							.withFilterSize(3, 3).withFilterCount(96).withValidPadding()
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withBatchNormAxons()
							.withBiasUnit()
							.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_6_beta0", 71,
									71, 96, 1))
							.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_6_moving_mean0", 71, 71, 96, 1))
							.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_6_moving_variance0", 71, 71, 96, 1))
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
					.endPath()
					.withPath()
						.withConvolutionalAxons()
							.withConnectionWeights(
									getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_7_kernel0", 1, 1, 160, 64))
							.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons()
							.withBiasUnit()
							.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_7_beta0", 73,
									73, 64, 1))
							.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_7_moving_mean0", 73, 73, 64, 1))
							.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_7_moving_variance0", 73, 73, 64, 1))
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
						.withConvolutionalAxons()
							.withConnectionWeights(
									getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_8_kernel0", 7, 1, 64, 64))
							.withFilterSize(7, 1).withFilterCount(64).withSamePadding()
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons()
							.withBiasUnit()
							.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_8_beta0", 73,
									73, 64, 1))
							.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_8_moving_mean0", 73, 73, 64, 1))
							.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_8_moving_variance0", 73, 73, 64, 1))
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
						.withConvolutionalAxons()
							.withConnectionWeights(
									getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_9_kernel0", 1, 7, 64, 64))
							.withFilterSize(1, 7).withFilterCount(64).withSamePadding()
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withBatchNormAxons()
						.withBiasUnit()
							.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_9_beta0", 73,
									73, 64, 1))
							.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_9_moving_mean0", 73, 73, 64, 1))
							.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_9_moving_variance0", 73, 73, 64, 1))
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(73, 73, 64, false))
						.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
						.withConvolutionalAxons()
							.withConnectionWeights(
									getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_10_kernel0", 3, 3, 64, 96))
							.withFilterSize(3, 3).withFilterCount(96).withValidPadding()
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
							.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withBatchNormAxons()
							.withBiasUnit()
							.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_10_beta0", 71,
									71, 96, 1))
							.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_10_moving_mean0", 71, 71, 96, 1))
							.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
									"batch_normalization_10_moving_variance0", 71, 71, 96, 1))
							.withAxonsContext(directedComponentsContext,
									c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(71, 71, 96, false))
						.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
					.endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT)
				.withParallelPaths()
					.withPath()
						.withConvolutionalAxons()
						.withConnectionWeights(
								getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_11_kernel0", 3, 3, 192, 192))
						.withStride(2, 2).withFilterSize(3, 3).withFilterCount(192).withValidPadding()
						.withAxonsContext(directedComponentsContext,
								c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
						.withBatchNormAxons().withBiasUnit()
						.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_11_beta0", 35,
								35, 192, 1))
						.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
								"batch_normalization_11_moving_mean0", 35, 35, 192, 1))
						.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
								"batch_normalization_11_moving_variance0", 35, 35, 192, 1))
						.withAxonsContext(directedComponentsContext,
								c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
						.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
						.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.endPath()
				.withPath()
					.withMaxPoolingAxons()
						.withFilterSize(3, 3)
						.withStride(2, 2)
						.withValidPadding()
					.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
				.endPath()
			.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

	public InitialComponents3DGraphBuilder createReductionA(DirectedComponentsContext directedComponentsContext)
			throws IOException {
		return components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(new Neurons3D(35, 35, 384, false))
				.withParallelPaths()
				.withPath()
				.withConvolutionalAxons().withFilterSize(3, 3)
				.withConnectionWeights(
						getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_40_kernel0", 3, 3, 384, 384))
				.withStride(2, 2).withFilterCount(384).withValidPadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_40_beta0", 17,
						17, 384, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_40_moving_mean0", 17, 17, 384, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_40_moving_variance0", 17, 17, 384, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(
						getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_41_kernel0", 1, 1, 384, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_41_beta0", 35,
						35, 192, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_41_moving_mean0", 35, 35, 192, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_41_moving_variance0", 35, 35, 192, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(
						getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_42_kernel0", 3, 3, 192, 224))
				.withFilterSize(3, 3).withFilterCount(224).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 224, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_42_beta0", 35,
						35, 224, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_42_moving_mean0", 35, 35, 224, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_42_moving_variance0", 35, 35, 224, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 224, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(
						getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(), "conv2d_43_kernel0", 3, 3, 224, 256))
				.withStride(2, 2).withFilterSize(3, 3).withFilterCount(256).withValidPadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(), "batch_normalization_43_beta0", 17,
						17, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_43_moving_mean0", 17, 17, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_43_moving_variance0", 17, 17, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withMaxPoolingAxons().withFilterSize(3, 3).withStride(2, 2).withValidPadding()
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false)).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

	public InitialComponents3DGraphBuilder createInceptionA(DirectedComponentsContext directedComponentsContext,
			int index) throws IOException {
		int initial = index * 7 + 12;
		return components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(new Neurons3D(35, 35, 384, false))
				.withParallelPaths().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial) + "_kernel0", 1, 1, 384, 96))
				.withFilterSize(1, 1).withFilterCount(96).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_beta0", 35, 35, 96, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_moving_mean0", 35, 35, 96, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_moving_variance0", 35, 35, 96, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 1) + "_kernel0", 1, 1, 384, 64))
				.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_beta0", 35, 35, 64, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_moving_mean0", 35, 35, 64, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_moving_variance0", 35, 35, 64, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 2) + "_kernel0", 3, 3, 64, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_beta0", 35, 35, 96, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_moving_mean0", 35, 35, 96, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_moving_variance0", 35, 35, 96, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 3) + "_kernel0", 1, 1, 384, 64))
				.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_beta0", 35, 35, 64, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_moving_mean0", 35, 35, 64, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_moving_variance0", 35, 35, 64, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 4) + "_kernel0", 3, 3, 64, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_beta0", 35, 35, 96, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_moving_mean0", 35, 35, 96, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_moving_variance0", 35, 35, 96, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 5) + "_kernel0", 3, 3, 96, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_beta0", 35, 35, 96, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_moving_mean0", 35, 35, 96, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_moving_variance0", 35, 35, 96, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()

				.withAveragePoolingAxons().withFilterSize(3, 3).withStride(1, 1).withSamePadding()
				.withConnectionToNeurons(new Neurons3D(35, 35, 384, false)).withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 6) + "_kernel0", 1, 1, 384, 96))
				.withFilterSize(1, 1).withFilterCount(96).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_beta0", 35, 35, 96, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_moving_mean0", 35, 35, 96, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_moving_variance0", 35, 35, 96, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

	public InitialComponents3DGraphBuilder createInceptionB(DirectedComponentsContext directedComponentsContext,
			int index) throws IOException {
		int initial = index * 10 + 44;
		return components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(new Neurons3D(17, 17, 1024, false))
				.withParallelPaths().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial) + "_kernel0", 1, 1, 1024, 384))
				.withFilterSize(1, 1).withFilterCount(384).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_beta0", 17, 17, 384, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_moving_mean0", 17, 17, 384, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_moving_variance0", 17, 17, 384, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 1) + "_kernel0", 1, 1, 1024, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_beta0", 17, 17, 192, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_moving_mean0", 17, 17, 192, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_moving_variance0", 17, 17, 192, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 2) + "_kernel0", 7, 1, 192, 224))
				.withFilterSize(7, 1).withFilterCount(224).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_beta0", 17, 17, 224, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_moving_mean0", 17, 17, 224, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_moving_variance0", 17, 17, 224, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 3) + "_kernel0", 1, 7, 224, 256))
				.withFilterSize(1, 7).withFilterCount(224).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_beta0", 17, 17, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_moving_mean0", 17, 17, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_moving_variance0", 17, 17, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 4) + "_kernel0", 1, 1, 1024, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_beta0", 17, 17, 192, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_moving_mean0", 17, 17, 192, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_moving_variance0", 17, 17, 192, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 5) + "_kernel0", 1, 7, 192, 192))
				.withFilterSize(1, 7).withFilterCount(192).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_beta0", 17, 17, 192, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_moving_mean0", 17, 17, 192, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_moving_variance0", 17, 17, 192, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 6) + "_kernel0", 7, 1, 192, 224))
				.withFilterSize(7, 1).withFilterCount(224).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_beta0", 17, 17, 224, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_moving_mean0", 17, 17, 224, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_moving_variance0", 17, 17, 224, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 7) + "_kernel0", 1, 7, 224, 224))
				.withFilterSize(1, 7).withFilterCount(224).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 7) + "_beta0", 17, 17, 224, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 7) + "_moving_mean0", 17, 17, 224, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 7) + "_moving_variance0", 17, 17, 224, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 224, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 8) + "_kernel0", 7, 1, 224, 256))
				.withFilterSize(7, 1).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 8) + "_beta0", 17, 17, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 8) + "_moving_mean0", 17, 17, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 8) + "_moving_variance0", 17, 17, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withAveragePoolingAxons().withFilterSize(3, 3).withStride(1, 1).withSamePadding()
				.withConnectionToNeurons(new Neurons3D(17, 17, 1024, false)).withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 9) + "_kernel0", 1, 1, 1024, 128))
				.withFilterSize(1, 1).withFilterCount(128).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 128, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 9) + "_beta0", 17, 17, 128, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 9) + "_moving_mean0", 17, 17, 128, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 9) + "_moving_variance0", 17, 17, 128, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 128, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

	public InitialComponents3DGraphBuilder createReductionB(DirectedComponentsContext directedComponentsContext)
			throws IOException {
		return components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(new Neurons3D(17, 17, 1024, false)).withParallelPaths().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (114) + "_kernel0", 1, 1, 1024, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (114) + "_beta0", 17, 17, 192, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (114) + "_moving_mean0", 17, 17, 192, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (114) + "_moving_variance0", 17, 17, 192, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 192, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (115) + "_kernel0", 3, 3, 192, 192))
				.withStride(2, 2).withFilterSize(3, 3).withFilterCount(192).withValidPadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 192, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (115) + "_beta0", 8, 8, 192, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (115) + "_moving_mean0", 8, 8, 192, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (115) + "_moving_variance0", 8, 8, 192, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 192, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (116) + "_kernel0", 1, 1, 1024, 256))
				.withFilterSize(1, 1).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (116) + "_beta0", 17, 17, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (116) + "_moving_mean0", 17, 17, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (116) + "_moving_variance0", 17, 17, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (117) + "_kernel0", 7, 1, 256, 256))
				.withFilterSize(7, 1).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (117) + "_beta0", 17, 17, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (117) + "_moving_mean0", 17, 17, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (117) + "_moving_variance0", 17, 17, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (118) + "_kernel0", 1, 7, 256, 320))
				.withFilterSize(1, 7).withFilterCount(320).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 320, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (118) + "_beta0", 17, 17, 320, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (118) + "_moving_mean0", 17, 17, 320, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (118) + "_moving_variance0", 17, 17, 320, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 320, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (119) + "_kernel0", 3, 3, 320, 320))
				.withStride(2, 2).withFilterSize(3, 3).withFilterCount(320).withValidPadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 320, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (119) + "_beta0", 8, 8, 320, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (119) + "_moving_mean0", 8, 8, 320, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (119) + "_moving_variance0", 8, 8, 320, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 320, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withMaxPoolingAxons().withFilterSize(3, 3).withStride(2, 2).withValidPadding()
				.withConnectionToNeurons(new Neurons3D(8, 8, 1024, false)).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}

	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> createTailComponents(
			DirectedComponentsContext directedComponentsContext) throws IOException {
		
		return components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(new Neurons3D(8, 8, 1536, false))
				.withSynapses()
				.withAveragePoolingAxons().withStride(1, 1).withFilterSize(8, 8).withValidPadding()
				.withConnectionToNeurons(new Neurons3D(1, 1, 1536, false))
				.endSynapses()
				.withFullyConnectedAxons()
				.withConnectionWeights(
						getDenseLayerWeights(directedComponentsContext.getMatrixFactory(), "dense_1_kernel0", 1001, 1536))
				.withBiases(getDenseLayerWeights(directedComponentsContext.getMatrixFactory(), "dense_1_bias0", 1001, 1))
				.withAxonsContext(directedComponentsContext, c -> c.withRegularisationLambda(regularisationLambda))
				.withBiasUnit()
				// .withAxonsContext(directedComponentsContext, c ->
				// c.withLeftHandInputDropoutKeepProbability(0.8f))
				.withConnectionToNeurons(new Neurons(1001, false))
				.withActivationFunction(activationFunctionFactory.createSoftmaxActivationFunction()).getComponents();
	
	}

	public InitialComponents3DGraphBuilder createInceptionC(DirectedComponentsContext directedComponentsContext,
			int index) throws IOException {
		int initial = index * 10 + 120;
		return components3DGraphBuilderFactory.createInitialComponents3DGraphBuilder(new Neurons3D(8, 8, 1536, false)).withParallelPaths()

				.withPath().withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial) + "_kernel0", 1, 1, 1536, 256))
				.withFilterSize(1, 1).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_beta0", 8, 8, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_moving_mean0", 8, 8, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial) + "_moving_variance0", 8, 8, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 1) + "_kernel0", 1, 1, 1536, 384))
				.withFilterSize(1, 1).withFilterCount(384).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_beta0", 8, 8, 384, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_moving_mean0", 8, 8, 384, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 1) + "_moving_variance0", 8, 8, 384, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).withParallelPaths()
				.withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 2) + "_kernel0", 3, 1, 384, 256))
				.withFilterSize(3, 1).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_beta0", 8, 8, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_moving_mean0", 8, 8, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 2) + "_moving_variance0", 8, 8, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 3) + "_kernel0", 1, 3, 384, 256))
				.withFilterSize(1, 3).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_beta0", 8, 8, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_moving_mean0", 8, 8, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 3) + "_moving_variance0", 8, 8, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT).endPath().withPath()
				// 124
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 4) + "_kernel0", 1, 1, 1536, 384))
				.withFilterSize(1, 1).withFilterCount(384).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_beta0", 8, 8, 384, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_moving_mean0", 8, 8, 384, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 4) + "_moving_variance0", 8, 8, 384, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 5) + "_kernel0", 1, 3, 384, 448))
				.withFilterSize(1, 3).withFilterCount(448).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 448, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_beta0", 8, 8, 448, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_moving_mean0", 8, 8, 448, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 5) + "_moving_variance0", 8, 8, 448, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 448, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction())
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 6) + "_kernel0", 3, 1, 448, 512))
				.withFilterSize(3, 1).withFilterCount(512).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 512, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_beta0", 8, 8, 512, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_moving_mean0", 8, 8, 512, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 6) + "_moving_variance0", 8, 8, 512, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 512, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).withParallelPaths()
				.withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 7) + "_kernel0", 3, 1, 512, 256))
				.withFilterSize(3, 1).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 7) + "_beta0", 8, 8, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 7) + "_moving_mean0", 8, 8, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 7) + "_moving_variance0", 8, 8, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath().withPath()
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 8) + "_kernel0", 1, 3, 512, 256))
				.withFilterSize(1, 3).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 8) + "_beta0", 8, 8, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 8) + "_moving_mean0", 8, 8, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 8) + "_moving_variance0", 8, 8, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT).endPath().withPath().withAveragePoolingAxons()
				.withFilterSize(3, 3).withStride(1, 1).withSamePadding()
				.withConnectionToNeurons(new Neurons3D(8, 8, 1536, false))
				.withConvolutionalAxons()
				.withConnectionWeights(getConvolutionalLayerWeights(directedComponentsContext.getMatrixFactory(),
						"conv2d_" + (initial + 9) + "_kernel0", 1, 1, 1536, 256))
				.withFilterSize(1, 1).withFilterCount(256).withSamePadding()
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(regularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons().withBiasUnit()
				.withBeta(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 9) + "_beta0", 8, 8, 256, 1))
				.withMean(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 9) + "_moving_mean0", 8, 8, 256, 1))
				.withVariance(getBatchNormLayerWeights(directedComponentsContext.getMatrixFactory(),
						"batch_normalization_" + (initial + 9) + "_moving_variance0", 8, 8, 256, 1))
				.withAxonsContext(directedComponentsContext,
						c -> c.withRegularisationLambda(batchNormRegularisationLambda).withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction(activationFunctionFactory.createReluActivationFunction()).endPath()
				.endParallelPaths(PathCombinationStrategy.FILTER_CONCAT);
	}
}
