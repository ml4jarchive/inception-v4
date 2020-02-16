package org.ml4j.nn.models.inceptionv4.impl;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public class PretrainedInceptionV4WeightsLoaderImplTest {

	@Mock
	private MatrixFactory mockMatrixFactory;
	
	@Mock
	private Matrix mockMatrix;

	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockMatrixFactory.createMatrixFromRowsByRowsArray(Mockito.eq(32), 
				Mockito.eq(27), Mockito.any())).thenReturn(mockMatrix);
	}

	@Test
	public void testGetConvolutionalLayerWeights() {

		InceptionV4WeightsLoader weightsLoader = new PretrainedInceptionV4WeightsLoaderImpl(
				PretrainedInceptionV4WeightsLoaderImplTest.class.getClassLoader(), mockMatrixFactory);

		WeightsMatrix weightsMatrix = weightsLoader.getConvolutionalLayerWeights("conv2d_1_kernel0", 3, 3, 3, 32);

		Assert.assertEquals(Arrays.asList(Dimension.OUTPUT_DEPTH, Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT,
				Dimension.FILTER_WIDTH), weightsMatrix.getFormat().getDimensions());
		
		Assert.assertEquals(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT,
				Dimension.FILTER_WIDTH), weightsMatrix.getFormat().getInputDimensions());
		
		Assert.assertEquals(Arrays.asList(Dimension.OUTPUT_DEPTH), weightsMatrix.getFormat().getOutputDimensions());
		
		Assert.assertEquals(WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS, weightsMatrix.getFormat().getOrientation());
		
		Assert.assertNotNull(weightsMatrix.getWeights());
		
		Assert.assertEquals(mockMatrix, weightsMatrix.getWeights());

	}
}
