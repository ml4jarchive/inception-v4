package org.ml4j.nn.models.inceptionv4;

import java.io.IOException;

import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;

/**
 * Interface for a factory for a Inception V4 Network
 * 
 * @author Michael Lavelle
 *
 */
public interface InceptionV4Factory {
	
	/**
	 * Create a new Inception V4 Network.
	 * 
	 * @param context The training or prediction context with which to construct this network.  This method may
	 * perform additional configurations of this config during execution.
	 * @return An inception V4 Network
	 * @throws IOException In the event that the network cannot be loaded
	 */
	SupervisedFeedForwardNeuralNetwork createInceptionV4(FeedForwardNeuralNetworkContext context) throws IOException;
}
