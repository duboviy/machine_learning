#include "trainingSet.h"
#include "neuron.h"
#include "net.h"


// Number of training samples to average over
double net::recentAverageSmoothingFactor = 100.0;


net::net(const std::vector<unsigned> &topology)
{
	auto numLayers = (unsigned) topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		layers.emplace_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			layers.back().push_back(neuron(numOutputs, neuronNum));
		}
		// add Bias ("constant" - with 1 output value) neuron to each layer to correct curve
		layers.back().back().setOutputValue(1.0);
	}
}


void net::getResults(std::vector<double> &resultValues) const
{
	resultValues.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		resultValues.push_back(layers.back()[n].getOutputValue());
	}
}


void net::backPropagation(const std::vector<double> &targetValues)
{
	Layer &outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetValues[n] - outputLayer[n].getOutputValue();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1; 
	error = sqrt(error); 

	recentAverageError =
		(recentAverageError * recentAverageSmoothingFactor + error)
		/ (recentAverageSmoothingFactor + 1.0);

	// calculate gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetValues[n]);
	}

	for (unsigned layerNum = (unsigned) layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// update weights
	for (unsigned layerNum = (unsigned) layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}


void net::feedForward(const std::vector<double> &inputValues)
{
	assert(inputValues.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < inputValues.size(); ++i) {
		layers[0][i].setOutputValue(inputValues[i]);
	}

	for (unsigned layerNum = 1; layerNum < layers.size(); ++layerNum) {
		Layer &prevLayer = layers[layerNum - 1];
		for (unsigned n = 0; n < layers[layerNum].size() - 1; ++n) {
			layers[layerNum][n].feedForward(prevLayer);
		}
	}
}
