#pragma once
#include <vector>



class net
{
public:
    // specify topology to initialize network - define amount of layers & amount of neurons in each layer.
    // e.g. topology: 2 3 1 - 2 neurons in input layer, 3 in hidden layer & 1 in output layer
	explicit net(const std::vector<unsigned> &topology);

    // moves data in only one direction, forward, from the input nodes,
    // through the hidden nodes (if any) and to the output nodes
	void feedForward(const std::vector<double> &inputValues);

	// network training algorithm (RMSE is used here)
	// argument - target values that are expected at output
	void backPropagation(const std::vector<double> &targetValues);

	// method to get network approximation results
	// iterate over output layer neurons and collect their outputs
	void getResults(std::vector<double> &resultValues) const;

	// helper method to debug how "good: network is trained
	double getRecentAverageError() const { return recentAverageError; }
private:
	std::vector<Layer> layers;	// layers[layerNumber][NeuronNumber]
	double error;
	double recentAverageError;
	static double recentAverageSmoothingFactor;
};
