#include "trainingSet.h"
#include "neuron.h"
#include "net.h"


void showVectorValues(const std::string &label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (auto i : v)
		std::cout << i << " ";
	std::cout << std::endl;
}


int main()
{
	trainingSet trainingData("/Users/dubovoy/CLionProjects/untitled/testData.txt");     // TODO: remove hardcode
	std::vector<unsigned> topology;
	trainingData.getTopology(topology);
	net net(topology);

	std::vector<double> inputValues, targetValues, resultValues;
	int trainingPass = 0;

	while (!trainingData.isEOF())
	{
		++trainingPass;
		std::cout << std::endl << "Pass: " << trainingPass << std::endl;

		if (trainingData.getNextInputs(inputValues) != topology[0])
			break;

		showVectorValues("Input:", inputValues);
		net.feedForward(inputValues);

		trainingData.getTargetOutputs(targetValues);
		showVectorValues("Targets:", targetValues);
		assert(targetValues.size() == topology.back());

		net.getResults(resultValues);
		showVectorValues("Outputs", resultValues);

		net.backPropagation(targetValues);
		std::cout << "Net average error: " << net.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;
	return 0;
}
