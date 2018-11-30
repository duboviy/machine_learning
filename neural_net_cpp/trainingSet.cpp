#include "trainingSet.h"


trainingSet::trainingSet(const std::string filename)
{
	trainingDataFile.open(filename.c_str());
}


void trainingSet::getTopology(std::vector<unsigned> &topology)
{
	std::string line;
	std::string label;

	std::getline(trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;

	assert(label == "topology:");
	assert(!this->isEOF());

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
}


unsigned trainingSet::getNextInputs(std::vector<double> &inputValues)
{
	inputValues.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;

	if (label == "in:") {
		double oneValue;
		while (ss >> oneValue) {
			inputValues.push_back(oneValue);
		}
	}

	return (unsigned) inputValues.size();
}


unsigned trainingSet::getTargetOutputs(std::vector<double> &targetOutputValues)
{
	targetOutputValues.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;

	if (label == "out:") {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputValues.push_back(oneValue);
		}
	}

	return (unsigned) targetOutputValues.size();
}
