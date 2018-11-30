#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>


class trainingSet
{
public:
	explicit trainingSet(std::string filename);
	bool isEOF() { return trainingDataFile.eof(); }
	void getTopology(std::vector<unsigned> &topology);
	unsigned getNextInputs(std::vector<double> &inputValues);
	unsigned getTargetOutputs(std::vector<double> &targetOutputValues);
private:
	std::ifstream trainingDataFile;
};
