//neural-net-tutorial.cpp

#include <vector>
#include <iostream>
#include <cstdlib>

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

// **************** class Neuron ****************

class Neuron
{
public:
	Neuron(unsigned numOutputs);

private:
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
};

Neuron::Neuron(unsigned numOutputs)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
}

// **************** class Net ****************

class Net
{
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals) {};
	void backProp(const std::vector<double> &targetVals) {};
	void getResults(std::vector<double> &resultVals) const {};

private:
	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};

Net::Net(const std::vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// we have made a new layer, now fill it with neurons and 
		// add a bias neuron to the layer:
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs));
			std::cout << "Made a Neuron!" << std::endl;
		}
	}
}

int main()
{
	// eg., {3, 2, 1}
	std::vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);

	std::vector<double> inputVals;
	myNet.feedForward(inputVals);

	std::vector<double> targetVals;
	myNet.backProp(targetVals);

	std::vector<double> resultVals;
	myNet.getResults(resultVals);
}