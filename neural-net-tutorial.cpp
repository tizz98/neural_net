//neural-net-tutorial.cpp

#include <vector>

class Net
{
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals) const;

private:
};

int main()
{
	// eg., {3, 2, 1}
	std::vector<unsigned> topology;
	Net myNet(topology);

	std::vector<double> inputVals;
	myNet.feedForward(inputVals);

	std::vector<double> targetVals;
	myNet.backProp(targetVals);

	std::vector<double> resultVals;
	myNet.getResults(resultVals);
}