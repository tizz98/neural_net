//neural-net-tutorial.cpp

class Net
{
public:
	Net(topology);
	void feedForward(inputVals);
	void backProp(targetVals);
	void getResults(resultVals) const;

private:
};

int main()
{
	Net myNet(topology);

	myNet.feedForward(inputVals);
	myNet.backProp(targetVals);
	myNet.getResults(resultVals);
}