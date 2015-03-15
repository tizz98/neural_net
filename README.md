# Neural Net Tutorial

from https://vimeo.com/19569529 and http://millermattson.com/dave/

# To Run
- `$ g++ neural-net-tutorial.cpp -o neural-net`
	- compile the neural net program, compiled program in `neural-net`
- `$ g++ makeTraningSamples.cpp -o makeTraningSamples`
	- compile the training sample data program, compiled program in `makeTraningSamples`
- `$ ./makeTrainingSamples > /tmp/trainingData.txt`
	- run the traning sample data program and redirect all output to a file in `/tmp/`
- `$ ./neural-net > out.txt`
	- Results are in `out.txt` or you can leave off `out.txt` to have the results printed to the console