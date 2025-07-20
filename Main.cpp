#include <iostream>
//#include <cmath>
#include <Eigen/Dense>
#include "include/Perceptron.h"
#include "include/Layers.h"

int main()
{
	Eigen::MatrixXd X(4, 2);
    X << -1, -1, 
        -1, 1,
        1, -1,
        1, 1;

    std::cout << "Input: \n" << X << std::endl;

    Layer layer1(X, 4, true, "relu");
    Eigen::MatrixXd out1 = layer1.forward();

    Layer layer2(out1, 1, true, "sigmoid");
    Eigen::MatrixXd out = layer2.forward();

    std::cout << "\nOutput: \n" << out << std::endl;

 	return 0;
}