#include <iostream>
#include <Eigen/Dense>
#include "include/nn/layers/Layers.h"
#include "include/nn/Sequential.h"
#include "include/nn/criteria/BinaryCrossEntropyLoss.h"

int main()
{
	Eigen::MatrixXd X(4, 2);
	X << 0, 0,
		0, 1,
		1, 0,
		1, 1;

	Eigen::MatrixXd y(4, 1);
	y << 0, 1, 1, 0;

	Sequential model(
		Linear(2, 4),
		ReLU(),
		Linear(4, 4),
		ReLU(),
		Linear(4, 1),
		Sigmoid()
	);

	Eigen::MatrixXd out = model(X);

	BinaryCrossEntropyLoss loss_fn;
	double loss = loss_fn(y, out);

	std::cout << "Output:\n" << out << std::endl;
	std::cout << "Loss: " << loss << std::endl;

	model.backward(loss_fn);

	std::cin.get();
 	return 0;
}