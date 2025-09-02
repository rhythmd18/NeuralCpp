#include <iostream>
#include <Eigen/Dense>
#include "include/nn/layers/Layers.h"
#include "include/nn/Sequential.h"
#include "include/nn/criteria/BinaryCrossEntropyLoss.h"
#include "include/nn/optimizers/SGD.h"

void train(
	Eigen::MatrixXd& X, Eigen::MatrixXd& y,
	Sequential& model, 
	Criterion& loss_fn, 
	Optimizer& optimizer,
	int epochs=100)
{
	for (int i = 0; i < epochs; i++)
	{
		Eigen::MatrixXd out = model(X);
		double loss = loss_fn(y, out);

		if (i % 100 == 0)
		{
			//std::cout << "\nOutput: " << out << std::endl;
			std::cout << "Epoch: " << i << ", Loss: " << loss << std::endl;
		}

		// Backpropagation
		model.backward(loss_fn);

		// Gradient descent updation step
		optimizer.step();

		// Resetting gradients
		optimizer.zero_grad();
	}
}

int main()
{
	Eigen::MatrixXd X(8, 3);
	X << 0, 0, 0,
		0, 0, 1,
		0, 1, 0,
		0, 1, 1,
		1, 0, 0,
		1, 0, 1,
		1, 1, 0,
		1, 1, 1;

	Eigen::MatrixXd y(8, 1);
	y << 0, 1, 1, 0, 1, 0, 0, 1;

	Sequential model(
		Linear(3, 4),
		ReLU(),
		Linear(4, 4),
		ReLU(),
		Linear(4, 1),
		Sigmoid()
	);



	BinaryCrossEntropyLoss loss_fn;
	SGD optimizer(model, 0.1, 0.9);

	train(X, y, model, loss_fn, optimizer, 10000);

	Eigen::MatrixXd out = model(X);

	std::cout << "\nInput: \n" << X << std::endl;
	std::cout << "\nOutput: \n " << out << std::endl;

	std::cin.get();
 	return 0;
}