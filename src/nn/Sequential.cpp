#include <vector>
#include <Eigen/Dense>
#include "../../include/nn/Sequential.h"
#include "../../include/nn/layers/Layer.h"
#include "../../include/nn/criteria/Criterion.h"

Eigen::MatrixXd Sequential::operator()(const Eigen::MatrixXd& X)
{
	Eigen::MatrixXd out = X;
	for (auto& layer : layers)
	{
		out = (*layer)(out);
	}
	return out;
}

void Sequential::backward(Criterion& objective)
{
	Eigen::MatrixXd dout = objective._backward();
	for (auto it = layers.rbegin(); it != layers.rend(); ++it)
	{
		dout = (*it)->_backward(dout);
	}
}