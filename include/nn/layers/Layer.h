#pragma once
#include <Eigen/Dense>

namespace nn
{
	namespace layers
	{
		class Layer
		{
		public:
			virtual ~Layer() = default;
			virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& X) = 0;
			Eigen::MatrixXd operator()(const Eigen::MatrixXd& X)
			{
				return forward(X);
			}
		};
	}
}