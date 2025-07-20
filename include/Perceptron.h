#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <random>

class Perceptron
{
	Eigen::MatrixXd m_X;
	Eigen::VectorXd m_W;

	Eigen::VectorXd g()
	{
		return m_X * m_W;
	}
	Eigen::VectorXi f(Eigen::VectorXd g)
	{
		auto f = [](double x) { return x >= 0 ? 1 : 0; };
		return g.unaryExpr(f);
	}

public:
    Perceptron() : m_X(), m_W() {}

	void Train(Eigen::MatrixXd& X, Eigen::VectorXi& Y, int epochs)
	{
		m_X = X;
		if (m_X.rows() != Y.size())
		{
			throw std::invalid_argument("Unequal number of samples in X and Y.");
		}

		// Add bias term to the input matrix
		Eigen::VectorXd ones(m_X.rows());
		ones.setOnes();
		Eigen::MatrixXd X_with_bias(m_X.rows(), m_X.cols() + 1);
		X_with_bias.col(0) = ones;
		X_with_bias.block(0, 1, m_X.rows(), m_X.cols()) = m_X;
		m_X = X_with_bias;

		m_W = Eigen::VectorXd::Random(m_X.cols()); // Initialize weights randomly

		std::cout << "Initial Weights: " << m_W.transpose() << std::endl;
		for (int i = 0; i < epochs; i++)
		{
			std::cout << "Epoch " << i + 1 << "-> ";

			int idx = std::rand() % m_X.rows(); // Randomly select a sample index
			if (Y(idx) == 1 && m_X.row(idx) * m_W < 0)
				m_W += m_X.row(idx);
			else if (Y(idx) == 0 && m_X.row(idx) * m_W >= 0)
				m_W -= m_X.row(idx);

			std::cout << "Weights: " << m_W.transpose() << std::endl;
		}
		std::cout << "Final Weights: " << m_W.transpose() << std::endl;
	}

    Eigen::VectorXi Predict()
    {
        Eigen::VectorXd g_X = g();
        return f(g_X);
    }
};

