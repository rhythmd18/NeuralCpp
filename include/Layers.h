#pragma once  
#include <string>  
#include <Eigen/Dense> 
#include "Activations.h"
  
class Layer
{  
    int m_in_dims;  
    int m_units;
    Eigen::MatrixXd m_X;
    Eigen::MatrixXd m_W;
    bool m_bias;
    std::string m_activation;
  
public:  
    Layer(Eigen::MatrixXd& X, int units, bool bias = true, std::string activation = "linear") :
        m_in_dims(X.cols()),
        m_units(units),
        m_bias(bias),
		m_activation(activation)
    {  
        if (m_in_dims <= 0 || m_units <= 0)
			throw std::invalid_argument("Input and output dimensions must be positive integers.");

        if (bias)
        {
            // Adding an additional column of ones to m_X
            Eigen::VectorXd ones(X.rows());
            ones.setOnes();
            Eigen::MatrixXd X_with_bias(X.rows(), X.cols() + 1);
            X_with_bias << ones, X;
            m_X = X_with_bias;
            m_W = Eigen::MatrixXd::Random(m_X.cols(), m_units);
        }
        else
        {
            m_X = X;
            m_W = Eigen::MatrixXd::Random(m_in_dims, m_units);
        }
    }

    Eigen::MatrixXd forward()
    {
        Eigen::MatrixXd g = m_X * m_W;

        if (m_activation == "linear")
        {
            return g;
        }
        else if (m_activation == "sigmoid")
        {
            return Activations::sigmoid(g);
        }
        else if (m_activation == "relu")
        {
            return Activations::relu(g);
        }
        else
        {
            throw std::invalid_argument("Unknown activation function: " + m_activation);
        }
    }
};
