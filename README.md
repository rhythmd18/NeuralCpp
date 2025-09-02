# NeuralCpp

A minimal neural network library in modern C++14 built on top of Eigen. It provides simple building blocks (layers, loss functions, optimizers) and a Sequential container to assemble and train feed‑forward models.

## Features
- Sequential model container
- Layers: Linear (fully‑connected), ReLU, Sigmoid, Tanh, Softmax
- Loss: Binary Cross Entropy (BCE)
- Optimizer: SGD with momentum
- Uses Eigen (MatrixXd) as the tensor backend

## Requirements
- C++14‑compatible compiler (GCC/Clang/MSVC)
- Eigen 3.3+ (header‑only)

Install Eigen:
- Ubuntu/Debian: `sudo apt-get install libeigen3-dev` (headers in `/usr/include/eigen3`)
- macOS (Homebrew): `brew install eigen`
- Windows (vcpkg): `vcpkg install eigen3` and integrate vcpkg
- From source: https://eigen.tuxfamily.org

## Repository layout
- `include/` Public headers (nn core types, layers, optimizers, criteria)
- `src/`      Library sources
- `Main.cpp`  Example program (can be used as your app entry point)

Key headers and sources used by the example:
- Headers: `include/nn/Sequential.h`, `include/nn/layers/Layers.h`, `include/nn/criteria/BinaryCrossEntropyLoss.h`, `include/nn/optimizers/SGD.h`
- Sources: `src/nn/Sequential.cpp`, `src/nn/layers/Linear.cpp`, `src/nn/layers/ReLU.cpp`, `src/nn/layers/Sigmoid.cpp`, `src/nn/layers/Tanh.cpp`, `src/nn/layers/Softmax.cpp`, `src/nn/criteria/BinaryCrossEntropyLoss.cpp`, `src/nn/optimizers/Optimizer.cpp`, `src/nn/optimizers/SGD.cpp`

## Usage
A minimal end‑to‑end example (see `Main.cpp`) training a small MLP with BCE and SGD:
```cpp
#include <iostream>
#include <Eigen/Dense>
#include "include/nn/layers/Layers.h"
#include "include/nn/Sequential.h"
#include "include/nn/criteria/BinaryCrossEntropyLoss.h"
#include "include/nn/optimizers/SGD.h"

void train(Eigen::MatrixXd& X, Eigen::MatrixXd& y,
           Sequential& model, Criterion& loss_fn, Optimizer& optimizer,
           int epochs = 100) {
  for (int i = 0; i < epochs; ++i) {
    Eigen::MatrixXd out = model(X);
    double loss = loss_fn(y, out);

    model.backward(loss_fn);  // gradients
    optimizer.step();         // update
    optimizer.zero_grad();    // clear

    if (i % 100 == 0) std::cout << "Epoch: " << i << ", Loss: " << loss << "\n";
  }
}

int main() {
  Eigen::MatrixXd X(8, 3);
  X << 0,0,0,
       0,0,1,
       0,1,0,
       0,1,1,
       1,0,0,
       1,0,1,
       1,1,0,
       1,1,1;

  Eigen::MatrixXd y(8, 1);
  y << 0,1,1,0,1,0,0,1;

  Sequential model(
    Linear(3, 4), ReLU(),
    Linear(4, 4), ReLU(),
    Linear(4, 1), Sigmoid()
  );

  BinaryCrossEntropyLoss loss_fn;
  SGD optimizer(model, 0.1, 0.9);

  train(X, y, model, loss_fn, optimizer, 10000);
  std::cout << "\nOutput:\n" << model(X) << "\n";
}
```

## API overview
- Tensor type: `Eigen::MatrixXd` (rows = batch size; cols = features/units)
- `Sequential`:
  - Construct with layers: `Sequential(Layers... layers)`
  - Forward: `Eigen::MatrixXd operator()(const Eigen::MatrixXd&)`
  - Backward: `void backward(Criterion&)`
- Layers (derive from `Layer`): `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- Loss (derive from `Criterion`): `BinaryCrossEntropyLoss`
- Optimizer (derive from `Optimizer`): `SGD(model, lr, momentum)`

## Tips
- Ensure Eigen headers are on the include path (often `/usr/include/eigen3` or provided by vcpkg on Windows).
- Data is expected as one sample per row.
- Start with learning rates like 0.1 and adjust as needed.

## Extending
- New Layer: inherit from `Layer`, implement forward and `_backward`, store parameters/gradients as needed.
- New Loss: inherit from `Criterion`, implement loss computation and `_backward` (returns dL/dA for last layer output).
- New Optimizer: inherit from `Optimizer`, implement `step()` and any state (e.g., momentum).

## License
See the repository for licensing information.