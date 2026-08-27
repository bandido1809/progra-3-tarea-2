#include "Tensor.h"

#include <iostream>

// Pruebas de las funcionalidades de la clase Tensor
void pruebas() {
    std::cout << "--- Fabricas y constructor ---\n";
    Tensor A = Tensor::zeros({2, 3});
    Tensor B = Tensor::ones({3, 3});
    Tensor C = Tensor::random({2, 2}, 0.0, 1.0);
    Tensor D = Tensor::arange(0, 6);
    std::cout << "zeros  = " << A << "\n";
    std::cout << "ones   = " << B << "\n";
    std::cout << "random = " << C << "\n";
    std::cout << "arange = " << D << "\n";

    Tensor E({2, 2}, {1, 2, 3, 4});
    std::cout << "manual = " << E << "\n\n";

    std::cout << "--- Operadores ---\n";
    Tensor X({2, 2}, {1, 2, 3, 4});
    Tensor Y({2, 2}, {5, 6, 7, 8});
    std::cout << "X + Y   = " << (X + Y) << "\n";
    std::cout << "X - Y   = " << (X - Y) << "\n";
    std::cout << "X * Y   = " << (X * Y) << "\n";
    std::cout << "X * 2.0 = " << (X * 2.0) << "\n";
    Tensor bias({1, 2}, {10, 20});
    std::cout << "X + bias(1x2) = " << (X + bias) << "\n\n";

    std::cout << "--- view y unsqueeze ---\n";
    Tensor V = Tensor::arange(0, 12);
    Tensor V2 = V.view({3, 4});
    std::cout << "view({3,4}) = " << V2 << "\n";
    Tensor u1 = Tensor::arange(0, 3);
    Tensor r1 = u1.unsqueeze(0);
    Tensor u2 = Tensor::arange(0, 3);
    Tensor r2 = u2.unsqueeze(1);
    std::cout << "unsqueeze(0) shape = [" << r1.shapeStr() << "]\n";
    std::cout << "unsqueeze(1) shape = [" << r2.shapeStr() << "]\n\n";

    std::cout << "--- concat ---\n";
    Tensor c1 = Tensor::ones({2, 3});
    Tensor c2 = Tensor::zeros({2, 3});
    Tensor cc0 = Tensor::concat({c1, c2}, 0);
    Tensor c3 = Tensor::ones({2, 3});
    Tensor c4 = Tensor::zeros({2, 3});
    Tensor cc1 = Tensor::concat({c3, c4}, 1);
    std::cout << "concat(axis=0) shape = [" << cc0.shapeStr() << "]\n";
    std::cout << "concat(axis=1) shape = [" << cc1.shapeStr() << "]\n\n";

    std::cout << "--- dot y matmul ---\n";
    Tensor d1({3}, {1, 2, 3});
    Tensor d2({3}, {4, 5, 6});
    std::cout << "dot = " << dot(d1, d2) << "\n";
    Tensor m1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor m2({3, 2}, {7, 8, 9, 10, 11, 12});
    std::cout << "matmul = " << matmul(m1, m2) << "\n\n";
}

// Red neuronal:
// entrada 1000x20x20 -> view 1000x400 -> matmul W1(400x100)
// -> +b1(1x100) -> ReLU -> matmul W2(100x10) -> +b2(1x10) -> Sigmoid
void redNeuronal() {
    std::cout << "--- Red neuronal ---\n";

    Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    input.printSize("Paso 1  Entrada         ");

    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor b1 = Tensor::random({1, 100},  -0.1, 0.1);
    Tensor W2 = Tensor::random({100, 10},  -0.1, 0.1);
    Tensor b2 = Tensor::random({1, 10},    -0.1, 0.1);

    Tensor x = input.view({1000, 400});
    x.printSize("Paso 2  view            ");

    Tensor h1 = matmul(x, W1);
    h1.printSize("Paso 3  matmul W1       ");

    Tensor z1 = h1 + b1;
    z1.printSize("Paso 4  + bias b1       ");

    Tensor a1 = z1.relu();
    a1.printSize("Paso 5  ReLU            ");

    Tensor h2 = matmul(a1, W2);
    h2.printSize("Paso 6  matmul W2       ");

    Tensor z2 = h2 + b2;
    z2.printSize("Paso 7  + bias b2       ");

    Tensor out = z2.sigmoid();
    out.printSize("Paso 8  Sigmoid         ");
}

int main() {
    try {
        pruebas();
        redNeuronal();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
