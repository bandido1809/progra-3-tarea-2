#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <string>
#include <iostream>

// Clase Tensor: maneja tensores de hasta 3 dimensiones.
// Los datos se guardan en un arreglo dinamico contiguo (double*)
// en orden row-major: para {d0,d1,d2} el elemento (i,j,k) esta en
// la posicion i*(d1*d2) + j*d2 + k.
class Tensor {
public:
    // Constructor principal
    Tensor(const std::vector<size_t>& shape,
           const std::vector<double>& values);

    // Ciclo de vida (regla de los 5)
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // Creacion de tensores pre-definidos
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor random(const std::vector<size_t>& shape,
                         double minVal, double maxVal);
    static Tensor arange(double start, double end, double step = 1.0);

    // Operadores
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;   // elemento a elemento
    Tensor operator*(double scalar) const;

    // Modificacion de dimensiones
    Tensor view(const std::vector<size_t>& newShape);
    Tensor unsqueeze(size_t axis);

    // Concatenacion
    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis);

    // Activaciones
    Tensor relu() const;
    Tensor sigmoid() const;

    // Funciones amigas
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    // Utilidades
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    std::string shapeStr() const;
    void printSize(const std::string& label) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    std::vector<size_t> shape_;
    double*             data_ = nullptr;
    size_t              size_ = 0;

    Tensor() = default;
    Tensor(double* data, std::vector<size_t> shape, size_t size);

    static size_t numElements(const std::vector<size_t>& shape);
    static Tensor broadcastOp(const Tensor& a, const Tensor& b, char op);
};

Tensor operator*(double scalar, const Tensor& t);

#endif
