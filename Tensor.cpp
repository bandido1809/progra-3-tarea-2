#include "Tensor.h"

#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>

// Producto de las dimensiones = numero total de elementos
size_t Tensor::numElements(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

// Constructor interno que toma posesion de un buffer ya reservado
Tensor::Tensor(double* data, std::vector<size_t> shape, size_t size)
    : shape_(std::move(shape)), data_(data), size_(size) {}

// Constructor principal
Tensor::Tensor(const std::vector<size_t>& shape,
               const std::vector<double>& values) {
    if (shape.empty() || shape.size() > 3)
        throw std::invalid_argument("La forma debe tener entre 1 y 3 dimensiones.");
    for (size_t d : shape)
        if (d == 0)
            throw std::invalid_argument("Ninguna dimension puede ser 0.");

    shape_ = shape;
    size_  = numElements(shape_);

    if (values.size() != size_)
        throw std::invalid_argument("El numero de valores no coincide con las dimensiones.");

    data_ = new double[size_];
    std::copy(values.begin(), values.end(), data_);
}

// Constructor de copia: copia profunda
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_) {
    if (other.data_) {
        data_ = new double[size_];
        std::copy(other.data_, other.data_ + size_, data_);
    }
}

// Constructor de movimiento: transfiere el puntero y deja el origen nulo
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.shape_.clear();
}

// Asignacion por copia: libera lo actual, copia y maneja auto-asignacion
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    delete[] data_;
    data_ = nullptr;

    shape_ = other.shape_;
    size_  = other.size_;
    if (other.data_) {
        data_ = new double[size_];
        std::copy(other.data_, other.data_ + size_, data_);
    }
    return *this;
}

// Asignacion por movimiento: libera lo actual y toma posesion del temporal
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    delete[] data_;

    shape_ = std::move(other.shape_);
    data_  = other.data_;
    size_  = other.size_;

    other.data_ = nullptr;
    other.size_ = 0;
    other.shape_.clear();
    return *this;
}

// Destructor
Tensor::~Tensor() {
    delete[] data_;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    return Tensor(shape, std::vector<double>(numElements(shape), 0.0));
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    return Tensor(shape, std::vector<double>(numElements(shape), 1.0));
}

Tensor Tensor::random(const std::vector<size_t>& shape,
                      double minVal, double maxVal) {
    if (minVal > maxVal)
        throw std::invalid_argument("min no puede ser mayor que max.");

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(minVal, maxVal);

    std::vector<double> values(numElements(shape));
    for (double& v : values) v = dist(gen);
    return Tensor(shape, values);
}

Tensor Tensor::arange(double start, double end, double step) {
    if (step == 0.0)
        throw std::invalid_argument("El paso no puede ser 0.");

    std::vector<double> values;
    if (step > 0)
        for (double x = start; x < end; x += step) values.push_back(x);
    else
        for (double x = start; x > end; x += step) values.push_back(x);

    return Tensor({values.size()}, values);
}

// Operacion elemento a elemento con broadcasting (hasta 3D)
Tensor Tensor::broadcastOp(const Tensor& a, const Tensor& b, char op) {
    // Rellena una forma a 3 dimensiones anteponiendo 1s
    auto pad3 = [](const std::vector<size_t>& s) {
        std::vector<size_t> r(3, 1);
        size_t off = 3 - s.size();
        for (size_t i = 0; i < s.size(); ++i) r[off + i] = s[i];
        return r;
    };

    std::vector<size_t> sa = pad3(a.shape_);
    std::vector<size_t> sb = pad3(b.shape_);
    std::vector<size_t> res(3);

    for (int i = 0; i < 3; ++i) {
        if (sa[i] == sb[i])       res[i] = sa[i];
        else if (sa[i] == 1)      res[i] = sb[i];
        else if (sb[i] == 1)      res[i] = sa[i];
        else throw std::invalid_argument("Dimensiones incompatibles.");
    }

    size_t N = res[0] * res[1] * res[2];
    double* out = new double[N];

    size_t idx = 0;
    for (size_t i = 0; i < res[0]; ++i)
        for (size_t j = 0; j < res[1]; ++j)
            for (size_t k = 0; k < res[2]; ++k) {
                size_t ai = (sa[0] == 1 ? 0 : i);
                size_t aj = (sa[1] == 1 ? 0 : j);
                size_t ak = (sa[2] == 1 ? 0 : k);
                size_t bi = (sb[0] == 1 ? 0 : i);
                size_t bj = (sb[1] == 1 ? 0 : j);
                size_t bk = (sb[2] == 1 ? 0 : k);

                double va = a.data_[ai * sa[1] * sa[2] + aj * sa[2] + ak];
                double vb = b.data_[bi * sb[1] * sb[2] + bj * sb[2] + bk];

                double r = 0.0;
                switch (op) {
                    case '+': r = va + vb; break;
                    case '-': r = va - vb; break;
                    case '*': r = va * vb; break;
                }
                out[idx++] = r;
            }

    size_t resRank = std::max(a.shape_.size(), b.shape_.size());
    std::vector<size_t> outShape(res.end() - resRank, res.end());
    return Tensor(out, outShape, N);
}

Tensor Tensor::operator+(const Tensor& other) const { return broadcastOp(*this, other, '+'); }
Tensor Tensor::operator-(const Tensor& other) const { return broadcastOp(*this, other, '-'); }
Tensor Tensor::operator*(const Tensor& other) const { return broadcastOp(*this, other, '*'); }

Tensor Tensor::operator*(double scalar) const {
    double* out = new double[size_];
    for (size_t i = 0; i < size_; ++i) out[i] = data_[i] * scalar;
    return Tensor(out, shape_, size_);
}

Tensor operator*(double scalar, const Tensor& t) { return t * scalar; }

// view: reinterpreta la forma sin copiar datos (traslada el buffer)
Tensor Tensor::view(const std::vector<size_t>& newShape) {
    if (newShape.empty() || newShape.size() > 3)
        throw std::invalid_argument("La nueva forma debe tener entre 1 y 3 dimensiones.");
    if (numElements(newShape) != size_)
        throw std::invalid_argument("El numero total de elementos debe mantenerse constante.");

    double* d = data_;
    size_t  n = size_;

    data_ = nullptr;
    size_ = 0;
    shape_.clear();

    return Tensor(d, newShape, n);
}

// unsqueeze: inserta una dimension de tamano 1 en 'axis'
Tensor Tensor::unsqueeze(size_t axis) {
    if (shape_.size() + 1 > 3)
        throw std::invalid_argument("El resultado excederia 3 dimensiones.");
    if (axis > shape_.size())
        throw std::invalid_argument("Posicion de eje invalida.");

    std::vector<size_t> ns = shape_;
    ns.insert(ns.begin() + axis, 1);

    double* d = data_;
    size_t  n = size_;

    data_ = nullptr;
    size_ = 0;
    shape_.clear();

    return Tensor(d, ns, n);
}

// concat: une varios tensores a lo largo de un eje
Tensor Tensor::concat(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty())
        throw std::invalid_argument("La lista de tensores esta vacia.");

    size_t rank = tensors[0].shape_.size();
    if (axis >= rank)
        throw std::invalid_argument("Eje fuera de rango.");

    for (const Tensor& t : tensors) {
        if (t.shape_.size() != rank)
            throw std::invalid_argument("Todos los tensores deben tener el mismo numero de dimensiones.");
        for (size_t d = 0; d < rank; ++d)
            if (d != axis && t.shape_[d] != tensors[0].shape_[d])
                throw std::invalid_argument("Dimensiones incompatibles fuera del eje de concatenacion.");
    }

    std::vector<size_t> outShape = tensors[0].shape_;
    size_t axisSum = 0;
    for (const Tensor& t : tensors) axisSum += t.shape_[axis];
    outShape[axis] = axisSum;

    size_t N = numElements(outShape);
    double* out = new double[N];

    auto pad3 = [](const std::vector<size_t>& s) {
        std::vector<size_t> r(3, 1);
        size_t off = 3 - s.size();
        for (size_t i = 0; i < s.size(); ++i) r[off + i] = s[i];
        return r;
    };

    std::vector<size_t> outPad = pad3(outShape);
    size_t s2 = 1;
    size_t s1 = outPad[2];
    size_t s0 = outPad[1] * outPad[2];
    size_t paddedAxis = axis + (3 - rank);

    size_t offset = 0;
    for (const Tensor& t : tensors) {
        std::vector<size_t> tp = pad3(t.shape_);
        for (size_t i = 0; i < tp[0]; ++i)
            for (size_t j = 0; j < tp[1]; ++j)
                for (size_t k = 0; k < tp[2]; ++k) {
                    size_t di = i, dj = j, dk = k;
                    if      (paddedAxis == 0) di += offset;
                    else if (paddedAxis == 1) dj += offset;
                    else                      dk += offset;

                    size_t destIdx = di * s0 + dj * s1 + dk * s2;
                    size_t srcIdx  = i * tp[1] * tp[2] + j * tp[2] + k;
                    out[destIdx]   = t.data_[srcIdx];
                }
        offset += tp[paddedAxis];
    }

    return Tensor(out, outShape, N);
}

Tensor Tensor::relu() const {
    double* out = new double[size_];
    for (size_t i = 0; i < size_; ++i)
        out[i] = data_[i] > 0.0 ? data_[i] : 0.0;
    return Tensor(out, shape_, size_);
}

Tensor Tensor::sigmoid() const {
    double* out = new double[size_];
    for (size_t i = 0; i < size_; ++i)
        out[i] = 1.0 / (1.0 + std::exp(-data_[i]));
    return Tensor(out, shape_, size_);
}

// Producto punto entre dos tensores del mismo numero de elementos
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.size_ != b.size_)
        throw std::invalid_argument("Los tensores deben tener el mismo numero de elementos.");

    double s = 0.0;
    for (size_t i = 0; i < a.size_; ++i) s += a.data_[i] * b.data_[i];

    double* out = new double[1];
    out[0] = s;
    return Tensor(out, {1}, 1);
}

// Multiplicacion matricial entre dos tensores 2D compatibles
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2)
        throw std::invalid_argument("Ambos operandos deben ser bidimensionales.");

    size_t m = a.shape_[0], n = a.shape_[1];
    size_t n2 = b.shape_[0], p = b.shape_[1];
    if (n != n2)
        throw std::invalid_argument("Dimensiones internas incompatibles.");

    double* out = new double[m * p];
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += a.data_[i * n + k] * b.data_[k * p + j];
            out[i * p + j] = s;
        }

    return Tensor(out, {m, p}, m * p);
}

std::string Tensor::shapeStr() const {
    std::ostringstream os;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) os << "x";
        os << shape_[i];
    }
    if (shape_.empty()) os << "vacio";
    return os.str();
}

void Tensor::printSize(const std::string& label) const {
    std::cout << label << " -> shape = [" << shapeStr()
              << "], size = " << size_ << "\n";
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape=[" << t.shapeStr() << "], size=" << t.size_ << ")";
    if (t.size_ <= 24 && t.data_) {
        os << " datos = [";
        for (size_t i = 0; i < t.size_; ++i) {
            if (i) os << ", ";
            os << t.data_[i];
        }
        os << "]";
    }
    return os;
}
