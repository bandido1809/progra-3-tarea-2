# Tensor++

Librería de tensores en C++ (hasta 3 dimensiones) para la Tarea #2 de
Programación III.

**Integrantes:**
- (Nombre y apellido)
- (Nombre y apellido)

## Descripción

La clase `Tensor` maneja tensores de 1, 2 o 3 dimensiones usando un arreglo
dinámico contiguo (`double*`) en orden row-major. Incluye fábricas estáticas,
operadores sobrecargados, cambio de dimensiones, concatenación, producto punto,
multiplicación matricial y las activaciones ReLU y Sigmoid.

## Archivos

- `Tensor.h` — declaración de la clase.
- `Tensor.cpp` — implementación.
- `main.cpp` — pruebas de cada método y la red neuronal.
- `CMakeLists.txt` — configuración de compilación.

## Compilación y ejecución

Con CMake:

```bash
mkdir build && cd build
cmake ..
cmake --build .
./tensorpp
```

Con g++ directamente:

```bash
g++ -std=c++17 Tensor.cpp main.cpp -o tensorpp
./tensorpp
```

## Red neuronal

| Paso | Operación           | Dimensión      |
|------|---------------------|----------------|
| 1    | Entrada             | 1000 × 20 × 20 |
| 2    | view                | 1000 × 400     |
| 3    | matmul con W1       | 1000 × 100     |
| 4    | + bias b1 (1 × 100) | 1000 × 100     |
| 5    | ReLU                | 1000 × 100     |
| 6    | matmul con W2       | 1000 × 10      |
| 7    | + bias b2 (1 × 10)  | 1000 × 10      |
| 8    | Sigmoid             | 1000 × 10      |
