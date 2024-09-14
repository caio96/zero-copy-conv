#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>

// Initialize data with random values from 0.0 to 10.0
inline void initialize_data(float *data, size_t size) {
  for (size_t i = 0; i < size; ++i)
    data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

#endif // UTILS_H
