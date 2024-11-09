#include <iostream>
#include <array>
#include <cmath>

#include "stopwatch.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ConstantParameter"
// Iterative constexpr implementation of the power function
constexpr double cpow(double base, int exp) {
  double result = 1.;
  while (exp > 0) {
    if (exp % 2 == 1) {
      result *= base;
    }
    base *= base;
    exp /= 2;
  }
  return result;
}
#pragma clang diagnostic pop

// constexpr implementation of fabs
constexpr double cfabs(double x) {
  return x < 0 ? -x : x;
}

// constexpr swap function
template <typename T>
constexpr void cswap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

// constexpr implementation of dot product
template<int N>
constexpr double cdot(const std::array<double, N>& a, const std::array<double, N>& b) {
  double result = 0;
  for (int i = 0; i < N; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

// constexpr implementation of the square root function
constexpr double csqrt(double x, double curr = 1.0, double prev = 0.0) {
  return curr == prev ? curr : csqrt(x, 0.5 * (curr + x / curr), curr);
}

// constexpr natural logarithm function using a series expansion
constexpr double clog(double x) {
  if (x <= 0) {
    return 0; // log is undefined for non-positive numbers, simplified for constexpr
  }

  // Using the identity log(x) = 2 * ( (x - 1) / (x + 1) + 1/3 * ((x - 1) / (x + 1))^3 + 1/5 * ((x - 1) / (x + 1))^5 + ... )
  double result = 0.0;
  double term = (x - 1) / (x + 1);
  double term2 = term * term;
  double numerator = term;
  double denominator = 1.0;

  for (int i = 0; i < 10; ++i) {
    result += numerator / denominator;
    numerator *= term2;
    denominator += 2.0;
  }

  return 2.0 * result;
}

// Template to get debias scaling at compile-time
template<int debias_order>
constexpr std::array<double, debias_order + 1> get_scaling() {
  std::array<double, debias_order + 1> scaling{};
  for (int r = 0; r <= debias_order; ++r) {
    scaling[r] = cpow(1.5, r);
  }
  return scaling;
}

// Gram-Schmidt process for QR Factorization
template<int debias_order>
constexpr void gram_schmidt(std::array<std::array<double, debias_order + 1>, debias_order + 1>& A,
                            std::array<std::array<double, debias_order + 1>, debias_order + 1>& Q,
                            std::array<std::array<double, debias_order + 1>, debias_order + 1>& R) {
  for (int k = 0; k <= debias_order; ++k) {
    double norm = 0;
    for (int i = 0; i <= debias_order; ++i) {
      Q[i][k] = A[i][k];
      norm += Q[i][k] * Q[i][k];
    }
    norm = csqrt(norm);

    for (int i = 0; i <= debias_order; ++i) {
      Q[i][k] /= norm;
    }
    for (int j = k + 1; j <= debias_order; ++j) {
      double dot_val = cdot<debias_order + 1>(Q[k], A[j]);
      for (int i = 0; i <= debias_order; ++i) {
        A[i][j] -= dot_val * Q[i][k];
      }
    }
  }

  for (int i = 0; i <= debias_order; ++i) {
    for (int j = 0; j <= debias_order; ++j) {
      R[i][j] = (i <= j) ? cdot<debias_order + 1>(Q[i], A[j]) : 0;
    }
  }
}

// Solve R * x = Q^T * b using back-substitution
template<int debias_order>
constexpr std::array<double, debias_order + 1> solve_upper(const std::array<std::array<double, debias_order + 1>, debias_order + 1>& R, const std::array<double, debias_order + 1>& Qt_b) {
  std::array<double, debias_order + 1> x{};
  for (int i = debias_order; i >= 0; --i) {
    double sum = 0;
    for (int j = i + 1; j <= debias_order; ++j) {
      sum += R[i][j] * x[j];
    }
    x[i] = (Qt_b[i] - sum) / R[i][i];
  }
  return x;
}

// Calculate the debias coefficients at compile-time
template<int debias_order>
constexpr std::array<double, debias_order + 1> get_coeffs() {
  constexpr std::array<double, debias_order + 1> scaling = get_scaling<debias_order>();

  std::array<std::array<double, debias_order + 1>, debias_order + 1> A{};
  for (int r = 0; r <= debias_order; ++r) {
    for (int s = 0; s <= debias_order; ++s) {
      A[r][s] = cpow(scaling[r], 2 - 2 * (s + 1));
    }
  }

  std::array<std::array<double, debias_order + 1>, debias_order + 1> Q{};
  std::array<std::array<double, debias_order + 1>, debias_order + 1> R{};
  gram_schmidt<debias_order>(A, Q, R);

  std::array<double, debias_order + 1> e0{};
  e0[0] = 1.0;

  std::array<double, debias_order + 1> Qt_e0{};
  for (int i = 0; i <= debias_order; ++i) {
    Qt_e0[i] = cdot<debias_order + 1>(Q[i], e0);
  }

  return solve_upper<debias_order>(R, Qt_e0);
}

template <int J>
constexpr std::array<std::array<double, J + 1>, J + 1>
get_V_omega_inner() {
  constexpr std::array<double, J + 1> a = []() {
    std::array<double, J + 1> arr = {};
    for (int r = 0; r <= J; ++r) {
      arr[r] = cpow(0.95, r);
    }
    return arr;
  }();

  std::array<std::array<double, J + 1>, J + 1> V = {};

  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      V[r][s] = 2.0 / (3 * a[r]) * (1 - a[s] / a[r] * clog(a[r] / a[s] + 1));
    }
  }

  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      V[r][s] += V[s][r];
    }
  }
  return V;
}

template <int J>
double get_V_omega(const size_t d) {
  constexpr auto V = get_V_omega_inner<J>();
  constexpr auto debias_coeffs = get_coeffs<J>();
  // only this computation actually depends on the data dimensionality
  double result = 0.0;
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      result += cpow(V[r][s], d) * debias_coeffs[r] * debias_coeffs[s];
    }
  }
  return result;
}

template <int debias_order>
constexpr double get_omega_bar() {
  constexpr auto debias_scaling = get_scaling<debias_order>();
  constexpr auto debias_coeffs = get_coeffs<debias_order>();

  double sum = 0.0;
  for (size_t i = 0; i <= debias_order; ++i) {
    sum += debias_coeffs[i] * cpow(debias_scaling[i], -2 * debias_order - 2);
  }
  return sum;
}
/*
int main() {
  constexpr int debias_order = 3;
  constexpr int d = 1;

  // Ensure constexpr functions are called to avoid "unreachable" warnings
  constexpr auto dummy_scaling = get_scaling<debias_order>();
  constexpr auto dummy_coeffs = get_coeffs<debias_order>();
  double V_omega;

  {
    Stopwatch sw;
    for(size_t i = 0; i < 100'000'000; i++) {
      V_omega = get_V_omega<debias_order>(d);
    }
  }
  constexpr double omega_bar = get_omega_bar<debias_order>();

  std::cout << "V_omega: " << V_omega << std::endl;
  std::cout << "omega_bar: " << omega_bar << std::endl;

  return 0;
}*/
