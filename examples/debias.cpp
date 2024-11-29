#include <iostream>
#include <array>
#include <cmath>

#include "../utils/tinyqr.h"

template <typename T> constexpr T cpow(T base, int exp) {
  if (exp == 0)
    return 1.0; // base case for zero exponent
  if (exp < 0)
    return 1.0 / cpow(base, -exp);
  T result = 1.0;
  while (exp > 0) {
    if (exp % 2 == 1) {
      result *= base;
    }
    base *= base;
    exp /= 2;
  }
  return result;
}

// Utility to compute power of 2 at compile time
template<typename T> constexpr T cpow2(int n) {
  return (n == 0) ? 1.0 : (n > 0) ? 2.0 * cpow2<T>(n - 1) : 0.5 * cpow2<T>(n + 1);
}

// Utility to compute absolute value at compile time
template<typename T> constexpr T cabs(T x) {
  return (x < 0) ? -x : x;
}

// Helper function to approximate logarithm using Richardson extrapolation
template<typename T> constexpr T crichardson_log(T x) {
  return (x - 1) - ((x - 1) * (x - 1) / 2.0) + ((x - 1) * (x - 1) * (x - 1) / 3.0) - ((x - 1) * (x - 1) * (x - 1) * (x - 1) / 4.0);
}

// Compute exp(x) using Taylor series expansion
template<typename T> constexpr T cexp(T x, T epsilon = 1e-10, const size_t max_iter = 100) {
  T result = 1.0;  // e^0 = 1
  T term = 1.0;    // First term is x^0 / 0! = 1
  for (size_t n = 1; n < max_iter; ++n) {
    term *= x / n;     // Compute the next term in the series
    result += term;    // Add the term to the result
    if(cabs(term) < epsilon) break;
  }
  return result;
}

template<typename T>
constexpr T clog(T x, T epsilon = 1e-3, const size_t iter = 100) {
  T x_ = x;
  if (x <= 0.0) {
    // Log is not defined for non-positive values.
    return std::numeric_limits<T>::quiet_NaN();
  } else if (x == 1.0) {
    return 0.0;
  }
  else if (x == 2.0) {
    return 0.693147180559945309417232121458; // ln(2)
  }
  T result = 0;
  int exponent = 0;
  // Scale x to the range [1, 2) by dividing by powers of 2
  while (x < 1) {
    x *= 2;
    --exponent;
  }
  while (x >= 2) {
    x /= 2;
    ++exponent;
  }

  // Use a polynomial approximation for the logarithm in the range [1, 2)
  const T log_scale = exponent * 0.6931472;
  T log_x = crichardson_log(x);
  result = log_scale + log_x;
  for(size_t i = 0; i < iter; i++) {
    T exp_result = cexp(result);
    if(cabs(exp_result - x_) < epsilon) break;
    result -= (exp_result - x) / exp_result;
  }
  return result;
}

// Template to get debias scaling at compile-time
template<int debias_order>
constexpr std::array<double, debias_order+1> get_scaling() {
  std::array<double, debias_order+1> scaling{};
  for (int r = 0; r <= debias_order; ++r) {
    scaling[r] = cpow(1.5, r);
  }
  return scaling;
}

// Calculate the debias coefficients at compile-time
template<int debias_order>
constexpr std::array<double, debias_order+1> get_coeffs() {
  constexpr std::array<double, debias_order+1> scaling = get_scaling<debias_order>();

  std::array<double, (debias_order+1) * (debias_order+1)> A{};
  for (int r = 0; r <= debias_order; ++r) {
    for (int s = 0; s <= debias_order; ++s) {
      A[r * (debias_order+1) + s] = cpow(scaling[r], - 2 * s);
    }
  }
  std::array<double, debias_order+1> e0{};
  e0[0] = 1.0;
  return tinyqr::clm<double, debias_order+1, debias_order+1>(A, e0);
}

template <int J>
constexpr std::pair<std::array<double, J+1>, std::array<std::array<double, J + 1>, J + 1>>
get_V_omega_inner() {
  std::array<double, J+1> a = {};
  for (int r = 0; r <= J; ++r) a[r] = cpow(0.95, r);
  std::array<double, (J+1) * (J+1)> A = {};
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      A[r * (J+1) + s] = cpow(a[r], 2 * s);
    }
  }
  std::array<double, J+1> e0{};
  e0[0] = 1.0;
  auto coeff = tinyqr::clm<double, J+1, J+1>(A, e0);
  std::array<std::array<double, J + 1>, J + 1> V = {};
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      V[r][s] = 2.0 / (3 * a[r]) * (1 - ((a[s]/a[r]) * clog(a[r]/a[s] + 1.)));
    }
  }

  for (int r = 0; r <= J; ++r) {
    for (int s = r+1; s <= J; ++s) {
      V[r][s] += V[s][r];
    }
  }
  for(int r =0; r <= J; r++) {
    V[r][r] = V[r][r] * 2.;
  }
  for (int s = 0; s <= J; ++s) {
    for (int r = s; r <= J; ++r) {
      V[r][s] = V[s][r];
    }
  }
  return {coeff, V};
}

template <int J>
double get_V_omega(const size_t d) {
  auto [debias_coeffs,V] = get_V_omega_inner<J>();
  double result = 0.;
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      result += std::pow(V[r][s], static_cast<double>(d)) * debias_coeffs[r] * debias_coeffs[s];
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
template<const size_t n> constexpr std::array<double, n*n> make_A() {
  constexpr std::array<double, n> scaling = get_scaling<n-1>();
  std::array<double, n * n> A{};
  for (int r = 0; r < n; ++r) {
    for (int s = 0; s < n; ++s) {
      A[r * n + s] = cpow(scaling[r], - 2 * s);
    }
  }
  return A;
}

template<const size_t n> constexpr std::array<double, n> make_e0() {
  std::array<double, n> e0{};
  e0[0] = 1.0;
  return e0;
}

template <typename T>
[[maybe_unused]] void print_vector(const std::vector<T>& vec) {
  std::cout << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << vec[i];
    if (i < vec.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
/*
int main() {
  constexpr int debias_order = 2;
  constexpr int d = 2;
  constexpr int J = debias_order;

  std::cout <<std::setprecision(10);

  double V_omega_ = get_V_omega<debias_order>(2);
  std::cout << "V_omega: " << V_omega_ << std::endl;
  auto [coeff, V_omega] = get_V_omega_inner<debias_order>();
  for(size_t i = 0; i < debias_order+1; i++) {
    for(size_t j = 0; j < debias_order+1; j++) {
      std::cout << V_omega[i][j] << ",";
    }
    std::cout << std::endl;
  }


  std::cout << "\n";
  for(size_t i = 0; i < debias_order+1; i++) {
    std::cout << coeff[i] << ",";
  }
  std::cout << "\n\n";




    std::array<double, debias_order+1> a = {};
    for (int r = 0; r <= debias_order; ++r) a[r] = cpow(0.95, r);

  for(size_t i = 0; i < a.size(); i++) std::cout << a[i] << ",";
  std::cout << "\n";


  std::array<double, J+1> a = {};
  for (int r = 0; r <= J; ++r) a[r] = cpow(0.95, r);

  std::array<double, (J+1) * (J+1)> A = {};
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      A[r * (J+1) + s] = cpow(a[r], 2 * s);
    }
  }
  std::cout << "\n";
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      std::cout << A[r * (J+1) + s] << ",";
    }
    std::cout<< std::endl;
  }
  std::cout << "\n";

  //2 / (3 * a[r]) * (1 - a[s] / a[r] * log(a[r] / a[s] + 1))
  std::array<std::array<double, debias_order + 1>, debias_order + 1> V = {};
  for (int r = 0; r <= debias_order; ++r) {
    for (int s = 0; s <= debias_order; ++s) {
      V[r][s] = 2.0 / (3 * a[r]) * (1 - ((a[s]/a[r]) * clog(a[r]/a[s] + 1.)));
      //std::cout << "Log inputs: \n"<< a[r] << ", "<< a[s] << "\n";
      //std::cout << clog(a[r]/a[s] + 1.) //<< " " <<a[r] << " " << a[s]
      //          <<  ",";
      //std::cout << V[r][s] << ",";
    }
    std::cout << "\n";
  }
  std::cout << "\n";



  // Ensure constexpr functions are called to avoid "unreachable" warnings
  constexpr auto dummy_scaling = get_scaling<debias_order>();
  constexpr auto dummy_coeffs = get_coeffs<debias_order>();
  for(const auto& coef: dummy_coeffs) {
    std::cout << coef << ",";
  }
  std::cout << "\n";

  constexpr std::array<double, debias_order+1> scaling = get_scaling<debias_order>();
  for(const auto s: scaling) {
    std::cout << s << ",";
  }
  std::cout << "\n";

  constexpr std::array<double, debias_order+1> scaling = get_scaling<debias_order>();
  std::cout << "A:\n ";
  constexpr auto A = make_A<debias_order+1>();
  constexpr auto e0 = make_e0<debias_order+1>();
  std::cout << "\n";
  std::vector<double> A_ = {};
  for(size_t i = 0; i < A.size(); i++) {
    A_.push_back(A[i]);
  }
  print_vector(A_);
  std::vector<double> e0_ = {};
  for(size_t i = 0; i < e0.size(); i++) {
    e0_.push_back(e0[i]);
  }
  print_vector(e0_);

  std::cout << "compile time solution:\n";

  constexpr auto solution = tinyqr::compile_time::clm<double, debias_order+1, debias_order+1>(A, e0);
  std::vector<double> sol = {};
  for(size_t i = 0; i < solution.size(); i++) {
    sol.push_back(solution[i]);
  }
  print_vector(sol);

  constexpr auto qr_ = tinyqr::compile_time::cqr_decomposition<double, debias_order+1, debias_order+1>(A);
  std::array<double, 3*3> Q_ =qr_.Q;
  std::array<double, 3*3> R_ =qr_.R;

  std::cout << "Ctime QR:\n";
  std::cout << "Q:\n";
  for(size_t i = 0; i < Q_.size(); i++) {
    std::cout << Q_[i] << ",";
  }
  std::cout << "\n";
  std::cout << "R:\n";
  for(size_t i = 0; i < R_.size(); i++) {
    std::cout << R_[i] << ",";
  }
  std::cout << "\n";

  auto qr_2 = tinyqr::qr_decomposition<double>(A_, debias_order+1, debias_order+1);
  std::vector<double> Q_2 =qr_2.Q;
  std::vector<double> R_2 =qr_2.R;
  std::cout << "QR:\n";
  std::cout << "Q:\n";
  for(size_t i = 0; i < Q_2.size(); i++) {
    std::cout << Q_2[i] << ",";
  }
  std::cout << "\n";
  std::cout << "R:\n";
  for(size_t i = 0; i < R_2.size(); i++) {
    std::cout << R_2[i] << ",";
  }
  std::cout << "\n";


  std::cout << "Runtime solution:\n";
  const auto solution2 = tinyqr::lm(A_, e0_);
  print_vector(solution2);


 void std::vector<std::pair<std::pair<unsigned long, unsigned long>,
 std::vector<float, std::allocator<float> > >,
 std::allocator<std::pair<std::pair<unsigned long, unsigned long>,
 std::vector<float, std::allocator<float> > > > >::_M_realloc_insert<std::pair<unsigned long, unsigned long>,
 std::vector<float, std::allocator<float> > const&>(
 __gnu_cxx::__normal_iterator<std::pair<std::pair<unsigned long, unsigned long>,
 std::vector<float, std::allocator<float> > >*,
 std::vector<std::pair<std::pair<unsigned long, unsigned long>,
 std::vector<float, std::allocator<float> > >, std::allocator<std::pair<std::pair<unsigned long, unsigned long>,
 std::vector<float, std::allocator<float> > > > > >,
 std::pair<unsigned long, unsigned long>&&, std::vector<float, std::allocator<float> > const&)


  double V_omega = get_V_omega<debias_order>(debias_order);
  constexpr double omega_bar = get_omega_bar<debias_order>();

  std::cout << "V_omega: " << V_omega << std::endl;
  std::cout << "omega_bar: " << omega_bar << std::endl;

  return 0;
}
*/