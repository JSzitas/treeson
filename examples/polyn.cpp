#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <random>

#include "debias.cpp"
#include "../utils/utils.h"
#include "../utils/stopwatch.h"
#include "../utils/tinyqr.h"

#include "../treeson.h"

template<typename T>void compare_two(
    const T& x, const T& y,
    double abs_tol = 1e-3,
    const double rel_tol = 0.01) {
  bool match = true;
  if (x.size() != y.size())
    std::cout << "  Sizes differ: " << x.size() << ", " << y.size() << "n";
  for (size_t i = 0; i < x.size(); i++) {
    if ((std::abs(x[i] - y[i]) > abs_tol) and (std::abs(1 - (x[i]/y[i])) > rel_tol)) {
      std::cout << "   Values differ: " << x[i] << " vs " << y[i] << "\n";
      match = false;
    }
  }
  if(match) std::cout << "Values match\n";
}

constexpr unsigned long long factorial(unsigned int n) {
  unsigned long long result = 1;
  for (unsigned int i = 2; i <= n; ++i) {
    result *= i;
  }
  return result;
}

std::vector<double> make_design_matrix_polynomial(const std::vector<double>& X_data, const size_t n, const size_t d, int debias_order) {
  size_t J = debias_order;
  const size_t ncol = (2 * J + 4) * d + 1; // Number of columns
  std::vector<double> design_matrix(n * ncol, 1.0);
  for (size_t j = 0; j < d; ++j) {
    for (size_t s = 1; s <= 2 * J + 4; ++s) {
      for (size_t i = 0; i < n; ++i) {
        design_matrix[(j * (2 * J + 4) + s) * n + i] = std::pow(X_data[j * n + i], s);
      }
    }
  }
  return design_matrix;
}


template<const int debias_order = 2>
std::pair<std::vector<double>, double>
get_derivative_estimates_polynomial(
    const std::vector<double>& X_data,
    const size_t n, const size_t d,
    const std::vector<double>& Y_data) {

  auto design_matrix = make_design_matrix_polynomial(X_data, n, d, debias_order);
  const auto regression_vector = tinyqr::lm(design_matrix, Y_data);
  const size_t ncol = design_matrix.size()/n;
  std::vector<double> derivative_estimates(n * d, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      double Xij = X_data[i + (j * n)];
      size_t start_idx = j * (2 * debias_order + 4) + (2 * debias_order + 1) + 1;
      double derivative_estimate = regression_vector[start_idx] +
                                   regression_vector[start_idx + 1] * Xij +
                                   regression_vector[start_idx + 2] * std::pow(Xij, 2) / 2.0;
      constexpr int mult = factorial((2 * debias_order) + 2);
      derivative_estimate *= mult;  // factorial(2 * debias_order + 2)
      derivative_estimates[j * n + i] = derivative_estimate;
    }
  }
  // also get variance estimates here 
  double sigma2_hat = 0.0;
  for (size_t i = 0; i < n; ++i) sigma2_hat += Y_data[i] * Y_data[i];

  for (size_t i = 0; i < n; ++i) {
    double design_times_regression = 0.0;
    for (size_t j = 0; j < ncol; ++j) {
      design_times_regression += design_matrix[j * n + i] * regression_vector[j];
    }
    sigma2_hat -= Y_data[i] * design_times_regression;
  }
  sigma2_hat /= static_cast<double>(n - ncol);
  return {derivative_estimates, sigma2_hat};
}


void generate_data(size_t nrow, size_t ncol, std::vector<double>& X_data, std::vector<double>& Y_data) {
  std::mt19937 gen(42); // Standard mersenne_twister_engine seeded with 42
  std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
  std::uniform_real_distribution<> eps_dist(-std::sqrt(3.0), std::sqrt(3.0));

  auto mu = [](const std::vector<double>& x) {
    // Function mu representing 3 * x[0]^2
    return 3.0 * x[0] * x[0];
  };

  auto sigma2 = [](const std::vector<double>& x) {
    // Function sigma2 representing 1 / 100
    return 1.0 / 100.0;
  };

  X_data.resize(nrow * ncol);
  Y_data.resize(nrow);

  for (size_t i = 0; i < nrow; ++i) {
    std::vector<double> x(ncol);
    for (size_t j = 0; j < ncol; ++j) {
      x[j] = uniform_dist(gen);
    }
    double y = mu(x) + eps_dist(gen) * std::sqrt(sigma2(x));
    for (size_t j = 0; j < ncol; ++j) {
      X_data[i * ncol + j] = x[j];
    }
    Y_data[i] = y;
  }
}

template <int debias_order>
double select_lifetime_polynomial(const std::vector<double>& X_data, const std::vector<double>& Y_data) {
  const size_t nrow = Y_data.size();
  const size_t ncol = X_data.size()/nrow;
  auto [derivative_estimates, sigma2_hat] = get_derivative_estimates_polynomial<debias_order>(X_data, nrow, ncol, Y_data);

  constexpr double omega_bar = get_omega_bar<debias_order>();
  //std::cout << "Omega bar: "<< omega_bar << std::endl;

  constexpr double numerator_ = (4 * debias_order + 4) * cpow(omega_bar,2) / cpow(debias_order + 2,2);
  //std::cout << "Numerator: "<< numerator_ << std::endl;
  double numerator = numerator_;
  //for()
  //std::cout << "\n";
  //print_vector(derivative_estimates);
  //std::cout << "\n";
  double temp = 0., numerator_temp = 0.;
  for(size_t i = 0; i < nrow; i++) {
    temp = 0.;
    for(size_t j = 0; j < ncol; j++) {
      temp += derivative_estimates[j*nrow + i];
    }
    numerator_temp += std::pow(temp,2);
  }
  numerator *= numerator_temp;


  /*numerator *= std::accumulate(derivative_estimates.begin(), derivative_estimates.end(), 0.0, [](double sum, double val) {
    return sum + pow(val,2);
  });*/

  //std::cout << "sigma2 hat: " << sigma2_hat << std::endl;
  //std::cout << "Numerator: "<< numerator << std::endl;
  const double V_omega = get_V_omega<debias_order>(ncol);
  std::cout << "V omega: "<< V_omega << std::endl;
  double denominator = ncol * sigma2_hat * V_omega;
  std::cout << "denominator: "<< denominator << std::endl;
  double lambda_hat = std::pow(numerator / denominator, 1.0 / (4 * debias_order + 4 + ncol));

  return lambda_hat;
}



int main() {
  constexpr int debias_order = 2;
  //for(const int ncol : {2}) {//,5,10,20,50,100}) {
  //  for(const int nrow : {100}) {//,200,500,1'000,5'000,10'000,100'000}) {
      std::vector<double> X_data;
      std::vector<double> Y_data;
      //std::cout << "Nrow: " << nrow << ", ncol: " << ncol << "\n";
      // Generate example data
//generate_data(nrow, ncol, X_data, Y_data);
      auto data = CSVLoader<int, double>::load_data<true, false>("test_data_polynomial.csv", 10000);
      //print_data(data);
      //X_data.resize(X.size() * std::get<std::vector<double>>(X[0]).size());
      //Y_data.resize(std::get<std::vector<double>>(X[0]).size());
      size_t which_col = 0;
      for(const auto& col : data) {
        if(which_col < data.size()-1) {
          const auto& vec = std::get<std::vector<double>>(col);
          for(const auto& val: vec) {
            X_data.push_back(val);
          }
        }
        else {
          const auto& vec = std::get<std::vector<double>>(col);
          for(const auto& val: vec) {
            Y_data.push_back(val);
          }
        }
        which_col++;
      }
      const size_t nrow = Y_data.size();
      const size_t ncol = X_data.size()/nrow;

      std::cout << "Nrow: " << nrow << ", ncol: " << ncol << "\n";


      //treeson::utils::print_vector(X_data);
      //treeson::utils::print_vector(Y_data);

      //treeson::utils::print_vector(tinyqr::lm(X_data, Y_data));


      auto design_mat = CSVLoader<int, double>::load_data<true, false>("test_data_design_mat.csv", 10000);
      std::vector<double> d_mat;
      for(const auto& col : design_mat) {
          const auto& vec = std::get<std::vector<double>>(col);
          for(const auto& val: vec) {
            d_mat.push_back(val);
          }
      }
      auto deriv_est = CSVLoader<int, double>::load_data<true, false>("test_data_deriv_est.csv", 10000);
      std::vector<double> der_mat;
      for(const auto& col : deriv_est) {
        const auto& vec = std::get<std::vector<double>>(col);
        for(const auto& val: vec) {
          der_mat.push_back(val);
        }
      }
      std::cout << "Design matrix test: ";
      compare_two(d_mat, make_design_matrix_polynomial(X_data, nrow, ncol, debias_order));

      auto design_matrix = make_design_matrix_polynomial(X_data, nrow, ncol, debias_order);
      const auto regression_vector = tinyqr::lm(design_matrix, Y_data);
      auto reg_est = CSVLoader<int, double>::load_data<true, false>("test_data_reg_coef.csv", 10000);
      std::vector<double> reg_vec = std::get<std::vector<double>>(reg_est[0]);

      std::cout << "Regression test: ";
      compare_two(regression_vector, reg_vec);


      {
        auto [res1, sigma2] = get_derivative_estimates_polynomial<debias_order>(
            X_data, nrow, ncol, Y_data);
        //treeson::utils::print_vector(res1);
        std::cout << "Derivatives test: ";
        compare_two(der_mat, res1);

        std::cout << "Selected lambda:" << select_lifetime_polynomial<debias_order>(X_data, Y_data);

        /*
        auto res2 = get_derivative_estimates_polynomial2(
            X_data, nrow, ncol, Y_data,debias_order);
        compare_two(res1, res2);
        Stopwatch sw;
        double lambda_hat = select_lifetime_polynomial<debias_order>(X_data, Y_data, ncol);
        const auto timing = sw() * 1e-3;
        std::cout << "Nrow: "<< nrow << " ncol: "<< ncol <<
            " passes: " << (timing < 0.02) << " lambda_hat: " <<
            lambda_hat << std::endl;
            */
      }
    //}
  //}
  return 0;
}
