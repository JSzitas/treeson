#ifndef TREESON_UTILS_H
#define TREESON_UTILS_H

#include <fstream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>

template<typename IntegralType, typename FloatingType>
class CSVLoader {
public:
  using FeatureData = std::variant<
      std::vector<IntegralType>, std::vector<FloatingType>>;

  template<const bool headers,
           const bool row_names>
  static std::vector<FeatureData> load_data(const std::string& filename,
                                            const size_t max_lines = 10000) {
    std::vector<FeatureData> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open CSV file.");
    }
    std::string line;
    std::vector<std::vector<std::string>> columns;
    // skip headers
    if constexpr(headers) std::getline(file, line);
    size_t n_line = 0;
    while (std::getline(file, line)) {
      if(n_line > max_lines) {
        break;
      }
      std::istringstream ss(line);
      std::string cell;
      size_t col_index = 0;
      bool first_cell = true;
      while (std::getline(ss, cell, ',')) {
        if (row_names && first_cell) {
          first_cell = false;
          continue;
        }
        if (col_index >= columns.size()) {
          columns.emplace_back();
        }
        columns[col_index].push_back(cell);
        ++col_index;
      }
      n_line++;
    }
    file.close();
    for (const auto& column : columns) {
      if (is_integral_column(column)) {
        std::vector<IntegralType> column_data = convert_to_integral(column);
        data.emplace_back(column_data);
      } else {
        std::vector<FloatingType> column_data = convert_to_floating(column);
        data.emplace_back(column_data);
      }
    }
    return data;
  }
private:
  static bool is_integral_column(const std::vector<std::string>& column) {
    for (const auto& value : column) {
      if (value.find('.') != std::string::npos) {
        return false;
      }
    }
    return true;
  }
  static std::vector<IntegralType> convert_to_integral(const std::vector<std::string>& column) {
    std::vector<IntegralType> result;
    for (const auto& value : column) {
      result.push_back(static_cast<IntegralType>(std::stoll(value)));
    }
    return result;
  }

  static std::vector<FloatingType> convert_to_floating(const std::vector<std::string>& column) {
    std::vector<FloatingType> result;
    for (const auto& value : column) {
      result.push_back(static_cast<FloatingType>(std::stod(value)));
    }
    return result;
  }
};

template<typename IntegralType, typename FloatingType>
class CSVWriter {
public:
  using FeatureData = std::variant<
      std::vector<IntegralType>, std::vector<FloatingType>>;
  using FloatingData = std::vector<std::vector<FloatingType>>;

  template<const bool headers, const bool row_names>
  static void write_data(const std::string& filename,
                         const std::vector<std::string>& header_names,
                         const std::vector<std::string>& row_names_data,
                         const std::vector<FeatureData>& data)
  {
    std::ofstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open CSV file.");
    }

    if constexpr (headers) {
      if constexpr (row_names) {
        file << "RowNames,";
      }
      for (size_t i = 0; i < header_names.size(); i++) {
        if (i != 0) file << ",";
        file << header_names[i];
      }
      file << "\n";
    }
    size_t numRows = std::visit([](auto&x) -> auto {
      return x.size();
    }, data[0]);
    for (size_t row = 0; row < numRows; row++) {
      if constexpr (row_names) {
        if (!row_names_data.empty()) {
          file << row_names_data[row];
        } else {
          file << row;
        }
        file << ",";
      }
      for (size_t col = 0; col < data.size(); col++) {
        if (col != 0) file << ",";
        std::visit([&file, row](auto&& arg) {
          if (row < arg.size()) {
            file << arg[row];
          }
        }, data[col]);
      }
      file << "\n";
    }
    file.close();
  }
  template<const bool headers, const bool row_names>
  static void write_data(const std::string& filename,
                         const std::vector<std::string>& header_names,
                         const std::vector<std::string>& row_names_data,
                         const FloatingData& data)
  {
    std::ofstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open CSV file.");
    }

    if constexpr (headers) {
      if constexpr (row_names) {
        file << "RowNames,";
      }
      for (size_t i = 0; i < header_names.size(); i++) {
        if (i != 0) file << ",";
        file << header_names[i];
      }
      file << "\n";
    }

    size_t numRows = data.size();

    for (size_t row = 0; row < numRows; row++) {
      if constexpr (row_names) {
        if (!row_names_data.empty()) {
          file << row_names_data[row];
        } else {
          file << row;
        }
        file << ",";
      }

      for (size_t col = 0; col < data[row].size(); col++) {
        if (col != 0) file << ",";
        file << data[row][col];
      }
      file << "\n";
    }
    file.close();
  }
};


template<typename DataType>
std::pair<DataType, DataType> train_test_split(
    const DataType& data, const double train_fraction) {
  if (train_fraction < 0.0 || train_fraction > 1.0) {
    throw std::invalid_argument("train_fraction must be between 0 and 1.");
  }

  DataType train_data;
  DataType test_data;

  std::random_device rd;
  std::mt19937 g(rd());

  size_t n = std::visit([](auto&& arg) { return arg.size(); }, data[0]);
  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), g);
  for (const auto& column : data) {
    size_t train_size = static_cast<size_t>(n * train_fraction);
    std::visit([&](auto&& vec) {
      std::decay_t<decltype(vec)> train_column, test_column;
      for (size_t i = 0; i < train_size; ++i) {
          train_column.push_back(vec[indices[i]]);
      }
      for(size_t i = train_size; i < n; ++i) {
        test_column.push_back(vec[indices[i]]);
      }
      train_data.push_back(train_column);
      test_data.push_back(test_column);
    }, column);
  }
  return {train_data, test_data};
}
template<typename T>
std::string format_number(T num, size_t decimal) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(decimal) << num;
  return oss.str();
}
template<typename DataType>
std::vector<size_t> get_column_widths(
    const DataType& data, const size_t decimal) {
  std::vector<size_t> widths(data.size(), 0);
  for (size_t col = 0; col < data.size(); ++col) {
    widths[col] = std::visit([decimal](auto&& vec) {
      size_t max_width = 0;
      for (const auto& elem : vec) {
        // Compute width based on whether the element is integral or floating
        std::string elem_str = format_number(elem, decimal);
        max_width = std::max(max_width, elem_str.length());
      }
      return max_width;
    }, data[col]);
  }

  return widths;
}

template<typename DataType>
void print_data(const DataType& data, const size_t n = 10,
                const size_t decimal = 2) {
  if (data.empty()) return;
  size_t num_rows = std::visit([](auto&& vec) {
    return vec.size();
  }, data[0]);
  num_rows = std::min(num_rows, n);
  std::vector<size_t> col_widths = get_column_widths(data, decimal);
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t col = 0; col < data.size(); ++col) {
      std::visit([&](auto&& vec) {
        std::cout << std::setw(col_widths[col]) << format_number(vec[row], decimal) << " ";
      }, data[col]);
    }
    std::cout << std::endl;
  }
}

#endif // TREESON_UTILS_H
