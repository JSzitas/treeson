//
// Created by jsco on 9/9/24.
//

#ifndef TREESON_UTILS_H
#define TREESON_UTILS_H
#include <fstream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>
#include <stdexcept>
#include <iostream>

template<typename IntegralType, typename FloatingType>
class CSVLoader {
public:
  using FeatureData = std::variant<
      std::vector<IntegralType>, std::vector<FloatingType>>;

  static std::vector<FeatureData> load_data(const std::string& filename) {
    std::vector<FeatureData> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open CSV file.");
    }

    std::string line;
    std::vector<std::vector<std::string>> columns;

    // Read header (first line) and initialize columns
    if (std::getline(file, line)) {
      std::istringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ',')) {
        columns.emplace_back();
      }
    }

    // Read each row
    while (std::getline(file, line)) {
      std::istringstream ss(line);
      std::string cell;
      size_t col_index = 0;
      while (std::getline(ss, cell, ',')) {
        if (col_index >= columns.size()) {
          throw std::runtime_error("CSV row has more columns than header.");
        }
        columns[col_index].push_back(cell);
        ++col_index;
      }
    }

    // Close the file
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


// Function to print the loaded data
template<typename IntegralType, typename FloatingType>
void print_data(const std::vector<std::variant<std::vector<IntegralType>, std::vector<FloatingType>>>& data) {
  for (const auto& column : data) {
    if (std::holds_alternative<std::vector<IntegralType>>(column)) {
      const auto& int_column = std::get<std::vector<IntegralType>>(column);
      std::cout << "Integral Column:" << std::endl;
      for (const auto& value : int_column) {
        std::cout << value << " ";
      }
      std::cout << std::endl;
    } else {
      const auto& double_column = std::get<std::vector<FloatingType>>(column);
      std::cout << "Floating Column:" << std::endl;
      for (const auto& value : double_column) {
        std::cout << value << " ";
      }
      std::cout << std::endl;
    }
  }
}

#endif // TREESON_UTILS_H
