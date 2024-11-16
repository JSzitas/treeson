#ifndef TREESON_PARQUET_H
#define TREESON_PARQUET_H
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/file_reader.h>
#include <variant>
#include <vector>
#include <iostream>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/file_reader.h>
#include <variant>
#include <vector>
#include <iostream>
#include <string>
#include <type_traits>


template<typename integral_t, typename scalar_t>
std::vector<std::variant<std::vector<integral_t>, std::vector<scalar_t>>> load_parquet_data(const std::string& file_path) {
  static_assert(std::is_integral<integral_t>::value, "integral_t must be an integral type");
  static_assert(std::is_floating_point<scalar_t>::value, "scalar_t must be a floating point type");

  using FeatureData = std::variant<std::vector<integral_t>, std::vector<scalar_t>>;

  // Open the Parquet file
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(
      infile,
      arrow::io::ReadableFile::Open(file_path, arrow::default_memory_pool()));

  // Read the file as a Parquet file
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &parquet_reader));

  // Create Arrow Table from Parquet file
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(parquet_reader->ReadTable(&table));

  std::vector<FeatureData> data;

  // Process each column in the table
  for (int i = 0; i < table->num_columns(); ++i) {
    auto column = table->column(i);
    auto type = column->type();

    if (type->id() == arrow::Type::INT32) {
      std::vector<integral_t> column_data;
      for (int64_t j = 0; j < column->length(); ++j) {
        std::shared_ptr<arrow::Int32Array> int_array =
            std::static_pointer_cast<arrow::Int32Array>(column->chunk(0));
        column_data.push_back(static_cast<integral_t>(int_array->Value(j)));
      }
      data.emplace_back(column_data);
    } else if (type->id() == arrow::Type::UINT32) {
      std::vector<integral_t> column_data;
      for (int64_t j = 0; j < column->length(); ++j) {
        std::shared_ptr<arrow::UInt32Array> uint_array =
            std::static_pointer_cast<arrow::UInt32Array>(column->chunk(0));
        column_data.push_back(static_cast<integral_t>(uint_array->Value(j)));
      }
      data.emplace_back(column_data);
    } else if (type->id() == arrow::Type::INT64) {
      std::vector<integral_t> column_data;
      for (int64_t j = 0; j < column->length(); ++j) {
        std::shared_ptr<arrow::Int64Array> int_array =
            std::static_pointer_cast<arrow::Int64Array>(column->chunk(0));
        column_data.push_back(static_cast<integral_t>(int_array->Value(j)));
      }
      data.emplace_back(column_data);
    } else if (type->id() == arrow::Type::UINT64) {
      std::vector<integral_t> column_data;
      for (int64_t j = 0; j < column->length(); ++j) {
        std::shared_ptr<arrow::UInt64Array> uint_array =
            std::static_pointer_cast<arrow::UInt64Array>(column->chunk(0));
        column_data.push_back(static_cast<integral_t>(uint_array->Value(j)));
      }
      data.emplace_back(column_data);
    } else if (type->id() == arrow::Type::DOUBLE) {
      std::vector<scalar_t> column_data;
      for (int64_t j = 0; j < column->length(); ++j) {
        std::shared_ptr<arrow::DoubleArray> double_array =
            std::static_pointer_cast<arrow::DoubleArray>(column->chunk(0));
        column_data.push_back(static_cast<scalar_t>(double_array->Value(j)));
      }
      data.emplace_back(column_data);
    } else if (type->id() == arrow::Type::FLOAT) {
      std::vector<scalar_t> column_data;
      for (int64_t j = 0; j < column->length(); ++j) {
        std::shared_ptr<arrow::FloatArray> float_array =
            std::static_pointer_cast<arrow::FloatArray>(column->chunk(0));
        column_data.push_back(static_cast<scalar_t>(float_array->Value(j)));
      }
      data.emplace_back(column_data);
    } else {
      std::cerr << "Unsupported column type: " << type->ToString() << std::endl;
    }
  }

  return data;
}

#endif // TREESON_PARQUET_H
