#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <iostream>
#include <vector>
#include <variant>

using DataType = std::vector<std::variant<std::vector<size_t>, std::vector<double>>>;

DataType load_parquet(const std::string& filename) {
  // Initialize Arrow memory pool
  auto pool = arrow::default_memory_pool();

  // Open Parquet file reader
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_THROW_NOT_OK(arrow::io::ReadableFile::Open(filename, pool, &infile));
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &parquet_reader));

  // Read entire file as a table
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(parquet_reader->ReadTable(&table));

  DataType result;

  // Process each column in the table
  for (int i = 0; i < table->num_columns(); ++i) {
    std::shared_ptr<arrow::ChunkedArray> column = table->column(i);

    if (column->type()->id() == arrow::Type::DOUBLE) {
      // Handle double column
      std::vector<double> values;

      for (const auto& chunk : column->chunks()) {
        auto double_array = std::static_pointer_cast<arrow::DoubleArray>(chunk);
        for (int64_t j = 0; j < double_array->length(); ++j) {
          if (!double_array->IsNull(j)) {
            values.push_back(double_array->Value(j));
          } else {
            values.push_back(std::numeric_limits<double>::quiet_NaN());
          }
        }
      }

      result.push_back(values);

    } else if (column->type()->id() == arrow::Type::INT64) {
      // Handle integer column
      std::vector<size_t> values;

      for (const auto& chunk : column->chunks()) {
        auto int64_array = std::static_pointer_cast<arrow::Int64Array>(chunk);
        for (int64_t j = 0; j < int64_array->length(); ++j) {
          if (!int64_array->IsNull(j)) {
            values.push_back(static_cast<size_t>(int64_array->Value(j)));
          } else {
            values.push_back(static_cast<size_t>(-1)); // Placeholder for NaN
          }
        }
      }

      result.push_back(values);
    } else {
      throw std::runtime_error("Unsupported column type in Parquet file");
    }
  }

  return result;
}

int main() {
  const std::string filename = "example.parquet";
  try {
    DataType data = load_parquet(filename);

  } catch (const std::exception& ex) {
    std::cerr << "Error loading Parquet file: " << ex.what() << std::endl;
  }

  return 0;
}