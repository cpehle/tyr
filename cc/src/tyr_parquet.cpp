/*
 * tyr_parquet.cpp - Parquet FFI Bindings using Apache Arrow
 *
 * Provides streaming access to parquet files for pretraining data loading.
 * Implements row-group-by-row-group iteration for memory-efficient processing.
 */

#include <lean/lean.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <filesystem>
#include <algorithm>
#include <string>
#include <vector>

namespace fs = std::filesystem;

extern "C" {

// ============================================================================
// Helper Functions
// ============================================================================

// Create a Lean String from C++ string
static lean_object* mk_lean_string(const std::string& s) {
    return lean_mk_string(s.c_str());
}

// Create a Lean Array of Strings from C++ vector
static lean_object* mk_string_array(const std::vector<std::string>& strings) {
    lean_object* arr = lean_alloc_array(strings.size(), strings.size());
    for (size_t i = 0; i < strings.size(); i++) {
        lean_array_set_core(arr, i, mk_lean_string(strings[i]));
    }
    return arr;
}

// Create IO error result
static lean_object* mk_io_error(const std::string& msg) {
    lean_object* err = lean_mk_io_user_error(lean_mk_string(msg.c_str()));
    return lean_io_result_mk_error(err);
}

// Create IO success result
static lean_object* mk_io_ok(lean_object* value) {
    return lean_io_result_mk_ok(value);
}

// ============================================================================
// ParquetMetadata Structure
// Structure: { numRowGroups : Nat, numRows : Nat, columns : Array String }
// ============================================================================

static lean_object* mk_parquet_metadata(size_t num_row_groups, size_t num_rows,
                                        const std::vector<std::string>& columns) {
    // Allocate constructor with 3 object fields
    // In Lean 4, Nat is represented as boxed integers for small values
    lean_object* obj = lean_alloc_ctor(0, 3, 0);
    lean_ctor_set(obj, 0, lean_box(num_row_groups));  // numRowGroups
    lean_ctor_set(obj, 1, lean_box(num_rows));        // numRows
    lean_ctor_set(obj, 2, mk_string_array(columns));  // columns
    return obj;
}

// ============================================================================
// RowGroupData Structure
// Structure: { documents : Array String, numDocs : Nat }
// ============================================================================

static lean_object* mk_row_group_data(const std::vector<std::string>& documents) {
    lean_object* obj = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(obj, 0, mk_string_array(documents));  // documents
    lean_ctor_set(obj, 1, lean_box(documents.size())); // numDocs
    return obj;
}

// ============================================================================
// FFI Implementation: lean_parquet_list_files
// Lists all .parquet files in a directory, sorted by name.
// ============================================================================

lean_object* lean_parquet_list_files(b_lean_obj_arg dir_path, lean_object* /* world */) {
    std::string path = lean_string_cstr(dir_path);

    // Check if directory exists
    if (!fs::exists(path)) {
        return mk_io_error("Directory does not exist: " + path);
    }

    if (!fs::is_directory(path)) {
        return mk_io_error("Path is not a directory: " + path);
    }

    // Collect parquet files
    std::vector<std::string> parquet_files;
    try {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
                parquet_files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        return mk_io_error(std::string("Error listing directory: ") + e.what());
    }

    // Sort for deterministic ordering
    std::sort(parquet_files.begin(), parquet_files.end());

    return mk_io_ok(mk_string_array(parquet_files));
}

// ============================================================================
// FFI Implementation: lean_parquet_get_metadata
// Gets metadata (row group count, row count, column names) for a parquet file.
// ============================================================================

lean_object* lean_parquet_get_metadata(b_lean_obj_arg file_path, lean_object* /* world */) {
    std::string path = lean_string_cstr(file_path);

    // Open file
    auto maybe_infile = arrow::io::ReadableFile::Open(path);
    if (!maybe_infile.ok()) {
        return mk_io_error("Failed to open file: " + maybe_infile.status().message());
    }
    auto infile = *maybe_infile;

    // Create parquet reader
    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto status = parquet::arrow::FileReader::Make(
        arrow::default_memory_pool(),
        parquet::ParquetFileReader::Open(infile),
        &reader);
    if (!status.ok()) {
        return mk_io_error("Failed to open parquet file: " + status.message());
    }

    // Get metadata from the parquet file reader
    auto parquet_reader = reader->parquet_reader();
    auto file_metadata = parquet_reader->metadata();

    size_t num_row_groups = file_metadata->num_row_groups();
    size_t num_rows = file_metadata->num_rows();

    // Get column names from schema
    auto schema = file_metadata->schema();
    std::vector<std::string> columns;
    for (int i = 0; i < schema->num_columns(); i++) {
        columns.push_back(schema->Column(i)->name());
    }

    return mk_io_ok(mk_parquet_metadata(num_row_groups, num_rows, columns));
}

// ============================================================================
// FFI Implementation: lean_parquet_read_row_group
// Reads a specific row group from a parquet file, returning text column data.
// ============================================================================

lean_object* lean_parquet_read_row_group(b_lean_obj_arg file_path, uint64_t row_group_idx,
                                          b_lean_obj_arg text_column, lean_object* /* world */) {
    std::string path = lean_string_cstr(file_path);
    std::string column_name = lean_string_cstr(text_column);

    // Open file
    auto maybe_infile = arrow::io::ReadableFile::Open(path);
    if (!maybe_infile.ok()) {
        return mk_io_error("Failed to open file: " + maybe_infile.status().message());
    }
    auto infile = *maybe_infile;

    // Create parquet reader
    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto status = parquet::arrow::FileReader::Make(
        arrow::default_memory_pool(),
        parquet::ParquetFileReader::Open(infile),
        &reader);
    if (!status.ok()) {
        return mk_io_error("Failed to open parquet file: " + status.message());
    }

    // Check row group index is valid
    auto parquet_reader = reader->parquet_reader();
    auto file_metadata = parquet_reader->metadata();
    if (row_group_idx >= static_cast<uint64_t>(file_metadata->num_row_groups())) {
        return mk_io_error("Row group index out of range: " + std::to_string(row_group_idx) +
                          " >= " + std::to_string(file_metadata->num_row_groups()));
    }

    // Find column index by name
    int column_idx = -1;
    auto schema = file_metadata->schema();
    for (int i = 0; i < schema->num_columns(); i++) {
        if (schema->Column(i)->name() == column_name) {
            column_idx = i;
            break;
        }
    }

    if (column_idx < 0) {
        return mk_io_error("Column not found: " + column_name);
    }

    // Read the specific row group
    std::shared_ptr<arrow::Table> table;
    status = reader->ReadRowGroup(static_cast<int>(row_group_idx), {column_idx}, &table);
    if (!status.ok()) {
        return mk_io_error("Failed to read row group: " + status.message());
    }

    // Extract text column data
    auto column = table->column(0);
    std::vector<std::string> documents;

    for (int chunk_idx = 0; chunk_idx < column->num_chunks(); chunk_idx++) {
        auto chunk = column->chunk(chunk_idx);

        // Handle different string array types
        if (auto string_array = std::dynamic_pointer_cast<arrow::StringArray>(chunk)) {
            for (int64_t i = 0; i < string_array->length(); i++) {
                if (string_array->IsNull(i)) {
                    documents.push_back("");  // Handle nulls as empty strings
                } else {
                    documents.push_back(string_array->GetString(i));
                }
            }
        } else if (auto large_string_array = std::dynamic_pointer_cast<arrow::LargeStringArray>(chunk)) {
            for (int64_t i = 0; i < large_string_array->length(); i++) {
                if (large_string_array->IsNull(i)) {
                    documents.push_back("");
                } else {
                    documents.push_back(std::string(large_string_array->GetView(i)));
                }
            }
        } else {
            return mk_io_error("Column is not a string type");
        }
    }

    return mk_io_ok(mk_row_group_data(documents));
}

// ============================================================================
// FFI Implementation: lean_parquet_file_exists
// Checks if a file exists.
// ============================================================================

lean_object* lean_parquet_file_exists(b_lean_obj_arg file_path, lean_object* /* world */) {
    std::string path = lean_string_cstr(file_path);
    bool exists = fs::exists(path) && fs::is_regular_file(path);
    return mk_io_ok(lean_box(exists ? 1 : 0));
}

} // extern "C"
