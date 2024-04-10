#include "sequences/sequences.h"
#include "tokenizers/tokenizers_cpp.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <regex>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <variant>
#include <vector>

class MMLU_Base {
public:
  virtual std::string format_question(const std::string &question,
                                      const std::vector<std::string> &options,
                                      std::string answer, bool ex) = 0;

  virtual SeqQ
  readDatasetFromCSV(const std::filesystem::path &datasetPath,
                     const std::filesystem::path &tokenizerPath) = 0;

  virtual SeqQ
  readDatasetFromCSVfolder(const std::filesystem::path &folderPath,
                           const std::filesystem::path &tokenizerPath) = 0;

  virtual float output_acc(
      SeqV V_input,
      std::variant<ResultV, std::vector<std::vector<float>>> Vector_output,
      const std::filesystem::path &tokenizerPath) = 0;
};