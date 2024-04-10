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
#include <vector>

std::string format_question(const std::string &question,
                            const std::vector<std::string> &options,
                            std::string answer, bool ex);

SeqQ readDatasetFromCSV(const std::filesystem::path &datasetPath,
                        const std::filesystem::path &tokenizerPath);

SeqQ readDatasetFromCSVfolder(const std::filesystem::path &folderPath,
                              const std::filesystem::path &tokenizerPath);

float output_acc(SeqV V_input, std::vector<std::vector<float>> logits_output,
                 const std::filesystem::path &tokenizerPath);