#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <regex>
#include <cassert>
#include <random>
#include <stdlib.h>
#include <time.h> 
#include "tokenizers/tokenizers_cpp.h"
#include "sequences/sequences.h"

std::string format_question(const std::string& question, const std::vector<std::string>& options, std::string answer, bool ex);

SeqQ readDatasetFromCSV(const std::filesystem::path& datasetPath, const std::filesystem::path& tokenizerPath);

SeqQ readDatasetFromCSVfolder(const std::filesystem::path& folderPath, const std::filesystem::path& tokenizerPath);

float output_acc(SeqV V_input,ResultV Vector_output,const std::filesystem::path& tokenizerPath);