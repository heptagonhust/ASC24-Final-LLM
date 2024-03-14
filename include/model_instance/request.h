#pragma once
#include <cstdint>
#include <vector>
#include <filesystem>
#include <map>

#include "tensorrt_llm/executor/executor.h"

namespace fs = std::filesystem;
namespace texec = tensorrt_llm::executor;

struct Sequence
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    float delay;
};

using Sequences = std::vector<Sequence>;

std::string LoadBytesFromFile(const fs::path& path);
Sequences readDatasetFromJson(
    const std::filesystem::path& datasetPath, 
    const std::filesystem::path& tokenizerPath, 
    int maxNumSequences
);
void writeResultsToJson(
    const std::filesystem::path& outputPath,
    const std::filesystem::path& tokenizerPath,
    std::map<texec::IdType, texec::Result> results
);