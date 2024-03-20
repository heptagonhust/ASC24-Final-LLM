#pragma once

#ifndef MANDELBROT_H_I0UEF3KR
#define MANDELBROT_H_I0UEF3KR

#include "rpc/msgpack.hpp"
#include <vector>
#include <queue>
#include <filesystem>

struct Sequence
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    int32_t reqId;
    MSGPACK_DEFINE_ARRAY(inputIds, outputLen, reqId)
};
namespace fs = std::filesystem;
using Sequences = std::vector<Sequence>;
using SeqQ = std::queue<Sequence>;

using outputTokenIds = std::vector<int32_t>; 
using ResultQ = std::queue<outputTokenIds>;

std::string LoadBytesFromFile(const fs::path& path);
SeqQ readDatasetFromJson(
    const std::filesystem::path& datasetPath, 
    const std::filesystem::path& tokenizerPath
);
void PrintEncodeResult(const std::vector<int>& ids);
void writeResultsToJson(
    const std::filesystem::path& outputPath,
    const std::filesystem::path& tokenizerPath,
    ResultQ Q
);
#endif
