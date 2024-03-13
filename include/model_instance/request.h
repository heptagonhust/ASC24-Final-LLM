#pragma once
#include <cstdint>
#include <vector>

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/executor/executor.h"

using namespace tensorrt_llm::runtime;

namespace texec = tensorrt_llm::executor;

struct Sequence
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    float delay;
};

using Sequences = std::vector<Sequence>;

texec::Request makeExecutorRequest(
    Sequence const& seq,
    SizeType const& beamWidth,
    std::optional<SizeType> const& eosId, 
    std::optional<SizeType> const& padId, 
    bool streaming = false,
    bool const& returnContextLogits = false, 
    bool const& returnGenerationLogits = false
);

Sequences parseDatasetJson(std::filesystem::path const& datasetPath, int maxNumSequences);