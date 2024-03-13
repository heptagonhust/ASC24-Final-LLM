#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "model_instance/request.h"

texec::Request makeExecutorRequest(
    Sequence const& seq, 
    SizeType const& beamWidth,
    std::optional<SizeType> const& eosId, 
    std::optional<SizeType> const& padId, 
    bool streaming,
    bool const& returnContextLogits, 
    bool const& returnGenerationLogits
){
    auto samplingConfig = texec::SamplingConfig{beamWidth};
    auto outputConfig = texec::OutputConfig{false, returnContextLogits, returnGenerationLogits, false};
    return texec::Request(seq.inputIds, seq.outputLen, streaming, samplingConfig, outputConfig, eosId, padId);
}

Sequences parseDatasetJson(
    std::filesystem::path const& datasetPath, 
    int maxNumSequences
){
    auto constexpr allowExceptions = true;
    auto constexpr ignoreComments = true;
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(datasetPath), "File does not exist: %s", datasetPath.c_str());
    std::ifstream jsonStream(datasetPath);
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ignoreComments);

    Sequences seqs;

    for (auto const& seq : json["samples"])
    {
        if (seqs.size() >= maxNumSequences)
            break;
        seqs.emplace_back(Sequence{seq["input_ids"], seq["output_len"], seq["delay"]});
    }
    return seqs;
}