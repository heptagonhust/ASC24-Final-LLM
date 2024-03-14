#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <iostream>
#include "model_instance/request.h"
#include "tensorrt_llm/tokenizers/tokenizers_cpp.h"
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

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

void PrintEncodeResult(const std::vector<int>& ids) {
  std::cout << "tokens=[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}

Sequences parseDatasetJson(
    std::filesystem::path const& datasetPath, 
    int maxNumSequences
){   
    auto blob = LoadBytesFromFile("tokenizer.json");
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    auto constexpr allowExceptions = true;
    auto constexpr ignoreComments = true;
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(datasetPath), "File does not exist: %s", datasetPath.c_str());
    std::ifstream jsonStream(datasetPath);
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ignoreComments);
    
    Sequences seqs;
    for (auto const& Prompt : json["Prompts"])
    {
        if (seqs.size() >= maxNumSequences)
            break;
        // std::cout<<"Prompt:"<<Prompt["input"]<<"\n";
        std::vector<int> ids = tok->Encode(Prompt["input"]);
        // PrintEncodeResult(ids);
        nlohmann::json seq;
        seq["input_ids"] = ids;
        seq["output_len"] = 20;
        seq["delay"] = 0.0;
        // std::cout<<sample;
        seqs.emplace_back(Sequence{seq["input_ids"], seq["output_len"],seq["delay"]});
    }
    return seqs;
}