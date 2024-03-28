#include <cstdint>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"

#include "model_instance/request.h"
#include "tokenizers/tokenizers_cpp.h"

namespace fs = std::filesystem;
namespace texec = tensorrt_llm::executor;

using namespace std::literals;

namespace {

void PrintEncodeResult(const std::vector<int>& ids) {
    std::cout << "tokens=[";
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i != 0) std::cout << ", ";
      std::cout << ids[i];
    }
    std::cout << "]" << std::endl;
}

} // namespace

std::string LoadBytesFromFile(const fs::path& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
      std::cerr << "Cannot open tokenzier: " << path.string() << std::endl;
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

Sequences readDatasetFromJson(
    const std::filesystem::path& datasetPath, 
    const std::filesystem::path& tokenizerPath, 
    int maxNumSequences
){   
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(tokenizerPath), "Tokenizer does not exist: %s", tokenizerPath.c_str());
    auto blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    auto constexpr allowExceptions = true;
    auto constexpr ignoreComments = true;
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(datasetPath), "Dataset does not exist: %s", datasetPath.c_str());
    std::ifstream jsonStream(datasetPath);
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ignoreComments);
    
    Sequences seqs;
    for (auto const& Prompt : json["Prompts"])
    {
        if (seqs.size() >= maxNumSequences)
            break;
        // std::cout<<"Prompt:"<<Prompt["input"]<<"\n";
        std::vector<int> ids = tok->Encode(Prompt);
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

void writeResultsToJson(
    const std::filesystem::path& outputPath,
    const std::filesystem::path& tokenizerPath,
    std::map<texec::IdType, texec::Result> results
){
    auto blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    
    nlohmann::json j;
    for (auto& [reqId, result] : results)
    {
        std::string decoded_prompt = tok->Decode(result.outputTokenIds[0]);
        // std::cout << "reqId: " << reqId << ", decode: \"" 
        //           << decoded_prompt << "\"" << std::endl;
        j.push_back("reqId: "s + std::to_string(reqId) + ", decode: \"" + decoded_prompt + "\""s);
        
        // if (result.generationLogits.has_value()) {
        //     // auto& tensorPtr = result.generationLogits.value();
        //     // std::string tensorString = tensorString(*tensorPtr);
        //     auto Logits = (*result.generationLogits);
        //     // std::string jsonValue = (std::string)Logits;
        //     // j.push_back(jsonValue);
        //     //std::cout <<(*result.generationLogits)->getDataType()<<"\n";

        //     auto data = static_cast<const float*>(Logits->getData());
        //     auto shape = Logits->getShape();
        //     // std::cout<<shape[0]<<std::endl;
        //     // std::cout<<shape[1]<<std::endl;
        //     // std::cout<<shape[2]<<std::endl;
        //     if (shape.size() <= 2)
        //     {
        //         std::cout << "Tensor doesn't have data in dimension 2." << std::endl;
        //     }
        //     else{
        //         size_t startIndex = 0;
        //         size_t endIndex = shape[2]; 
        //         std::cout<<shape[2];
        //         std::string logits;
        //         for (size_t i = startIndex; i < endIndex; ++i)
        //         {
        //             float value = data[i];
        //             if(i < endIndex-1)
        //                 logits = logits + std::to_string(value) + ",";
        //             else
        //                 logits = logits + std::to_string(value);
        //         }
        //         j.push_back(logits);
        //     }
        // }
    }
    std::ofstream outputFile(
        outputPath, 
        std::ios::out
    );
    outputFile << j << std::endl;
}