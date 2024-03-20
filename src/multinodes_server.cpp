#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <nlohmann/json.hpp>

#include "multinodes/multinodes_server.h"

#include "tokenizers/tokenizers_cpp.h"

#include "rpc/server.h"
#include "rpc/msgpack.hpp"

namespace fs = std::filesystem;

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
void PrintEncodeResult(const std::vector<int>& ids) {
    std::cout << "tokens=[";
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i != 0) std::cout << ", ";
      std::cout << ids[i];
    }
    std::cout << "]" << std::endl;
}
SeqQ readDatasetFromJson(
    const std::filesystem::path& datasetPath, 
    const std::filesystem::path& tokenizerPath){   
    auto blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    auto constexpr allowExceptions = true;
    auto constexpr ignoreComments = true;
    std::ifstream jsonStream(datasetPath);
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ignoreComments);
    
    SeqQ seqs;
    for (auto const& Prompt : json["Prompts"])
    {
        // std::cout<<"Prompt:"<<Prompt["input"]<<"\n";
        std::vector<int> ids = tok->Encode(Prompt["input"]);
        // PrintEncodeResult(ids);
        nlohmann::json seq;
        seq["input_ids"] = ids;
        seq["output_len"] = 200;        
        // std::cout<<sample;
        //seqs.emplace_back(Sequence{seq["input_ids"], seq["output_len"],seq["delay"]});
        seqs.push(Sequence{seq["input_ids"], seq["output_len"]});
    }
    return seqs;
}
void writeResultsToJson(
    const std::filesystem::path& outputPath,
    const std::filesystem::path& tokenizerPath,
    ResultQ Q
){
    auto blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    
    nlohmann::json j;
    int size = Q.size();
    for (int i = 0;i < size; i++)
    {
        outputTokenIds output = Q.front();
        Q.pop();
        std::string decoded_prompt = tok->Decode(output);
        j.push_back("reqId: " + std::to_string(i) + ", decode: \"" + decoded_prompt + "\"");
    }
    std::ofstream outputFile(
        outputPath, 
        std::ios::out
    );
    outputFile << j << std::endl;
}

int main(int argc, char* argv[]) {
    
    cxxopts::Options options(
        "TensorRT-LLM multinodes Server");
    options.add_options()("dataset", "Dataset that is used for benchmarking BatchManager.",
        cxxopts::value<std::string>()->default_value("data.json"));
    options.add_options()("tokenizer", "Tokenizer of the model.", 
        cxxopts::value<std::string>()->default_value("tokenizer.json"));
    options.add_options()("output", "Output file for the results.", 
        cxxopts::value<std::string>()->default_value("output.json"));
    auto result = options.parse(argc, argv);
    std::string datasetPath = result["dataset"].as<std::string>();
    std::string tokenizerPath = result["tokenizer"].as<std::string>();
    std::string outputPath = result["output"].as<std::string>();
    SeqQ Queue_input = readDatasetFromJson(datasetPath,tokenizerPath);
    ResultQ Queue_output;
    int length = Queue_input.size();
    rpc::server srv("192.168.250.100",10000);
    int batch_size = 200;
    srv.bind("getseqs",[&](){
        Sequences seqs;
        if(Queue_input.size() > 0){
            for (int i = 0; i < batch_size; ++i) {
                if(Queue_input.size() == 0)
                    break;
                else{
                    Sequence seq = Queue_input.front();
                    Queue_input.pop();
                    seqs.emplace_back(Sequence{seq.inputIds, seq.outputLen});
                }
            }
        }
        return seqs;
    });

    srv.bind("outseqs_back",[&](outputTokenIds outIdqueue){
        Queue_output.push(outIdqueue);
        if (Queue_output.size() == length) {
            writeResultsToJson(outputPath, tokenizerPath, Queue_output);
            srv.stop();
        }
    });
    srv.run();
    //writeResultsToJson(outputPath,tokenizerPath,Queue_output);
    // std::cout << "Press [ENTER] to exit the server." << std::endl;
    // std::cin.ignore();
    return 0;
}