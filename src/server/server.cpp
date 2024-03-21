#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>

#include "rpc/server.h"
#include "rpc/this_server.h"
#include "rpc/msgpack.hpp"
#include "tokenizers/tokenizers_cpp.h"

#include "sequences/sequences.h"


namespace fs = std::filesystem;
using SeqQ = std::queue<Sequence>;
using outputTokenIds = std::vector<int32_t>; 
using ResultQ = std::queue<outputTokenIds>;

namespace 
{

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
    const std::filesystem::path& tokenizerPath
){   
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


class Recorder
{
public:
    Recorder() = default;

    void initialize() {
        mStart = std::chrono::steady_clock::now();
    }

    void finalize() {
        mEnd = std::chrono::steady_clock::now();
    }

    void record(outputTokenIds& tokenIds) {
        mNumSeqs += 1;
        mNumTokens += tokenIds.size();
    }

    void calculateMetrics() {
        mTotalLatency = std::chrono::duration<float, std::milli>(mEnd - mStart).count();
        mSeqThroughput = mNumSeqs / (mTotalLatency / 1000);
        mTokenThroughput = mNumTokens / (mTotalLatency / 1000);
    }

    void report() {
        printf("[SERVER BENCHMARK] num_seqs %d\n", mNumSeqs);
        printf("[SERVER BENCHMARK] total_latency(ms) %.2f\n", mTotalLatency);
        printf("[SERVER BENCHMARK] seq_throughput(seq/sec) %.2f\n", mSeqThroughput);
        printf("[SERVER BENCHMARK] token_throughput(token/sec) %.2f\n", mTokenThroughput);
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> mStart;
    std::chrono::time_point<std::chrono::steady_clock> mEnd;
    int mNumSeqs{};
    int mNumTokens{};
    float mTotalLatency{};
    float mSeqThroughput{};
    float mTokenThroughput{};
}; // class Recorder

} // namespace


int main(int argc, char* argv[]) {
    
    cxxopts::Options options(
        "TensorRT-LLM multinodes Server");
    options.add_options()("dataset", "Dataset that is used for benchmarking BatchManager.",
        cxxopts::value<std::string>()->default_value("data.json"));
    options.add_options()("tokenizer", "Tokenizer of the model.", 
        cxxopts::value<std::string>()->default_value("tokenizer.json"));
    options.add_options()("output", "Output file for the results.", 
        cxxopts::value<std::string>()->default_value("output.json"));
    options.add_options()("server_addr", "Specify the address of the server.", 
        cxxopts::value<std::string>());
    options.add_options()("server_port", "Specify the port of the server.",
        cxxopts::value<int>());
    options.add_options()("batch_size", "Specify the transmitted batch size of seqs in a single rpc request.",
        cxxopts::value<int>()->default_value("200"));
    auto result = options.parse(argc, argv);
    std::string datasetPath = result["dataset"].as<std::string>();
    std::string tokenizerPath = result["tokenizer"].as<std::string>();
    std::string outputPath = result["output"].as<std::string>();
    std::string addr = result["server_addr"].as<std::string>();
    int port = result["server_port"].as<int>();
    int batch_size = result["batch_size"].as<int>();
    SeqQ Queue_input = readDatasetFromJson(datasetPath,tokenizerPath);
    ResultQ Queue_output;
    int num_seqs = Queue_input.size();

    Recorder recorder;
    rpc::server srv(addr, port);
    srv.bind("getseqs",[&](){
        if (Queue_input.size() == num_seqs) {
            recorder.initialize();
        }
        Sequences seqs;
        std::cout << "Queue_input.size():" << Queue_input.size() << "\n";
        std::flush(std::cout);
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

    srv.bind("outseqs_back",[&](outputTokenIds tokenIds){
        recorder.record(tokenIds);
        Queue_output.push(tokenIds);
        if (Queue_output.size() == num_seqs) {
            recorder.finalize();
            recorder.calculateMetrics();
            recorder.report();

            writeResultsToJson(outputPath, tokenizerPath, Queue_output);
            rpc::this_server().stop();
        }
    });

    srv.run();
    return 0;
}