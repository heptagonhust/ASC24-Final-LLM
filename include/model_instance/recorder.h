#pragma once
#include <string>
#include <chrono>
#include <unordered_map>
#include <memory>

#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/runtime/common.h"


using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;

struct BenchInfo
{
    BenchInfo() = default;

    BenchInfo(int _inputLength, int _outputLength, std::chrono::time_point<std::chrono::steady_clock> _start)
        : inputLength(_inputLength)
        , outputLength(_outputLength)
        , start(_start)
        , latency() {}

    int inputLength;
    int outputLength;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    float latency; // millisecond
};




class Recorder
{
public:
    explicit Recorder(std::string opCsvFile)
        : mOpCsvFile(std::move(opCsvFile)) {}
    
    class Config {
        std::string opCsvFile;
        
    };

    void initialize() {
        mStart = std::chrono::steady_clock::now();
    }

    void finalize() {
        mEnd = std::chrono::steady_clock::now();
    }

    void recordStart(
        std::shared_ptr<InferenceRequest> request, 
        uint64_t requestId
    );

    void recordStart(
        SizeType inputLength, 
        SizeType maxNewTokens, 
        uint64_t requestId,
        std::chrono::time_point<std::chrono::steady_clock> const& start
    );

    void recordEnd(uint64_t requestId);

    void calculateMetrics();
    void report();
    void writeOpMetricsToCsv();

private:
    std::unordered_map<uint64_t, BenchInfo> mRequestBenchInfos;

    std::chrono::time_point<std::chrono::steady_clock> mStart;
    std::chrono::time_point<std::chrono::steady_clock> mEnd;
    int mNumSamples{};
    float mTotalLatency{};
    float mSeqThroughput{};
    float mAvgSeqLatency{};
    float mTokenThroughput{};
    std::string mOpCsvFile;
}; // class Recorder