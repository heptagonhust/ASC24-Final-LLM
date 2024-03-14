#include <fstream>
#include <iostream>

#include "model_instance/recorder.h"

void Recorder::recordStart(std::shared_ptr<InferenceRequest> request, uint64_t requestId)
{
    auto const inputLength = request->getInputIds()->getSize();
    auto const maxNewTokens = request->getMaxNewTokensNamed();
    auto const& outputLengthTensor = maxNewTokens.tensor;
    TLLM_CHECK_WITH_INFO(outputLengthTensor != nullptr && outputLengthTensor->getSize() > 0,
        "Undefined scalar vector for %s", maxNewTokens.name.c_str());
    auto const outputLength = *bufferCast<SizeType>(*outputLengthTensor);
    auto const start = std::chrono::steady_clock::now();
    mRequestBenchInfos[requestId] = BenchInfo(inputLength, outputLength, start);
}

void Recorder::recordStart(SizeType inputLength, SizeType maxNewTokens, uint64_t requestId,
    std::chrono::time_point<std::chrono::steady_clock> const& start)
{
    mRequestBenchInfos[requestId] = BenchInfo(inputLength, maxNewTokens, start);
}

void Recorder::recordEnd(uint64_t requestId)
{
    mRequestBenchInfos[requestId].end = std::chrono::steady_clock::now();
    mRequestBenchInfos[requestId].latency = std::chrono::duration<float, std::milli>(
        mRequestBenchInfos[requestId].end - mRequestBenchInfos[requestId].start)
                                                .count();
}

void Recorder::calculateMetrics() {
    mNumSamples = mRequestBenchInfos.size();
    mTotalLatency = std::chrono::duration<float, std::milli>(mEnd - mStart).count();
    mSeqThroughput = mNumSamples / (mTotalLatency / 1000);
    mAvgSeqLatency = 0;
    int totalOutputTokens = 0;
    for (auto reqInfo : mRequestBenchInfos)
    {
        mAvgSeqLatency += reqInfo.second.latency;
        totalOutputTokens += reqInfo.second.outputLength;
    }
    mAvgSeqLatency /= mNumSamples;
    mTokenThroughput = totalOutputTokens / (mTotalLatency / 1000);
}

void Recorder::report() {
    printf("[BENCHMARK] num_samples %d\n", mNumSamples);
    printf("[BENCHMARK] total_latency(ms) %.2f\n", mTotalLatency);
    printf("[BENCHMARK] seq_throughput(seq/sec) %.2f\n", mSeqThroughput);
    printf("[BENCHMARK] avg_sequence_latency(ms) %.2f\n", mAvgSeqLatency);
    printf("[BENCHMARK] token_throughput(token/sec) %.2f\n", mTokenThroughput);
}

void Recorder::writeOpMetricsToCsv() {
    if (!mOpCsvFile.empty())
    {
        std::vector<std::string> headers = {"num_samples", "total_latency(ms)", "seq_throughput(seq/sec)",
            "avg_sequence_latency(ms)", "token_throughput(token/sec)"};

        std::ofstream outputFile(mOpCsvFile);

        if (outputFile.is_open())
        {
            for (const auto& header : headers)
            {
                outputFile << header << ",";
            }
            outputFile << "\n";
            outputFile << mNumSamples << "," << mTotalLatency << "," << mSeqThroughput << "," << mAvgSeqLatency
                       << "," << mTokenThroughput;
            outputFile << "\n";
        }
        else
        {
            std::cerr << "Error opening file '" << mOpCsvFile << "' for writing.\n";
        }
    }
}