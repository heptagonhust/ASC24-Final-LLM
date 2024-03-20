#include <filesystem>
#include <chrono>
#include <thread>

#include "model_instance/executor.h"

using namespace tensorrt_llm::runtime;

ExecutorServer::ExecutorServer(
    fs::path engineDir,
    texec::ExecutorConfig const& executorConfig,
    std::shared_ptr<Recorder> recorder,
    std::chrono::milliseconds waitSleep)
    : recorder_(std::move(recorder))
    , waitSleep_(waitSleep)
    , activeCount_(0)
{
    executor_ = std::make_shared<texec::Executor>(engineDir, texec::ModelType::kDECODER_ONLY, executorConfig);
}


void ExecutorServer::enqueue(std::vector<texec::Request> requests, bool warmup)
{
    try
    {
        std::vector<SizeType> inputLengths, maxNewTokens;
        for (auto const& request : requests)
        {
            inputLengths.push_back(request.getInputTokenIds().size());
            maxNewTokens.push_back(request.getMaxNewTokens());
        }
        auto const start = std::chrono::steady_clock::now();
        auto reqIds = executor_->enqueueRequests(std::move(requests));
        for (int req = 0; req < reqIds.size(); ++req)
        {
            if (!warmup)
            {
                recorder_->recordStart(inputLengths.at(req), maxNewTokens.at(req), reqIds.at(req), start);
            }
            activeCount_++;
        }
    }
    catch (const std::exception& e)
    {
        TLLM_THROW("%s", e.what());
    }
    return;
}

void ExecutorServer::waitForGetReqs(SizeType threshold)
{
    SizeType numReadyResponse = executor_->getNumResponsesReady();
    while (numReadyResponse < activeCount_ - threshold) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        numReadyResponse = executor_->getNumResponsesReady();
    }
    return;
}

void ExecutorServer::waitForResponses(std::optional<SizeType> numRequests, bool warmup)
{
    SizeType numFinished = 0;
    while (activeCount_ || (numRequests && numFinished < numRequests.value()))
    {
        auto responses = executor_->awaitResponses(std::nullopt, waitSleep_);
        for (auto const& response : responses)
        {
            if (!response.hasError())
            {
                texec::Result result = response.getResult();
                if (result.isFinal)
                {
                    auto reqId = response.getRequestId();
                    activeCount_--;
                    numFinished++;
                    if (!warmup)
                    {
                        recorder_->recordEnd(reqId);
                        results_.emplace(reqId, std::move(result));
                    }
                }
            }
            else 
            {
                // This request failed for some reason, get error msg
                std::string errStr = "Request id " + std::to_string(response.getRequestId()) + " failed with err "
                    + response.getErrorMsg();
                TLLM_THROW(errStr);
            }

        }
    }
}
