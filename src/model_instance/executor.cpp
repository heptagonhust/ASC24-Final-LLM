#include "model_instance/executor.h"
#include <filesystem>

using namespace tensorrt_llm::runtime;

ExecutorServer::ExecutorServer(
    fs::path engineDir,
    texec::ExecutorConfig const& executorConfig,
    std::shared_ptr<Recorder> recorder,
    std::chrono::milliseconds waitSleep)
    : mRecorder(std::move(recorder))
    , mWaitSleep(waitSleep)
    , mActiveCount(0)
{
    mExecutor = std::make_shared<texec::Executor>(engineDir, texec::ModelType::kDECODER_ONLY, executorConfig);
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
        auto reqIds = mExecutor->enqueueRequests(std::move(requests));
        for (int req = 0; req < reqIds.size(); ++req)
        {
            if (!warmup)
            {
                mRecorder->recordStart(inputLengths.at(req), maxNewTokens.at(req), reqIds.at(req), start);
            }
            mActiveCount++;
        }
    }
    catch (const std::exception& e)
    {
        TLLM_THROW("%s", e.what());
    }
    return;
}

void ExecutorServer::waitForResponses(std::optional<SizeType> numRequests, bool warmup)
{
    SizeType numFinished = 0;
    while (mActiveCount || (numRequests && numFinished < numRequests.value()))
    {
        auto responses = mExecutor->awaitResponses(std::nullopt, mWaitSleep);
        for (auto const& response : responses)
        {
            if (response.hasError())
            {
                // This request failed for some reason, get error msg
                std::string errStr = "Request id " + std::to_string(response.getRequestId()) + " failed with err "
                    + response.getErrorMsg();
                TLLM_THROW(errStr);
            }
            else if (response.getResult().isFinal)
            {
                auto reqId = response.getRequestId();
                mActiveCount--;
                numFinished++;
                if (!warmup)
                {
                    mRecorder->recordEnd(reqId);
                }
            }
        }
    }
}
