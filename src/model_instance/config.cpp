#include <filesystem>
#include <memory>
#include <optional>
#include <algorithm>

#include "tensorrt_llm/runtime/worldConfig.h"
#include "model_instance/config.h"


std::shared_ptr<InstanceConfig> InstanceConfig::from_params(const InstanceParams &instanceParams) {
    auto world_config = WorldConfig::mpi();
    if (world_config.getSize() > 1)
    {
        TLLM_THROW("benchmarkExecutor does not yet support mpiSize > 1");
    }
    auto dataset_path = instanceParams.reqParams.datasetPath;

    texec::SchedulerConfig schedulerConfig {
        batch_scheduler::batchManagerToExecSchedPolicy(
            instanceParams.scheduleParams.schedulerPolicy)
    };
    texec::KvCacheConfig kvCacheConfig {
        instanceParams.cacheParams.enableBlockReuse,
        instanceParams.cacheParams.maxTokensInPagedKvCache,
        std::nullopt,
        std::nullopt,
        instanceParams.cacheParams.freeGpuMemoryFraction,
        false
    };
    texec::ExecutorConfig executorConfig {
        instanceParams.sampleParms.maxBeamWidth,
        schedulerConfig,
        kvCacheConfig,
        instanceParams.cacheParams.enableChunkedContext,
        true,
        instanceParams.engineParams.enableTrtOverlap
    };
    texec::SamplingConfig samplingConfig {
        instanceParams.sampleParms.maxBeamWidth
    };
    texec::OutputConfig outputConfig {
        false,
        instanceParams.outputParams.returnContextLogits,
        instanceParams.outputParams.returnGenerationLogits,
        false
    };

    return std::make_shared<InstanceConfig>(
        instanceParams.engineParams.engine_dir,
        dataset_path,
        world_config,
        executorConfig,
        samplingConfig,
        outputConfig,
        instanceParams
    );
}

std::shared_ptr<ExecutorServer> InstanceConfig::getExecutorServer(std::shared_ptr<Recorder> recorder) const {
    return std::make_shared<ExecutorServer>(
        engineDir_,
        executorConfig_,
        recorder,
        instanceParams_.serverParams.waitSleep
    );
}

std::shared_ptr<Recorder> InstanceConfig::getRecorder() const {
    return std::make_shared<Recorder>(instanceParams_.loggerParams.opCsvFile);
}

Sequences InstanceConfig::getSequences() const {
    return parseDatasetJson(
        datasetPath_,
        instanceParams_.reqParams.maxNumSequences
    );
}

std::vector<texec::Request> InstanceConfig::getRequests(std::optional<size_t> num) const {
    auto const seqs = getSequences();
    auto numSequences = seqs.size();

    num = num.has_value() ? std::min(num.value(), numSequences) : numSequences;
    std::vector<texec::Request> requests;
    for (int i = 0; i < num; ++i) {
        requests.emplace_back(
            texec::Request {
                seqs[i].inputIds,
                seqs[i].outputLen,
                instanceParams_.engineParams.streaming,
                samplingConfig_,
                outputConfig_,
                instanceParams_.modelParams.eosId,
                instanceParams_.modelParams.padId
            }
        );
    }
    return requests;
}