#include <memory>
#include <optional>

#include "tensorrt_llm/runtime/worldConfig.h"
#include "model_instance/config.h"


std::shared_ptr<InstanceConfig> InstanceConfig::from_params(const InstanceParams &instanceParams) {
    auto world_config = WorldConfig::mpi();
    if (world_config.getSize() > 1)
    {
        TLLM_THROW("benchmarkExecutor does not yet support mpiSize > 1");
    }

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
        instanceParams.sampleParms.maxBeamWidth,
        50,
        1.0,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        1.0
    };
    texec::OutputConfig outputConfig {
        false,
        instanceParams.outputParams.returnContextLogits,
        instanceParams.outputParams.returnGenerationLogits,
        false
    };

    return std::make_shared<InstanceConfig>(
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

