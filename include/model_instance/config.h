#pragma once
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "tensorrt_llm/runtime/worldConfig.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/schedulerPolicy.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include "model_instance/recorder.h"
#include "model_instance/executor.h"

namespace tb = tensorrt_llm::batch_manager;
namespace tbb = tensorrt_llm::batch_manager::batch_scheduler;
namespace fs = std::filesystem;
namespace texec = tensorrt_llm::executor;

struct InstanceParams
{

    struct ModelParams {
        std::optional<int32_t> padId = std::nullopt;
        std::optional<int32_t> eosId = std::nullopt;
    };

    struct EngineParams {
        fs::path engine_dir;
        bool enableTrtOverlap = false;
        bool streaming = false;
    };

    struct ScheduleParams {
        tb::TrtGptModelType BatchingType;
        tbb::SchedulerPolicy schedulerPolicy;
    };

    struct CacheParams {
        bool enableBlockReuse = false;
        bool enableChunkedContext = false;
        std::optional<SizeType> maxTokensInPagedKvCache = std::nullopt;
        std::optional<float> freeGpuMemoryFraction = std::nullopt;
    };

    struct ServerParams {
        std::chrono::milliseconds waitSleep;
        int warm_up_iterations;
    };

    struct RequestsParams {
        fs::path datasetPath;
        fs::path tokenzierPath;
        int maxNumSequences;
    };

    struct OutputParams {
        fs::path outputPath;
        bool returnContextLogits;
        bool returnGenerationLogits;
    };

    struct SampleParams {
        SizeType maxBeamWidth;
        std::optional<texec::FloatType> topP = 0.9;
    };

    struct LoggerParams {
        bool logIterationData;
        std::string opCsvFile;
    };
    
    ModelParams modelParams;
    EngineParams engineParams;
    ScheduleParams scheduleParams;
    CacheParams cacheParams;
    ServerParams serverParams;
    RequestsParams reqParams;
    OutputParams outputParams;
    SampleParams sampleParms;
    LoggerParams loggerParams;
};

class InstanceConfig {

public:
    InstanceConfig() = delete;
    InstanceConfig(
        WorldConfig worldConfig, 
        texec::ExecutorConfig executorConfig,
        texec::SamplingConfig samplingConfig,
        texec::OutputConfig outputConfig,
        InstanceParams instanceParams)
        : engineDir_(instanceParams.engineParams.engine_dir)
        , datasetPath_(instanceParams.reqParams.datasetPath)
        , tokenizerPath_(instanceParams.reqParams.tokenzierPath)
        , outputPath_(instanceParams.outputParams.outputPath)
        , worldConfig_(std::move(worldConfig))
        , executorConfig_(std::move(executorConfig)) 
        , samplingConfig_(std::move(samplingConfig))
        , outputConfig_(std::move(outputConfig))
        , instanceParams_(std::move(instanceParams)) {}

    [[nodiscard]] static std::shared_ptr<InstanceConfig> from_params(
        const InstanceParams& instanceParams
    );

    [[nodiscard]] std::shared_ptr<ExecutorServer> getExecutorServer(
        std::shared_ptr<Recorder> recorder
    ) const;

    [[nodiscard]] std::shared_ptr<Recorder> getRecorder() const;

    [[nodiscard]] const auto& getEngineDir() const { return engineDir_; }
    [[nodiscard]] const auto& getDatasetPath() const { return datasetPath_; }
    [[nodiscard]] const auto& getTokenizerPath() const { return tokenizerPath_; }
    [[nodiscard]] const auto& getOutputPath() const { return outputPath_; }

    [[nodiscard]] const auto& getWorldConfig() const { return worldConfig_; }
    [[nodiscard]] const auto& getExecutorConfig() const { return executorConfig_; }
    [[nodiscard]] const auto& getSamplingConfig() const { return samplingConfig_; }
    [[nodiscard]] const auto& getOutputConfig() const { return outputConfig_; }
    [[nodiscard]] const auto& getInstanceParams() const { return instanceParams_; }

private:
    fs::path engineDir_;
    fs::path datasetPath_;
    fs::path tokenizerPath_;
    fs::path outputPath_;

    WorldConfig worldConfig_;
    texec::ExecutorConfig executorConfig_;
    texec::SamplingConfig samplingConfig_;
    texec::OutputConfig outputConfig_;
    InstanceParams instanceParams_;
};