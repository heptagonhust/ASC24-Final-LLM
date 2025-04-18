#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <string>

#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "model_instance/config.h"
#include "model_instance/instance.h"

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;


namespace
{

void singleGPUinstance(InstanceParams instanceParams) {
    Instance instance(std::move(instanceParams));
    instance.run();
}

} // namespace

int main(int argc, char* argv[])
{
    cxxopts::Options options(
        "TensorRT-LLM BatchManager Benchmark", "TensorRT-LLM BatchManager Benchmark for GPT and GPT-like models.");
    options.add_options()("h,help", "Print usage");
    // TODO(rkobus): remove because unused
    options.add_options()("engine_dir", "Directory that store the engines.", cxxopts::value<std::string>());
    options.add_options()(
        "eos_id", "Specify the end-of-sequence token id.", cxxopts::value<int>()->default_value("-1"));
    options.add_options()("pad_id", "Specify the padding token id.", cxxopts::value<int>());
    options.add_options()("output", "Output file for the results.", 
        cxxopts::value<std::string>()->default_value("output.json"));
    options.add_options()(
        "output_csv", "Write output metrics to CSV", cxxopts::value<std::string>()->default_value(""));
    options.add_options()(
        "beam_width", "Specify beam width you want to benchmark.", cxxopts::value<int>()->default_value("1"));
    options.add_options()(
        "warm_up", "Specify warm up iterations before benchmark starts.", cxxopts::value<int>()->default_value("2"));
    options.add_options()("max_tokens_in_paged_kvcache", "Max tokens in paged K-V Cache.", cxxopts::value<int>());
    options.add_options()(
        "kv_cache_free_gpu_mem_fraction", "K-V Cache Free Gpu Mem Fraction.", cxxopts::value<float>());
    options.add_options()("enable_trt_overlap", "Overlap TRT context preparation and execution",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("streaming", "Operate in streaming mode", cxxopts::value<bool>()->default_value("false"));
    options.add_options()(
        "enable_kv_cache_reuse", "Enables the KV cache reuse.", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("enable_chunked_context", "Whether to enable context chunking.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()(
        "return_context_logits", "Whether to return context logits.", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("return_generation_logits", "Whether to return generation logits.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("scheduler_policy", "Choose scheduler policy between max_utilization/guaranteed_no_evict.",
        cxxopts::value<std::string>()->default_value("guaranteed_no_evict"));
    options.add_options()("log_level", "Choose log level between verbose/info/warning/error/internal_error.",
        cxxopts::value<std::string>()->default_value("error"));
    options.add_options()("log_iteration_data", "On each decoder iteration, print batch state metadata.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("show_result", "Show the result of the model.", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("wait_sleep", "Specify how many milliseconds to sleep each iteration of waitForEmpty loop.",
        cxxopts::value<int>()->default_value("25"));
    options.add_options()("rpc_address", "Specify the address of the server.", cxxopts::value<std::string>());
    options.add_options()("rpc_port", "Specify the port of the server.", cxxopts::value<int>());
    options.add_options()("rpc_seqs_threshold", "Specify the threshold number of seqs that trigger the next rpc getseqs call", 
        cxxopts::value<int>()->default_value("300"));
    options.add_options()("rpc_batch_size", "Specify the transmitted batch size of seqs in a single rpc request.",
        cxxopts::value<int>()->default_value("500"));
    auto result = options.parse(argc, argv);

    InstanceParams instanceParams;

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Argument: Engine directory
    if (!result.count("engine_dir"))
    {
        std::cout << options.help() << std::endl;
        TLLM_LOG_ERROR("Please specify engine directory.");
        return 1;
    }

    // model parameters
    {
        // Argument: Padding token id
        if (result.count("pad_id"))
        {
            instanceParams.modelParams.padId = result["pad_id"].as<int>();
        }

        // Argument: End-of-sentence token id
        if (result.count("eos_id"))
        {
            instanceParams.modelParams.eosId = result["eos_id"].as<int>();
        }
    }

    // Engine Parameters
    {
        instanceParams.engineParams.engine_dir = result["engine_dir"].as<std::string>();
        // Argument: Enable TRT overlap
        instanceParams.engineParams.enableTrtOverlap = result["enable_trt_overlap"].as<bool>();
        // Argument: streaming
        instanceParams.engineParams.streaming = result["streaming"].as<bool>();
    }

    // Scheduler Parameters
    {
        // Argument: Scheduler policy
        auto const schedulerPolicyArg = result["scheduler_policy"].as<std::string>();
        if (schedulerPolicyArg == "max_utilization") {
            instanceParams.scheduleParams.schedulerPolicy = 
                batch_scheduler::SchedulerPolicy::MAX_UTILIZATION;
        }
        else if (schedulerPolicyArg == "guaranteed_no_evict") {
            instanceParams.scheduleParams.schedulerPolicy = 
                batch_scheduler::SchedulerPolicy::GUARANTEED_NO_EVICT;
        }
        else {
            TLLM_LOG_ERROR("Unexpected scheduler policy: " + schedulerPolicyArg);
            return 1;
        }
    }

    // Cache Parameters
    {
        // Argument: Enable KV cache reuse
        instanceParams.cacheParams.enableBlockReuse = 
            result["enable_kv_cache_reuse"].as<bool>();

        // Argument: Enable chunked context
        instanceParams.cacheParams.enableChunkedContext = 
            result["enable_chunked_context"].as<bool>();

        // Argument: Max tokens in paged K-V Cache
        if (result.count("max_tokens_in_paged_kvcache")) {
            instanceParams.cacheParams.maxTokensInPagedKvCache = 
                result["max_tokens_in_paged_kvcache"].as<int>();
        }

        // Argument: K-V Cache Free Gpu Mem Fraction
        if (result.count("kv_cache_free_gpu_mem_fraction")) {
            instanceParams.cacheParams.freeGpuMemoryFraction = 
                result["kv_cache_free_gpu_mem_fraction"].as<float>();
        }
        
    }

    // Server Parameters
    {
        // Argument: wait_sleep
        instanceParams.serverParams.waitSleep = std::chrono::milliseconds(result["wait_sleep"].as<int>());

        // Argument: Warm up iterations
        instanceParams.serverParams.warm_up_iterations = result["warm_up"].as<int>();
    }

    // Output Parameters
    {
        // Argument: Enable return context logits
        instanceParams.outputParams.returnContextLogits = result["return_context_logits"].as<bool>();
        // Argument: Enable return generation logits
        instanceParams.outputParams.returnGenerationLogits = result["return_generation_logits"].as<bool>();

    }

    // sample Parameters
    {
        // Argument: beam width
        instanceParams.sampleParms.maxBeamWidth = result["beam_width"].as<int>();
    }

    // logger Parameters
    {
        // Argument: Enable batch stats output
        instanceParams.loggerParams.logIterationData = result["log_iteration_data"].as<bool>();

        // Argument: Output metrics CSV
        instanceParams.loggerParams.opCsvFile = result["output_csv"].as<std::string>();
        instanceParams.loggerParams.showResults = result["show_result"].as<bool>();
    }

    // rpc Parameters
    {
        instanceParams.rpcParams.rpcAddress = result["rpc_address"].as<std::string>();
        instanceParams.rpcParams.rpcPort = result["rpc_port"].as<int>();
        instanceParams.rpcParams.rpcNseqsThreshold = result["rpc_seqs_threshold"].as<int>();
        instanceParams.rpcParams.rpcNseqsBatchsize = result["rpc_batch_size"].as<int>();
    }

    // Argument: Log level
    auto logger = std::make_shared<TllmLogger>();
    auto const logLevel = result["log_level"].as<std::string>();
    if (logLevel == "verbose")
    {
        logger->setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else if (logLevel == "info")
    {
        logger->setLevel(nvinfer1::ILogger::Severity::kINFO);
    }
    else if (logLevel == "warning")
    {
        logger->setLevel(nvinfer1::ILogger::Severity::kWARNING);
    }
    else if (logLevel == "error")
    {
        logger->setLevel(nvinfer1::ILogger::Severity::kERROR);
    }
    else if (logLevel == "internal_error")
    {
        logger->setLevel(nvinfer1::ILogger::Severity::kINTERNAL_ERROR);
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected log level: " + logLevel);
        return 1;
    }

    initTrtLlmPlugins(logger.get());

    try
    {
        singleGPUinstance(std::move(instanceParams));
    }
    catch (const std::exception& e)
    {
        TLLM_LOG_ERROR(e.what());
        return 1;
    }
    return 0;
}
