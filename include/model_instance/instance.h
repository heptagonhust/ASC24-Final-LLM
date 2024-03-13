#pragma once
#include "model_instance/config.h"
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

class Instance {
public:
    Instance(InstanceParams instanceParams);
    void run();
    void writeResultsToJson(const fs::path& outputPath) const;

private:
    InstanceParams instanceParams_;
    std::shared_ptr<InstanceConfig> config_;
    std::shared_ptr<ExecutorServer> executorServer_;
    std::shared_ptr<Recorder> recorder_;
};
