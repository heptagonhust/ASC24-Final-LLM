#include <iostream>
#include <cxxopts.hpp>

using namespace std;

int main(int argc, char* argv[]) {
    cxxopts::Options options("TensorRT-LLM single GPU runtime instance", 
                             "A instance to run trt on single gpu");
    options.add_options()("engine_dir", "Directory that store the engines.", cxxopts::value<string>());
    auto args = options.parse(argc, argv);

    if (!args.count("engine_dir")) {
        cout << "Please specify the engine directory." << endl;
        return 1;
    }
    else {
        cout << "Engine directory: " << args["engine_dir"].as<string>() << endl;
    }
    cout << "TEST" << endl;
    return 0;
}