#include "cmmlu/cmmlu.h"
#include <unordered_map>

std::unordered_map<std::string, std::string> name_en2zh = {
    {"agronomy", "农学"},
    {"anatomy", "解剖学"},
    {"ancient_chinese", "古汉语"},
    {"arts", "艺术学"},
    {"astronomy", "天文学"},
    {"business_ethics", "商业伦理"},
    {"chinese_civil_service_exam", "中国公务员考试"},
    {"chinese_driving_rule", "中国驾驶规则"},
    {"chinese_food_culture", "中国饮食文化"},
    {"chinese_foreign_policy", "中国外交政策"},
    {"chinese_history", "中国历史"},
    {"chinese_literature", "中国文学"},
    {"chinese_teacher_qualification", "中国教师资格"},
    {"clinical_knowledge", "临床知识"},
    {"college_actuarial_science", "大学精算学"},
    {"college_education", "大学教育学"},
    {"college_engineering_hydrology", "大学工程水文学"},
    {"college_law", "大学法律"},
    {"college_mathematics", "大学数学"},
    {"college_medical_statistics", "大学医学统计"},
    {"college_medicine", "大学医学"},
    {"computer_science", "计算机科学"},
    {"computer_security", "计算机安全"},
    {"conceptual_physics", "概念物理学"},
    {"construction_project_management", "建设工程管理"},
    {"economics", "经济学"},
    {"education", "教育学"},
    {"electrical_engineering", "电气工程"},
    {"elementary_chinese", "小学语文"},
    {"elementary_commonsense", "小学常识"},
    {"elementary_information_and_technology", "小学信息技术"},
    {"elementary_mathematics", "初等数学"},
    {"ethnology", "民族学"},
    {"food_science", "食品科学"},
    {"genetics", "遗传学"},
    {"global_facts", "全球事实"},
    {"high_school_biology", "高中生物"},
    {"high_school_chemistry", "高中化学"},
    {"high_school_geography", "高中地理"},
    {"high_school_mathematics", "高中数学"},
    {"high_school_physics", "高中物理学"},
    {"high_school_politics", "高中政治"},
    {"human_sexuality", "人类性行为"},
    {"international_law", "国际法学"},
    {"journalism", "新闻学"},
    {"jurisprudence", "法理学"},
    {"legal_and_moral_basis", "法律与道德基础"},
    {"logical", "逻辑学"},
    {"machine_learning", "机器学习"},
    {"management", "管理学"},
    {"marketing", "市场营销"},
    {"marxist_theory", "马克思主义理论"},
    {"modern_chinese", "现代汉语"},
    {"nutrition", "营养学"},
    {"philosophy", "哲学"},
    {"professional_accounting", "专业会计"},
    {"professional_law", "专业法学"},
    {"professional_medicine", "专业医学"},
    {"professional_psychology", "专业心理学"},
    {"public_relations", "公共关系"},
    {"security_study", "安全研究"},
    {"sociology", "社会学"},
    {"sports_science", "体育学"},
    {"traditional_chinese_medicine", "中医中药"},
    {"virology", "病毒学"},
    {"world_history", "世界历史"},
    {"world_religions", "世界宗教"}
};

inline static std::string LoadBytesFromFile(const std::filesystem::path& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
      std::cerr << "Cannot open tokenzier: " << path.string() << std::endl;
      exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

std::string CMMLU::format_question(const std::string& question, const std::vector<std::string>& options, std::string answer, bool ex = false) {
    std::string clabels = "ABCD";
    std::string text = "问题:\n";
    text += question + "\n\n选项:\n";
    // for (size_t i = 0; i < options.size(); ++i) {
    //     text += "A " + ": " + options[i] + "\n";
    // }
    text += "A: " ;
    text += options[0] + "\n";
    text += "B: " ;
    text += options[1] + "\n";
    text += "C: " ;
    text += options[2] + "\n";
    text += "D: " ;
    text += options[3] + "\n";
    text += "\n答案: ";
    if (ex) {
        text += answer[0];
        text += "\n";
    }
    //std::cout<<text;
    return text;
}

// csv format:
// ,Question,A,B,C,D,Answer
// 0,下列作物的果实为荚果的是,花生,向日葵,油菜,荞麦,A
SeqQ CMMLU::readDatasetFromCSV(
    const std::filesystem::path& datasetPath, 
    const std::filesystem::path& tokenizerPath
){   
    auto blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
 
    SeqQ seqs;
    std::ifstream file(datasetPath);
    std::string fileNameWithoutExtension = datasetPath.stem();
    std::string line;
    getline(file, line);
    int examples_num = 3;
    std::string examples_prompt;
    for(int i = 0;i < examples_num ;i++){
        getline(file, line);
        std::stringstream ss(line);
        std::string temp;
        getline(ss, temp, ',');
        Sequence seq;
        std::string question;
        std::string optionA;
        std::string optionB;
        std::string optionC;
        std::string optionD;
        std::string answer;
        getline(ss,question, ',');
        getline(ss, optionA, ',');
        getline(ss, optionB, ',');
        getline(ss, optionC, ',');
        getline(ss, optionD, ',');
        getline(ss, answer, ' '); 

        examples_prompt += format_question(
            question,
            {
                optionA,optionB,optionC,optionD
            },
            answer,
            true
            );
        examples_prompt += "\n\n";  
    }
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string temp;
        getline(ss, temp, ',');
        Sequence seq;
        std::string question ;
        std::string optionA;
        std::string optionB;
        std::string optionC;
        std::string optionD;
        std::string answer;
        getline(ss,question, ',');
        getline(ss, optionA, ',');
        getline(ss, optionB, ',');
        getline(ss, optionC, ',');
        getline(ss, optionD, ',');
        getline(ss, answer, ' ');

        std::string q_prompt = format_question(
            question,
            {
                optionA,optionB,optionC,optionD
            },
            answer
            );
        std::string category = name_en2zh[fileNameWithoutExtension];
        std::string prompts = "以下是关于(" + category + ")的单项选择题，请直接给出正确答案的选项。\n\n";
        prompts += examples_prompt;
        prompts += q_prompt;
        seq.inputIds = tok->Encode(prompts);
        seq.outputLen = 20;
        seq.answer = answer[0];
        seq.optionA = optionA;
        seq.optionB = optionB;
        seq.optionC = optionC;
        seq.optionD = optionD;
        seqs.push(seq);
    }
    return seqs;
}

SeqQ CMMLU::readDatasetFromCSVfolder(const std::filesystem::path& folderPath, const std::filesystem::path& tokenizerPath) {
    
    auto blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    SeqQ seqs;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                std::filesystem::path datasetPath = entry.path();
                std::string fileNameWithoutExtension = datasetPath.stem();
                std::ifstream file(datasetPath);
                std::string line;
                getline(file, line);
                int examples_num = 3;
                std::string examples_prompt;
                for(int i = 0;i < examples_num ;i++){
                    getline(file, line);
                    std::stringstream ss(line);
                    std::string temp;
                    getline(ss, temp, ',');
                    Sequence seq;
                    std::string question;
                    std::string optionA;
                    std::string optionB;
                    std::string optionC;
                    std::string optionD;
                    std::string answer;
                    getline(ss,question, ',');
                    getline(ss, optionA, ',');
                    getline(ss, optionB, ',');
                    getline(ss, optionC, ',');
                    getline(ss, optionD, ',');
                    getline(ss, answer, ' ');

                    examples_prompt += format_question(
                        question,
                        {
                            optionA,optionB,optionC,optionD
                        },
                        answer,
                        true
                        );
                    examples_prompt += "\n\n";  
                }

                while (getline(file, line)) {
                    std::stringstream ss(line);
                    std::string temp;
                    getline(ss, temp, ',');
                    Sequence seq;
                    std::string question ;
                    std::string optionA;
                    std::string optionB;
                    std::string optionC;
                    std::string optionD;
                    std::string answer;
                    getline(ss,question, ',');
                    getline(ss, optionA, ',');
                    getline(ss, optionB, ',');
                    getline(ss, optionC, ',');
                    getline(ss, optionD, ',');
                    getline(ss, answer, ' ');
                    // prompts.append("以下是关于({})的单项选择题，请直接给出正确答案的选项。\n\n".format(name_en2zh[file_name_without_extension]) 
                    //     + examples_prompt + q_prompt)
                    
                    std::string q_prompt = format_question(
                        question,
                        {
                            optionA,optionB,optionC,optionD
                        },
                        answer
                        );
                    std::string category = name_en2zh[fileNameWithoutExtension];
                    std::string prompts = "以下是关于(" + category + ")的单项选择题，请直接给出正确答案的选项。\n\n";
                    prompts += examples_prompt;
                    prompts += q_prompt;
                    seq.inputIds = tok->Encode(prompts);
                    seq.outputLen = 20;
                    seq.answer = answer[0];
                    seq.optionA = optionA;
                    seq.optionB = optionB;
                    seq.optionC = optionC;
                    seq.optionD = optionD;
                    seqs.push(seq);
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
    return seqs; 
}

char extract_choice(std::string response, std::vector<std::string> option_contents) {
    std::vector<char> choices = {'A', 'B', 'C', 'D'};
    std::string answer;
    std::vector<std::pair<std::string, int>> patterns;
    if (response[0] == 'A' || response[0] == 'B' || response[0] == 'C' || response[0] == 'D') {
        return response[0];
    }
    // 1. Single match
    patterns = {
        {R"(答案(选项)?(是|为)：? ?([ABCD]))", 3},
        {R"(答案(是|为)选项 ?([ABCD]))", 2},
        {R"(故?选择?：? ?([ABCD]))", 1},
        {R"(([ABCD]) ?选?项(是｜为)?正确)", 1},
        {R"(正确的?选项(是|为) ?([ABCD]))", 2},
        {R"(答案(应该)?(是|为)([ABCD]))", 3},
        {R"(选项 ?([ABCD]) ?(是|为)?正确)", 1},
        {R"(选择答案 ?([ABCD]))", 1},
        {R"(答案?：?([ABCD]))", 1},
        {R"(([ABCD])(选?项)?是?符合题意)", 1},
        {R"(答案选项：? ?([ABCD]))", 1}, 
        {R"(答案(选项)?为(.*?)([ABCD]))", 3}, 
    };
    
    for (const auto& [pattern, idx] : patterns) {
        std::smatch m;
        if (std::regex_search(response, m, std::regex(pattern))) {
            answer = m.str(idx);
            assert(answer[0] == 'A' || answer[0] == 'B' || answer[0] == 'C' || answer[0] == 'D');
            return answer[0];
        }
    }
    
    //2. Recursive match
    patterns = {
        {R"(([ABCD])(.*?)当选)", 1},
        {R"(([ABCD])(.*?)正确)", 1}
    };
    
    for (const auto& [pattern, idx] : patterns) {
        std::regex re(pattern);
        std::smatch m;
        if (std::regex_search(response, m, re)) {
            std::string matchStr = m.str();
            while (std::regex_search(matchStr, m, re)) {
                answer = m.str(idx);
                matchStr = m.suffix();
            }
            assert(answer[0] == 'A' || answer[0] == 'B' || answer[0] == 'C' || answer[0] == 'D');
            return answer[0];
        }
    }
    // 3. Weak single match
    patterns = {
        {R"([^不]是：? ?([ABCD]))", 1}
    };
    for (const auto& [pattern, idx] : patterns) {
        std::regex re(pattern);
        std::smatch m;
        if (std::regex_search(response, m, re)) {
            answer = m.str(idx);
            assert(answer[0] == 'A' || answer[0] == 'B' || answer[0] == 'C' || answer[0] == 'D');
            return answer[0];
        }
    }
    
    // 4. Check the only mentioend choices
    std::regex pattern(R"(^[^ABCD]*([ABCD])[^ABCD]*$)");
    std::smatch m;
    if (std::regex_match(response, m, pattern)) {
        answer = m.str(1);
        assert(answer[0] == 'A' || answer[0] == 'B' || answer[0] == 'C' || answer[0] == 'D');
        return answer[0];
    }
    
    // 5. Match the option contents
    for (size_t i = 0; i < option_contents.size(); ++i) {
        if (response.find(option_contents[i]) != std::string::npos) {
            return choices[i];
        }
    }
    srand((unsigned)time(NULL)); 
    return choices[(rand() % (4))];
}

float CMMLU::output_acc(SeqV V_input,std::variant<ResultV, std::vector<std::vector<float>>> Vector_output,const std::filesystem::path& tokenizerPath){
    auto blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    int num = V_input.size();
    ResultV output_ids;
    assert(std::holds_alternative<ResultV>(Vector_output));
    output_ids = std::move(std::get<ResultV>(Vector_output));
    float acc = 0;
    for(int i  = 0;i < num ;i++ ){
        
        std::vector<int> result_id(output_ids.at(i).begin() + V_input.at(i).inputIds.size(), output_ids.at(i).end());
        char answer = V_input.at(i).answer.value();
        std::string result = tok->Decode(result_id);
        
        // std::cout<<i<<":[result]"<<result<<std::endl;
        // std::cout<<":[answer]"<<answer<<std::endl;
        
        char output_answer = extract_choice(result,{V_input.at(i).optionA.value(),V_input.at(i).optionB.value(),V_input.at(i).optionC.value(),V_input.at(i).optionD.value()});
        if (output_answer==answer){
            acc++;
            //std::cout<<"=====yes===="<<std::endl;
        }
        
    }
    acc = acc/num;
    return acc;
}