#include "mmlu/mmlu.h"
#include <unordered_map>

std::set<std::string> categories = {"biology", "chemistry", "chinese",
                                    "english", "geography", "history",
                                    "math",    "physics"};

std::string LoadBytesFromFile(const std::filesystem::path &path) {
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

std::string format_question(const std::string &question,
                            const std::vector<std::string> &options,
                            std::string answer, bool ex = false) {
  std::string clabels = "ABCD";
  std::string text = "问题:\n";
  text += question + "\n\n选项:\n";
  // for (size_t i = 0; i < options.size(); ++i) {
  //     text += "A " + ": " + options[i] + "\n";
  // }
  text += "A: ";
  text += options[0] + "\n";
  text += "B: ";
  text += options[1] + "\n";
  text += "C: ";
  text += options[2] + "\n";
  text += "D: ";
  text += options[3] + "\n";
  text += "\n答案: ";
  if (ex) {
    text += answer[0];
    text += "\n";
  }
  // std::cout<<text;
  return text;
}

// csv format:
// ,Question,A,B,C,D,Answer
// 0,下列作物的果实为荚果的是,花生,向日葵,油菜,荞麦,A
SeqQ readDatasetFromCSV(const std::filesystem::path &datasetPath,
                        const std::filesystem::path &tokenizerPath) {
  auto blob = LoadBytesFromFile(tokenizerPath);
  auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);

  SeqQ seqs;
  std::ifstream file(datasetPath);
  std::string fileNameWithoutExtension = datasetPath.stem();
  std::string line;
  getline(file, line);
  int examples_num = 3;
  std::string examples_prompt;
  for (int i = 0; i < examples_num; i++) {
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
    getline(ss, question, ',');
    getline(ss, optionA, ',');
    getline(ss, optionB, ',');
    getline(ss, optionC, ',');
    getline(ss, optionD, ',');
    getline(ss, answer, ' ');

    examples_prompt += format_question(
        question, {optionA, optionB, optionC, optionD}, answer, true);
    examples_prompt += "\n\n";
  }
  while (getline(file, line)) {
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
    getline(ss, question, ',');
    getline(ss, optionA, ',');
    getline(ss, optionB, ',');
    getline(ss, optionC, ',');
    getline(ss, optionD, ',');
    getline(ss, answer, ' ');

    std::string q_prompt =
        format_question(question, {optionA, optionB, optionC, optionD}, answer);
    std::string category = fileNameWithoutExtension;
    std::string prompts = "以下是关于(" + category +
                          ")的单项选择题，请直接给出正确答案的选项。\n\n";
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

SeqQ readDatasetFromCSVfolder(const std::filesystem::path &folderPath,
                              const std::filesystem::path &tokenizerPath) {

  auto blob = LoadBytesFromFile(tokenizerPath);
  auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
  SeqQ seqs;
  try {
    for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
      if (entry.is_regular_file() && entry.path().extension() == ".csv") {
        std::filesystem::path datasetPath = entry.path();
        std::string fileNameWithoutExtension = datasetPath.stem();
        std::ifstream file(datasetPath);
        std::string line;
        getline(file, line);
        int examples_num = 3;
        std::string examples_prompt;
        for (int i = 0; i < examples_num; i++) {
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
          getline(ss, question, ',');
          getline(ss, optionA, ',');
          getline(ss, optionB, ',');
          getline(ss, optionC, ',');
          getline(ss, optionD, ',');
          getline(ss, answer, ' ');

          examples_prompt += format_question(
              question, {optionA, optionB, optionC, optionD}, answer, true);
          examples_prompt += "\n\n";
        }

        while (getline(file, line)) {
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
          getline(ss, question, ',');
          getline(ss, optionA, ',');
          getline(ss, optionB, ',');
          getline(ss, optionC, ',');
          getline(ss, optionD, ',');
          getline(ss, answer, ' ');

          std::string q_prompt = format_question(
              question, {optionA, optionB, optionC, optionD}, answer);
          std::string category = fileNameWithoutExtension;
          std::string prompts = "以下是关于(" + category +
                                ")的单项选择题，请直接给出正确答案的选项。\n\n";
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
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
  }
  return seqs;
}

float output_acc(SeqV V_input, std::vector<std::vector<float>> logits_output,
                 const std::filesystem::path &tokenizerPath) {
  auto blob = LoadBytesFromFile(tokenizerPath);
  auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
  int num = logits_output.size();
  float acc = 0;
  std::vector<int> ABCD_token_id = {
      tok->TokenToId("A"),
      tok->TokenToId("B"),
      tok->TokenToId("C"),
      tok->TokenToId("D"),
  };
  for (int i = 0; i < num; i++) {
    std::vector<float> logits = logits_output[i];
    std::vector<int> inputIds = V_input[i].inputIds;
    assert(!V_input[i].answer.has_value());
    char answer = V_input[i].answer.value();
    int max_logits_id = ABCD_token_id[0];
    for (int j = 1; j < 4; j++) {
      if (logits[ABCD_token_id[j]] > logits[max_logits_id]) {
        max_logits_id = ABCD_token_id[j];
      }
    }
    if (max_logits_id == ABCD_token_id[answer - 'A']) {
      acc += 1;
    }
  }
  acc = acc / num;
  return acc;
}
