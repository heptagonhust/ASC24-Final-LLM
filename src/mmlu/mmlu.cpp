#include "mmlu/mmlu.h"
#include "sequences/sequences.h"
#include "tokenizers/tokenizers_cpp.h"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


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

void convertSeqAndQuestionToStr(
    std::string& ret,
    const Sequence& seq,
    const std::string& question
) {
    ret = "";
    ret += "Question:\n";
    ret += question;
    ret += "\n\nChoices:";
    ret += "\nA: "; ret += seq.optionA.value();
    ret += "\nB: "; ret += seq.optionB.value();
    ret += "\nC: "; ret += seq.optionC.value();
    ret += "\nD: "; ret += seq.optionD.value();
    ret += "\n\nAnswer: "; ret.push_back(seq.answer.value());
    ret += "\n\n\n";
}


/**
 * @brief Generate heading prompts
 * 
 * @param prompt 
 * @param seqs 
 * @param questions 
 * @param categoryName 
 */
void generateHeadingPrompt(
    std::string& ret,
    const std::vector<Sequence>& seqs,
    const std::vector<std::string>& questions,
    const std::string& categoryName
) {
    int size = seqs.size();
    ret = "The following are multiple choice questions (with answers) about ";
    ret += categoryName;
    ret += ".\n\n";

    // Put example questions
    for (int i = 0; i < size; i++) {
        std::string localPrompt;
        convertSeqAndQuestionToStr(localPrompt, seqs[i], questions[i]);
        ret += localPrompt;
    }
}

void getFormattedQuestionIntoSeq(
    Sequence& seq,
    const std::string& question,
    const std::string& headingPrompt,
    std::shared_ptr<tokenizers::Tokenizer> tknizer
) {
    std::string prompt = headingPrompt;

    std::string realQuestion;
    convertSeqAndQuestionToStr(realQuestion, seq, question);
    // Get rid of Answer
    realQuestion.erase(realQuestion.size() - 4);

    prompt += realQuestion;

    seq.inputIds = tknizer->Encode(prompt);
}

std::string MMLU::format_question(const std::string &question,
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


/**
 * @brief Read one question in to seq. This function will not
          modify seq'inputIds. Question will be return in &string.
 * 
 * @param ss 
 * @param seq 
 * @param question 
 * @return true 
 * @return false 
 */
bool readOneQuestion(
    std::stringstream& ss,
    Sequence& seq,
    std::string& question
) {
    // read 
    const int segCount = 6;
    for (int i = 0; i < segCount; i++) {
        std::string seg;
        char tmpChar = 0;
        // We have a quoted segmentation
        if (ss.peek() == '"') {
            while (!ss.eof() && ss.get(tmpChar)) {
                // closing quote is found
                if (tmpChar == '"') {
                    seg.erase(0, 1);
                    break;
                }
                seg.push_back(tmpChar);
            }
        } else {
            getline(ss, seg, ',');
        }
        switch (i) {
        case 0: question = seg; break;
        case 1: seq.optionA = seg; break;
        case 2: seq.optionB = seg; break;
        case 3: seq.optionC = seg; break;
        case 4: seq.optionD = seg; break;
        case 5: seq.answer = seg[0]; break;
        default: throw "Invalid segmentation is detected!";
        }
    }
    return true;
}

// csv format:
// ,Question,A,B,C,D,Answer
// 0,下列作物的果实为荚果的是,花生,向日葵,油菜,荞麦,A
SeqQ MMLU::readDatasetFromCSV(
    const std::filesystem::path& datasetPath, 
    const std::filesystem::path& tokenizerPath
){   
    const int NUM_EXAMPLES = 3;
    std::string blob = LoadBytesFromFile(tokenizerPath);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    std::shared_ptr<tokenizers::Tokenizer> tknizer = std::move(tok);
 
    SeqQ seqs;
    std::ifstream file(datasetPath);
    std::string fileNameWithoutExtension = datasetPath.stem();
    std::string line;
    std::vector<Sequence> example_seqs;
    std::vector<std::string> example_questions;
    // skip first line
    getline(file, line);
    
    // Get example problems
    for(int i = 0; i < NUM_EXAMPLES; i++){
        getline(file, line);
        std::stringstream ss(line);

        Sequence seq;
        std::string question;

        readOneQuestion(ss, seq, question);

        example_seqs.push_back(seq);
        example_questions.push_back(question);
    }

    // prepare heading prompt
    std::string headingPrompt;
    generateHeadingPrompt(headingPrompt, example_seqs, example_questions, 
                          fileNameWithoutExtension.substr(0, fileNameWithoutExtension.size() - 5));
    // get each problem
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string temp;

        Sequence seq;
        std::string question;
        readOneQuestion(ss, seq, question);

        getFormattedQuestionIntoSeq(seq, question, headingPrompt, tknizer);
        
        seqs.push(seq);
    }
    return seqs;
}

SeqQ MMLU::readDatasetFromCSVfolder(const std::filesystem::path &folderPath,
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

float MMLU::output_acc(SeqV V_input, std::variant<ResultV, std::vector<std::vector<float>>> Vector_output,
                 const std::filesystem::path &tokenizerPath) {
  auto blob = LoadBytesFromFile(tokenizerPath);
  auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
  std::vector<std::vector<float>> logits_output;
  assert(std::holds_alternative<std::vector<std::vector<float>>>(Vector_output));
  logits_output = std::get<std::vector<std::vector<float>>>(Vector_output);

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