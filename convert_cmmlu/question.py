import os
import csv
import json

from name_en2zh import name_en2zh
categories = list(name_en2zh.keys())

# 指定csv文件夹和json文件夹的路径
csv_folder = 'test_csv'
json_folder = 'test_json'

examples_per_category = 3

def format_question(question, options, answer, ex = False):

    clabels = "ABCD"
    text = f"问题:\n"
    text += question
    text += "\n\n选项:\n"
    for i, o in enumerate(options):
        text += clabels[i] + ": " + o + "\n"
    text += "\n答案: "
    if ex == True:
        text += answer + "\n"
    # text += "\nAnswer: "
    # if ex:
    #     text += ", " + options[answer]
    return text


def main():
    # 如果json文件夹不存在，就创建一个
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)


    # 遍历csv文件夹中的所有csv文件
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            with open(os.path.join(csv_folder, filename), newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                prompts = []


                examples_prompt = ""
                i = 0
                for row in reader:
                    examples_prompt += format_question(
                        row["Question"], 
                        [
                            f"{row['A']}",
                            f"{row['B']}",
                            f"{row['C']}",
                            f"{row['D']}",
                        ], 
                        row["Answer"], 
                        ex = True
                    )
                    i+=1
                    examples_prompt += "\n\n"
                    if i>2:
                        break
                for row in reader:
                    q_prompt = format_question(
                        row["Question"], 
                        [
                            f"{row['A']}",
                            f"{row['B']}",
                            f"{row['C']}",
                            f"{row['D']}",
                        ], 
                        row["Answer"]
                        )
                    file_name_without_extension = os.path.splitext(filename)[0]
                    prompts.append("以下是关于({})的单项选择题，请直接给出正确答案的选项。\n\n".format(name_en2zh[file_name_without_extension]) 
                            + examples_prompt + q_prompt)

            data = {"Prompts": prompts}

            # 生成对应的json文件并放入json文件夹
            json_filename = os.path.splitext(filename)[0] + '.json'  # 使用csv文件名来命名对应的json文件
            with open(os.path.join(json_folder, json_filename), 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, ensure_ascii=False, indent=4)

    print("操作完成！")


if __name__ == "__main__":
    main()