import os
import csv
import json

from name_en2zh import name_en2zh
categories = list(name_en2zh.keys())

# 指定csv文件夹和json文件夹的路径
csv_folder = 'test_csv'
json_folder = 'rrr'
def main():
    # 如果json文件夹不存在，就创建一个
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    prompts = []
    # 遍历csv文件夹中的所有csv文件
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            with open(os.path.join(csv_folder, filename), newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                file_name_without_extension = os.path.splitext(filename)[0]
                prompts.append("./build/batched_inference --engine_dir ~/trt-llm-engines/bs256-h800/ --dataset ./test_json/{}.json --output ./output_json/{}.json --eos_id 100007 --pad_id 0;".format(file_name_without_extension,file_name_without_extension))
    
            json_filename = os.path.splitext(filename)[0] + '.json'  # 使用csv文件名来命名对应的json文件
    data = {"Prompts": prompts}

            # 生成对应的json文件并放入json文件夹
    
    with open(os.path.join(json_folder, json_filename), 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)

    print("操作完成！")


if __name__ == "__main__":
    main()