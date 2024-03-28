import os
import json
import csv
import re
import random
input_folder = 'output_json'
output_folder = 'answer'
csv_folder = 'test_csv'

# extract choice from response
def extract_choice(response, option_contents):
    '''
        Always return a choice, even cannot match by regex,
        to ensure fair comparison to other models.
    '''
    import re
    
    choices = ['A', 'B', 'C', 'D']
    response = str(response)
    if response[4] in choices:
        return response[4]
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])',1),
        (r'([ABCD]) ?选?项(是|为)?正确',1),
        (r'正确的?选项(是|为) ?([ABCD])',2),
        (r'答案(应该)?(是|为)([ABCD])',3),
        (r'选项 ?([ABCD]) ?(是|为)?正确',1),
        (r'选择答案 ?([ABCD])',1),
        (r'答案?：?([ABCD])',1),
        (r'([ABCD])(选?项)?是?符合题意',1),
        (r'答案选项：? ?([ABCD])', 1), # chatglm
        (r'答案(选项)?为(.*?)([ABCD])', 3), # chatgpt

    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 2. Recursive match
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer

    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCD])', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 4. Check the only mentioend choices
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer
    
    # 5. Match the option contents
    for i, content in enumerate(option_contents):
        if content in response[3:-1]:
            return choices[i]

    return choices[random.randint(0,3)]

def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        file_name_without_extension = os.path.splitext(filename)[0]
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
            with open(os.path.join(csv_folder, file_name_without_extension+".csv"), newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                output_data = []
                for _ in range(3):
                    next(reader)
                for row ,item in zip(reader,data):
                    content = [
                    f"{row['A']}",
                    f"{row['B']}",
                    f"{row['C']}",
                    f"{row['D']}",
                    ]
                    match = re.search(r'答案:(.*?)\"', item)

                    if match:
                        pred = match.group()
                        answer = extract_choice(pred, content)
                        if answer:
                            output_data.append({
                                "answer": answer
                            })
                    else:
                        choices = ['A', 'B', 'C', 'D']
                        output_data.append({
                            "answer": choices[random.randint(0,3)]
                        })
        output_filename = os.path.splitext(filename)[0] + '.json'
        with open(os.path.join(output_folder, output_filename), 'w', encoding='utf-8') as jsonfile:
            json.dump(output_data, jsonfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()