import os
import json
import csv
input_folder = 'answer'
true_answer_folder = 'test_json'
csv_folder = 'test_csv'
suma = 0
i = 0
result=[]
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_name_without_extension = os.path.splitext(filename)[0]
        answers = []
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
            answers_data = json.load(file)
            for ans in answers_data:
                answers.append(ans["answer"])

        A_answers = []
        
        with open(os.path.join(csv_folder, file_name_without_extension+".csv"), newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                A_answers.append(row["Answer"])

        
        # answers = [item["answer"] for item in answers_data]
        # A_answers = [prompt["answer"] for prompt in A_data["Prompts"]]
        total = len(answers)
        
        
        correct_count = sum([1 for ans1, ans2 in zip(answers, A_answers[3:]) if ans1 == ans2])
        accuracy = correct_count / total if total > 0 else 0

        # result += ";"+"{accuracy:.4f }"
        print(f"{accuracy:.4f}", end="; ")

        # print(result)
        suma = accuracy+suma
        i += 1
print(suma/i)