# 运行步骤

1. 运行question.py 先把A.csv文件转换成A.json 文件

2. 运行trt-llm 用A.json生成一个output.json文件

3. 运行output.py 把output.json转换成answers.json文件

4. 运行compare.py 比较A.json和answers.json生成结果

mv ./output_json ./convert/output_json
cd convert
python output.py
python compare.py
```