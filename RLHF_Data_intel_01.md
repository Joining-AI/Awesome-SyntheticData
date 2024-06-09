# 使用数据表格提示来改善RAG和LLMs的结果
原链接: https://medium.com/intel-tech/tabular-data-rag-llms-improve-results-through-data-table-prompting-bcb42678914b
**作者：Eduardo Rojas Oviedo 和 Ezequiel Lanza**

假设你是一名投资公司的金融分析师。你的工作是了解市场趋势并识别潜在的投资机会。你的客户往往对世界上最富有的人及其财富来源感兴趣。你可能会考虑使用检索增强生成（RAG）系统来快速识别市场趋势、投资机会和经济风险，并回答诸如“哪个行业有最多的亿万富翁？”或“不同地区的亿万富翁性别分布如何？”的问题。

你的第一步是获取这些信息。然而，在初步检查时，你发现一个可能的障碍。文档不仅包含文本，还包含表格！

### LLMs和结构化数据的挑战
对我们人类来说，将文本和表格连接起来很容易。但对LLMs来说，这就像在没有所有拼图块的情况下拼图。

LLMs被设计用于顺序处理文本，逐字逐句地理解信息。然而，表格以行和列的方式组织数据，形成多维结构。理解这种结构需要LLMs以不同于处理顺序文本的方式识别行和列之间的模式、理解不同数据点之间的关系，并解释表头和单元格的含义。

### 常见的表格数据示例
表格数据比我们想象的更常见，以下是三个示例：

**小型数据**：用户可能需要支持包含嵌入小型表格数据的文档查询。一个行业报告是另一个很好的示例，它包含文本分析和销售数据表。

**中型数据**：需要分析更多的表格数据，如电子表格、CSV文件等。例如，你在零售公司分析过去季度的销售数据，这些数据通常以数据库文件的形式出现。

**大型数据**：涉及事务数据库和多维数据集，如OLAP立方体，它们提供快速查询性能、复杂计算支持和跨多个维度的数据切片和切块功能。

### 实战演示
我们将基于一个世界亿万富翁名单的示例，演示如何将表格数据用作用户问题的上下文，并验证是否可能让模型出错。

```python
import os 
import sys
import pandas as pd
from typing import List
from beautifultable import BeautifulTable
import camelot

def get_tables(path: str, pages: List[int]):    
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        if table_list.n>0:
            for tab in range(table_list.n):
                table_df = table_list[tab].df 
                table_df = (
                    table_df.rename(columns=table_df.iloc[0])
                    .drop(table_df.index[0])
                    .reset_index(drop=True)
                )        
                table_df = table_df.apply(lambda x: x.str.replace('\n',''))
                table_df.columns = [col.replace('\n', ' ').replace(' ', '') for col in table_df.columns]
                table_df.columns = [col.replace('(', '').replace(')', '') for col in table_df.columns]
    return table_df

file_path="./World_Billionaires_Wikipedia.pdf"
df = get_tables(file_path, pages=[3])
```

### 数据格式转换
将表格数据转换为多种格式：

```python
eval_df = pd.DataFrame(columns=["Data Format", "Data raw"])

data_json = df.to_json(orient='records')
eval_df.loc[len(eval_df)] = ["JSON", data_json]

data_list_dict = df.to_dict(orient='records')
eval_df.loc[len(eval_df)] = ["DICT", data_list_dict]

csv_data = df.to_csv(index=False)
eval_df.loc[len(eval_df)] = ["CSV", csv_data]

tsv_data = df.to_csv(index=False, sep='\t')
eval_df.loc[len(eval_df)] = ["TSV (tab-separated)", tsv_data]

html_data = df.to_html(index=False)
eval_df.loc[len(eval_df)] = ["HTML", html_data]

latex_data = df.to_latex(index=False)
eval_df.loc[len(eval_df)] = ["LaTeX", latex_data]

markdown_data = df.to_markdown(index=False)
eval_df.loc[len(eval_df)] = ["Markdown", markdown_data]

string_data = df.to_string(index=False)
eval_df.loc[len(eval_df)] = ["STRING", string_data]

numpy_data = df.to_numpy()
eval_df.loc[len(eval_df)] = ["NumPy", numpy_data]

xml_data = df.to_xml(index=False)
eval_df.loc[len(eval_df)] = ["XML", xml_data]
```

### 模型连接和测试
设置基本提示和模型连接设置，并运行测试：

```python
from openai import AzureOpenAI

MESSAGE_SYSTEM_CONTENT = """You are a customer service agent that helps a customer with answering questions. 
Please answer the question based on the provided context below. 
Make sure not to make any changes to the context, if possible, when preparing answers to provide accurate responses. 
If the answer cannot be found in context, just politely say that you do not know, do not try to make up an answer."""

client = AzureOpenAI(
    api_key=OAI_API_Key, 
    api_version=OAI_API_Version, 
    azure_endpoint=OAI_API_Base)

def response_test(question:str, context:str, model:str = "gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": MESSAGE_SYSTEM_CONTENT,
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": context},
        ],
    )
    return response.choices[0].message.content

def run_question_test(query: str, eval_df:str):
    questions = []
    answers = []
    for index, row in eval_df.iterrows():
        questions.append(query)
        response = response_test(query, str(row['Data raw']))
        answers.append(response)
    eval_df['Question'] = questions
    eval_df['Answer'] = answers
    return eval_df

def BeautifulTableformat(query:str, results:pd.DataFrame, MaxWidth:int = 250):
    table = BeautifulTable(maxwidth=MaxWidth, default_alignment=BeautifulTable.ALIGN_LEFT)
    table.columns.header = ["Data Format", "Query", "Answer"]
    for index, row in results.iterrows():
        table.rows.append([row['Data Format'], query, row['Answer']])
    return table

query = "What's the Elon Musk's net worth?"
result_df1 = run_question_test(query, eval_df.copy())
table = BeautifulTableformat(query, result_df1, 150)
print(table)
```

### 测试和验证
通过设置不同的问题来测试模型的回答：

```python
query = "What's the sixth richest billionaire in 2023 net worth?"
result_df2 = run_question_test(query, eval_df.copy())
table = BeautifulTableformat(query, result_df2, 150)
print(table)

query = "What's the Michael Jordan net worth?"
result_df3 = run_question_test(query, eval_df.copy())
table = BeautifulTableformat(query, result_df3, 150)
print(table)

query = "What's Michael Musk's net worth?"
result_df4 = run_question_test(query, eval_df.copy())
table = BeautifulTableformat(query, result_df4, 180)
print(table)
```

### 结论
通过嵌入小型表格数据，我们看到LLM可以理解表格的上下文，即使在错误信息的情况下也能正确回答。保持结构完整性和上下文信息是关键。

若想深入探索，可以尝试使用Tabula、pdfplumber等其他表格提取库。
