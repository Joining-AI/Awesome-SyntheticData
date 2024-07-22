# 四种数据清理技术以提高大型语言模型（LLM）的性能

[Intel tech原链接](https://medium.com/intel-tech/four-data-cleaning-techniques-to-improve-large-language-model-llm-performance-77bee9003625)

**作者：Eduardo Rojas Oviedo 和 Ezequiel Lanza**

检索增强生成（RAG）过程由于其能够增强大型语言模型（LLM）的理解能力、提供上下文并帮助防止幻觉而备受关注。RAG过程包括几个步骤，从将文档分块到提取上下文，再到使用该上下文提示LLM模型。尽管RAG已知能显著改善预测，但偶尔也会导致不正确的结果。文档的摄取方式在这一过程中起着至关重要的作用。例如，如果我们的“上下文文档”包含拼写错误或LLM难以理解的字符（如表情符号），可能会干扰LLM对所提供上下文的理解。

在本文中，我们将演示在将文本分块摄取并进一步处理之前，使用四种常见的自然语言处理（NLP）技术来清理文本。我们还将展示这些技术如何显著增强模型对提示的响应。

## 为什么清理文档很重要？
在将文本输入任何类型的机器学习算法之前，清理文本是标准做法。无论是使用监督或无监督算法，还是为生成式AI（GAI）模型创建上下文，整理好文本有助于：

- **确保准确性**：通过消除错误和保持一致性，可以减少混淆模型或产生幻觉的可能性。
- **提高质量**：更干净的数据可确保模型处理可靠且一致的信息，有助于模型从准确数据中推断。
- **便于分析**：干净的数据更容易解释和分析。例如，使用纯文本训练的模型可能难以理解表格数据。

通过清理我们的数据——尤其是非结构化数据——我们为模型提供了可靠和相关的上下文，这改善了生成效果，降低了幻觉的概率，并提高了GAI的速度和性能，因为大量信息会导致更长的等待时间。

## 我们如何实现数据清理？
为了帮助你构建数据清理工具箱，我们将探讨四种NLP技术及其如何帮助模型。

### 步骤1：数据清理和降噪
我们将首先去除不提供意义的符号或字符，如HTML标签、XML解析、JSON、表情符号和标签。多余的字符往往会混淆模型，并增加上下文标记的数量，从而增加计算成本。

### 技术：
- **分词**：将文本拆分成单个单词或标记。
- **去噪**：去除不需要的符号、表情符号、标签和Unicode字符。
- **归一化**：将文本转换为小写以保持一致性。
- **去除停用词**：丢弃不增加意义的常见或重复的词，如“a”，“in”，“of”和“the”。
- **词形还原或词干提取**：将单词还原为其基本形式或词根。

让我们以这条推文为例：

“I love coding! 😊 #PythonProgramming is fun! 🐍✨ Let’s clean some text 🧹”

虽然我们能清楚理解其意思，但让我们通过应用常见的技术简化它以便于模型理解。

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 样本文本带有表情符号、标签和其他字符
text = "I love coding! 😊 #PythonProgramming is fun! 🐍✨ Let’s clean some text 🧹"

# 分词
tokens = word_tokenize(text)

# 去噪
cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]

# 归一化（转换为小写）
cleaned_tokens = [token.lower() for token in cleaned_tokens]

# 去除停用词
stop_words = set(stopwords.words('english'))
cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]

# 词形还原
lemmatizer = WordNetLemmatizer()
cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

print(cleaned_tokens)
```
输出：
```
['love', 'coding', 'pythonprogramming', 'fun', 'clean', 'text']
```
这个过程去除了无关的字符，留下了模型可以理解的干净且有意义的文本。

### 步骤2：文本标准化和归一化
接下来，我们应始终优先考虑文本的一致性和连贯性。这对于确保准确的检索和生成至关重要。以下Python示例展示了如何扫描文本输入中的拼写错误和其他可能导致不准确和性能下降的不一致。

```python
import re

# 样本文本带有拼写错误
text_with_errors = """
But 's not  oherence  about more language  oherence .
Other important aspect is ensuring accurte retrievel by  oherence  product name spellings.
Additionally, refning descriptions  oherenc the  oherence of the contnt.
"""

# 修正拼写错误的函数
def correct_spelling_errors(text):
    # 定义常见拼写错误及其修正的字典
    spelling_corrections = {
        " oherence ": "everything",
        " oherence ": "refinement",
        "accurte": "accurate",
        "retrievel": "retrieval",
        " oherence ": "correcting",
        "refning": "refining",
        " oherenc": "enhances",
        " oherence": "coherence",
        "contnt": "content",
    }

    # 遍历字典中的每对键值，并用正确版本替换拼写错误的单词
    for mistake, correction in spelling_corrections.items():
        text = re.sub(mistake, correction, text)

    return text

# 修正样本文本中的拼写错误
cleaned_text = correct_spelling_errors(text_with_errors)

print(cleaned_text)
```
输出：
```
But it's not everything about more language refinement.
Other important aspect is ensuring accurate retrieval by correcting product name spellings.
Additionally, refining descriptions enhances the coherence of the content.
```

### 步骤3：元数据处理
元数据收集，如识别重要的关键词和实体，使我们能够识别文本中的元素，这些元素可以用于改善语义搜索结果，特别是在内容推荐系统等企业应用中。这一过程为模型提供了额外的上下文，通常需要以提高RAG性能。让我们将此步骤应用于另一个Python示例。

```python
import spacy
import json

# 加载英文语言模型
nlp = spacy.load("en_core_web_sm")

# 样本文本带有元数据候选项
text = """
In a blog post titled 'The Top 10 Tech Trends of 2024,' 
John Doe discusses the rise of artificial intelligence and machine learning 
in various industries. The article mentions companies like Google and Microsoft 
as pioneers in AI research. Additionally, it highlights emerging technologies 
such as natural language processing and computer vision.
"""

# 使用spaCy处理文本
doc = nlp(text)

# 提取命名实体及其标签
meta_data = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

# 将元数据转换为JSON格式
meta_data_json = json.dumps(meta_data)

print(meta_data_json)
```
输出：
```
[
    {"text": "2024", "label": "DATE"},
    {"text": "John Doe", "label": "PERSON"},
    {"text": "Google", "label": "ORG"},
    {"text": "Microsoft", "label": "ORG"},
    {"text": "AI", "label": "ORG"},
    {"text": "natural language processing", "label": "ORG"},
    {"text": "computer vision", "label": "ORG"}
]
```

### 步骤4：上下文信息处理
在处理LLMs时，你可能会经常处理多种语言或管理大量充满各种主题的文档，这对模型来说可能很难理解。让我们看看两种可以帮助模型更好理解数据的技术。

首先是语言翻译。使用Google翻译API，代码将原始文本“Hello, how are you?”从英语翻译成西班牙语。

```python
from googletrans import Translator

# 原始文本
text = "Hello, how are you?"

# 翻译文本
translator = Translator()
translated_text = translator.translate(text, src='en', dest='es').text

print("Original Text:", text)
print("Translated Text:", translated_text)
```

接下来是主题建模，包括数据聚类技术，帮助你的模型识别文档的主题并快速处理大量信息。潜在狄利克雷分配（LDA）是最流行的自动化主题建模技术，它通过仔细观察单词模式来帮助发现文本中的隐藏主题。

以下示例中，我们将使用sklearn处理一组文档并识别关键主题。

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 样本文档
documents = [
    "Machine

 learning is a subset of artificial intelligence.",
    "Natural language processing involves analyzing and understanding human languages.",
    "Deep learning algorithms mimic the structure and function of the human brain.",
    "Sentiment analysis aims to determine the emotional tone of a text."
]

# 将文本转换为数值特征向量
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# 应用潜在狄利克雷分配（LDA）进行主题建模
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# 显示主题
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-5 - 1:-1]]))
```
输出：
```
Topic 1:
learning machine subset artificial intelligence
Topic 2:
processing natural language involves analyzing understanding
```

如果你想探索更多主题建模技术，推荐从以下几种开始：

- **非负矩阵分解（NMF）**：适用于图像等负值没有意义的情况。在图像处理等需要清晰理解因素的任务中很有用。
- **潜在语义分析（LSA）**：在需要处理大规模文本和识别单词与文档之间关系时非常出色。通过奇异值分解（SVD）识别语义关系。
- **层次狄利克雷过程（HDP）**：在不确定有多少主题时，帮助快速分类和识别文档中的主题。HDP作为LDA的扩展，允许无限主题。
- **概率潜在语义分析（PLSA）**：帮助计算文档关于某些主题的概率，在构建基于过去交互的个性化推荐系统时很有用。

## DEMO：清理GAI文本输入
让我们通过一个示例将所有步骤结合起来。在此演示中，我们使用ChatGPT生成了两个技术专家之间的对话。我们将应用基本的清理技术，以展示这些实践如何确保可靠和一致的结果。

```python
synthetic_text = """
Sarah (S): Technology Enthusiast
Mark (M): AI Expert
S: Hey Mark! How's it going? Heard about the latest advancements in Generative AI (GA)?
M: Hey Sarah! Yes, I've been diving deep into the realm of GA lately. It's fascinating how it's shaping the future of technology!
S: Absolutely! I mean, GA has been making waves across various industries. What do you think is driving its significance?
M: Well, GA, especially Retrieval Augmented Generative (RAG), is revolutionizing content generation. It's not just about regurgitating information anymore; it's about creating contextually relevant and engaging content.
S: Right! And with Machine Learning (ML) becoming more sophisticated, the possibilities seem endless.
M: Exactly! With advancements in ML algorithms like GPT (Generative Pre-trained Transformer), we're seeing unprecedented levels of creativity in AI-generated content.
S: But what about concerns regarding bias and ethics in GA?
M: Ah, the age-old question! While it's true that GA can inadvertently perpetuate biases present in the training data, there are techniques like Adversarial Training (AT) that aim to mitigate such issues.
S: Interesting! So, where do you see GA headed in the next few years?
M: Well, I believe we'll witness a surge in applications leveraging GA for personalized experiences. From virtual assistants to content creation tools, GA will become ubiquitous in our daily lives.
S: That's exciting! Imagine AI-powered virtual companions tailored to our preferences.
M: Indeed! And with advancements in Natural Language Processing (NLP) and computer vision, these virtual companions will be more intuitive and lifelike than ever before.
S: I can't wait to see what the future holds!
M: Agreed! It's an exciting time to be in the field of AI.
S: Absolutely! Thanks for sharing your insights, Mark.
M: Anytime, Sarah. Let's keep pushing the boundaries of Generative AI together!
S: Definitely! Catch you later, Mark!
M: Take care, Sarah!
"""
```

### 步骤1：基本清理
首先，让我们移除对话中的表情符号、标签和Unicode字符。

```python
# 样本文本带有表情符号、标签和其他字符

# 分词
tokens = word_tokenize(synthetic_text)

# 去噪
cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]

# 归一化（转换为小写）
cleaned_tokens = [token.lower() for token in cleaned_tokens]

# 去除停用词
stop_words = set(stopwords.words('english'))
cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]

# 词形还原
lemmatizer = WordNetLemmatizer()
cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

print(cleaned_tokens)
```

### 步骤2：准备提示
接下来，我们将构建一个提示，要求模型根据从我们的合成对话中获取的信息，作为友好的客户服务代理进行响应。

```python
MESSAGE_SYSTEM_CONTENT = """
You are a customer service agent that helps a customer with answering questions. 
Please answer the question based on the provided context below. 
Make sure not to make any changes to the context if possible, when prepare answers so as to provide accurate responses. 
If the answer cannot be found in context, just politely say that you do not know, do not try to make up an answer.
"""
```

### 步骤3：准备交互
让我们准备与模型的交互。在这个示例中，我们将使用GPT-4。

```python
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
```

### 步骤4：准备问题
最后，让我们问模型一个问题，并比较清理前后的结果。

```python
question1 = "What are some specific techniques in Adversarial Training (AT) that can help mitigate biases in Generative AI models?"
```

清理前，我们的模型生成了这个响应：

```python
response = response_test(question1, synthetic_text)
print(response)

#输出：
# I'm sorry, but the context provided doesn't contain specific techniques in Adversarial Training (AT) that can help mitigate biases in Generative AI models.
```

清理后，模型生成了以下响应。通过基本清理技术增强理解，模型可以提供更全面的答案。

```python
response = response_test(question1, new_content_string)
print(response)

#输出：
# The context mentions Adversarial Training (AT) as a technique that can 
# help mitigate biases in Generative AI models. However, it does not provide 
# any specific techniques within Adversarial Training itself.
```

## 更光明的AI生成结果的未来
RAG模型提供了多个优势，通过提供相关的上下文，显著增强了AI生成结果的可靠性和连贯性。这种上下文化显著提高了AI生成内容的准确性。

为了充分利用RAG模型，稳健的数据清理技术在文档摄取过程中至关重要。这些技术解决了文本数据中的差异、不精确术语和其他潜在错误，显著提高了输入数据的质量。当操作在更干净、更可靠的数据上时，RAG模型会提供更准确和有意义的结果，从而在各个领域中实现更好的决策和问题解决能力。
