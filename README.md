# LLMsQA

ChatGLM6b知识外挂、文档问答。实测可以整合多个文档知识做逻辑问答，案例中根据[‘2023年徐伦工作是算法工程师,徐伦2021年毕业于信息科技大学计算机专业’]两条信息推断出"徐伦工作两年"
- 1、config配置embeding_model与llm_model的存储路径
- 2、main.py指定模型名称、知识路径、问题query
- 3、python main.py

路径相对关系如下：
- main.py
- config.py
- utils.py
- knowledge.txt
- map.txt
- myindex
- embedding_model
- - sentence_transformers
- - - config.json
- - - tokenizer.json
- - - pytorch_model.bin
- - - vocab.json
- llm_model
- - chatglm
- - - pytorch_model-00001-of-00008.bin
- - - ...

# ToDO
受embedding影响极大，直接影响了faiss的搜索结果，后面测试一下针对数据集finetune一下是否会更好。
1、pdf等其它格式文件
2、token超出限制后处理
3、finetune

# 参考
https://github.com/imClumsyPanda/langchain-ChatGLM.git
