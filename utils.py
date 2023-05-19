import re

def split_text(texts):
    texts = re.sub('\n{2,}','\n',texts)
    splited_list = re.split(r'[\.。？\?!！；;\n]', texts)
    if len(splited_list[-1]) != 0:
        return splited_list
    else:
        return splited_list[:-1]

def read_texts(file_path):
    res = ''
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res += line
    return res

def generate_prompt(texts, query, tokens_limit=2048):

    template_size = 120
    context = ''
    for i in texts:
        context += f'{i}\n'
        if len(context) > tokens_limit - template_size - len(query):
            break

    #print_log(context)
    #print_log(query)
    prompt = f'''
要求： 基于已知内容，请用中文以要求的格式简短直接地回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题"。不允许在答案中添加额外成分。

已知内容：

{context[:tokens_limit]}
问题： {query}

格式： 问题:答案

请根据提供的信息回答问题。
'''
    return prompt