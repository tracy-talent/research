基于tensorflow实现的ELMo，代码源自[allenai/bilm-tf](https://github.com/allenai/bilm-tf)
model.py包含整个模型的结构设计:
1. 多kernel卷积拼接的character embedding 
2. 2 layer highway network
3. 2 layer BiLSTM
