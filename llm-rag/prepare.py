
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat, GLM4Chat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding

## 文档分块
docs = ReadFiles('data/fuzzy').get_content(min_token_len=600, max_token_len=1800, cover_content=150) # 获得data目录下的所有文件内容并分割
print(len(docs))        # chunks个数 (28)
# print(len(docs[0]))   # 第一个chunk长度

## 求词嵌入
vector = VectorStore(docs)
embedding = ZhipuEmbedding()    # 用智谱的词嵌入模型 (embedding-2)
docEmbed = vector.get_vector(EmbeddingModel=embedding)
print(len(docEmbed))    # 词嵌入后chunk个数 (28)
print(len(docEmbed[0])) # 词嵌入每一个向量维度 (embedding_dim=1024)

# vector.persist(path='storage/fuzzy') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库