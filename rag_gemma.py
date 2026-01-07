from sentence_transformers import SentenceTransformer
import torch


NUM_RELEVANT_CHUNKS = 5


def get_gemma():
    """
    Model is hosted on Hugging Face

    @article{embedding_gemma_2025,
        title={EmbeddingGemma: Powerful and Lightweight Text Representations},
        author={Schechter Vera, Henrique* and Dua, Sahil* and Zhang, Biao and Salz, Daniel and Mullins, Ryan and Raghuram Panyam, Sindhu and Smoot, Sara and Naim, Iftekhar and Zou, Joe and Chen, Feiyang and Cer, Daniel and Lisak, Alice and Choi, Min and Gonzalez, Lucas and Sanseviero, Omar and Cameron, Glenn and Ballantyne, Ian and Black, Kat and Chen, Kaifeng and Wang, Weiyi and Li, Zhe and Martins, Gus and Lee, Jinhyuk and Sherwood, Mark and Ji, Juyeong and Wu, Renjie and Zheng, Jingxiao and Singh, Jyotinder and Sharma, Abheesht and Sreepat, Divya and Jain, Aashi and Elarabawy, Adham and Co, AJ and Doumanoglou, Andreas and Samari, Babak and Hora, Ben and Potetz, Brian and Kim, Dahun and Alfonseca, Enrique and Moiseev, Fedor and Han, Feng and Palma Gomez, Frank and Hernández Ábrego, Gustavo and Zhang, Hesen and Hui, Hui and Han, Jay and Gill, Karan and Chen, Ke and Chen, Koert and Shanbhogue, Madhuri and Boratko, Michael and Suganthan, Paul and Duddu, Sai Meher Karthik and Mariserla, Sandeep and Ariafar, Setareh and Zhang, Shanfeng and Zhang, Shijie and Baumgartner, Simon and Goenka, Sonam and Qiu, Steve and Dabral, Tanmaya and Walker, Trevor and Rao, Vikram and Khawaja, Waleed and Zhou, Wenlei and Ren, Xiaoqi and Xia, Ye and Chen, Yichang and Chen, Yi-Ting and Dong, Zhe and Ding, Zhongli and Visin, Francesco and Liu, Gaël and Zhang, Jiageng and Kenealy, Kathleen and Casbon, Michelle and Kumar, Ravin and Mesnard, Thomas and Gleicher, Zach and Brick, Cormac and Lacombe, Olivier and Roberts, Adam and Sung, Yunhsuan and Hoffmann, Raphael and Warkentin, Tris and Joulin, Armand and Duerig, Tom and Seyedhosseini, Mojtaba},
        publisher={Google DeepMind},
        year={2025},
        url={https://arxiv.org/abs/2509.20354}
    }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    return SentenceTransformer("google/embeddinggemma-300m", device=device), device


def generate_embeddings(chunks):
    """
    Uses Google's Gemma to generate embeddings for a list of chunks

    In the future, I would like to allow users to choose from a variety of models.
    This is good for now - SOTA for its size, pretty good for general local tasks.
    """
    model, device = get_gemma()

    embeddings = model.encode_document(chunks)

    return torch.tensor(embeddings, device=device)


def run_query(query, chunks):
    # In the future, embeddings should be persisted and only generated upon 
    # initial load
    chunk_embeddings = generate_embeddings(chunks)

    # Get an embedding for the query using the same Gemma model
    model, device = get_gemma()
    query_embeddings = model.encode_query(query)
    query_embeddings = torch.tensor(query_embeddings, device=device)

    # Compute similarity between query and chunks and gather most similar chunks
    similarities = model.similarity(query_embeddings, chunk_embeddings)
    similarities_flat = similarities.squeeze(0)
    k = min(NUM_RELEVANT_CHUNKS, len(similarities_flat))
    topk_values, topk_indices = torch.topk(similarities_flat, k)
    topk_text = [chunks[i] for i in topk_indices]

    print(topk_text)
    print(topk_values)
    print(topk_indices)
