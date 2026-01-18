"""
rag_gemma.py

Author: Benjamin Bremer (benjaminrbremer@gmail.com)

This file is a simple RAG framework based on Google's Gemma. It 
will generate embeddings based on provided document chunks,
perform a similarity search, and generate an answer based on the 
user's query and the most relevant provided text chunks.
"""

from typing import List, Literal

from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import requests
import torch


NUM_RELEVANT_CHUNKS = 5
GEMMA_MODEL_SIZES = ["270m", "1b", "4b", "12b", "27b"]


def get_embed_model():
    """
    Get Google Gemma 300M embedding model. Model is hosted on Hugging Face

    @article{embedding_gemma_2025,
        title={EmbeddingGemma: Powerful and Lightweight Text Representations},
        author={Schechter Vera, Henrique* and Dua, Sahil* and Zhang, Biao and Salz, Daniel and Mullins, Ryan and Raghuram Panyam, Sindhu and Smoot, Sara and Naim, Iftekhar and Zou, Joe and Chen, Feiyang and Cer, Daniel and Lisak, Alice and Choi, Min and Gonzalez, Lucas and Sanseviero, Omar and Cameron, Glenn and Ballantyne, Ian and Black, Kat and Chen, Kaifeng and Wang, Weiyi and Li, Zhe and Martins, Gus and Lee, Jinhyuk and Sherwood, Mark and Ji, Juyeong and Wu, Renjie and Zheng, Jingxiao and Singh, Jyotinder and Sharma, Abheesht and Sreepat, Divya and Jain, Aashi and Elarabawy, Adham and Co, AJ and Doumanoglou, Andreas and Samari, Babak and Hora, Ben and Potetz, Brian and Kim, Dahun and Alfonseca, Enrique and Moiseev, Fedor and Han, Feng and Palma Gomez, Frank and Hernández Ábrego, Gustavo and Zhang, Hesen and Hui, Hui and Han, Jay and Gill, Karan and Chen, Ke and Chen, Koert and Shanbhogue, Madhuri and Boratko, Michael and Suganthan, Paul and Duddu, Sai Meher Karthik and Mariserla, Sandeep and Ariafar, Setareh and Zhang, Shanfeng and Zhang, Shijie and Baumgartner, Simon and Goenka, Sonam and Qiu, Steve and Dabral, Tanmaya and Walker, Trevor and Rao, Vikram and Khawaja, Waleed and Zhou, Wenlei and Ren, Xiaoqi and Xia, Ye and Chen, Yichang and Chen, Yi-Ting and Dong, Zhe and Ding, Zhongli and Visin, Francesco and Liu, Gaël and Zhang, Jiageng and Kenealy, Kathleen and Casbon, Michelle and Kumar, Ravin and Mesnard, Thomas and Gleicher, Zach and Brick, Cormac and Lacombe, Olivier and Roberts, Adam and Sung, Yunhsuan and Hoffmann, Raphael and Warkentin, Tris and Joulin, Armand and Duerig, Tom and Seyedhosseini, Mojtaba},
        publisher={Google DeepMind},
        year={2025},
        url={https://arxiv.org/abs/2509.20354}
    }

    Returns:
        SentenceTransformer object with Gemma embedding model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return SentenceTransformer("google/embeddinggemma-300m", device=device), device


def get_generation_model(model_size: Literal["270m", "1b", "4b", "12b", "27b"]="4b"):
    """
    Get Google Gemma model for generation. Models are hosted on Hugging Face 

    Model size can be specified and should be in ["270m", "1b", "4b", "12b", "27b"].
    For reference, the 4b model maxes out my 3090 with 24gb of VRAM, so it is default.

    @article{gemma_2025,
        title={Gemma 3},
        url={https://goo.gle/Gemma3Report},
        publisher={Kaggle},
        author={Gemma Team},
        year={2025}
    }

    Args:
        model_size (Literal["270m", "1b", "4b", "12b", "27b"]):
            The size of the model to load. These are all of the Gemma 3 Iteration 
            models provided on Hugging Face
    
    Returns:
        The model object, plus a processor object used for prompt templates
    """
    model_id = f"google/gemma-3-{model_size}-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def generate_embeddings(chunks: List[str]):
    """
    Uses Google's Gemma to generate embeddings for a list of chunks

    In the future, I would like to allow users to choose from a variety of models.
    This is good for now - SOTA for its size, pretty good for general local tasks.

    Args:
        chunks (list[str]):
            The list of text chunks that may contain relevant context
    """
    model, device = get_embed_model()

    embeddings = model.encode_document(chunks)

    return torch.tensor(embeddings, device=device)


def _compute_similarities(embeddings: torch.tensor, query: str):
    # Get an embedding for the query using the same Gemma model
    model, device = get_embed_model()
    query_embeddings = model.encode_query(query)
    query_embeddings = torch.tensor(query_embeddings, device=device)

    # Compute similarity between query and chunks and gather most similar chunks
    similarities = model.similarity(query_embeddings, embeddings)
    similarities_flat = similarities.squeeze(0)
    k = min(NUM_RELEVANT_CHUNKS, len(similarities_flat))
    topk_values, topk_indices = torch.topk(similarities_flat, k)
    topk_text = [chunks[i] for i in topk_indices]

    return topk_text


def _augment_query(query: str, topk_text: List[str]):
    # Augment query and generate answer
    aug_query = "Based on the context, answer the users question:\n"
    for i in range(len(topk_text)):
        aug_query += f"{i}: {topk_text[i]}\n"
    aug_query += f"\nUsers Query:\n{query}"
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a research assistant that will be given a question from a user and a list of relevant chunks of text from US patent documents. Please use the provided context to answer research questions from the user."
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": aug_query,
            }]
        }
    ]

    gen_model, processor = get_generation_model()
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    return inputs


def run_query(query: str, chunks: List[str]):
    """
    This function takes a user query, a list of text chunks, and 
    generates an answer based on the most relevant chunks.

    Args:
        query (str):
            The user's question
        chunks (list[str]):
            The list of text chunks that may contain relevant context
    
    Returns:
        str:
            The generated response based on the query and context
    """
    # In the future, embeddings should be persisted and only generated upon 
    # initial load
    chunk_embeddings = generate_embeddings(chunks)

    # Get most relevant text chunks
    topk_text = _compute_similarities(chunk_embeddings, query)

    # Augment query and convert to model inputs
    inputs = _augment_query(query, topk_text)
    
    # Generate response
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = gen_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    
    return decoded
