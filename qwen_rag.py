from argparse import ArgumentParser
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


SYSTEM_PROMPT = 'You are a professional Python software engineer performing code reviews. I give you a question about the function and the function itself in Python, as well as three historical questions, code and answers. The code is given in a special form without syntax errors: colons, brackets, commas, and parentheses are omitted. Your task is to understand the logic of the code despite the missing syntax. Respond to each code snippet or question concisely in no more than ten words, one sentence or word. Focus on correctness and efficiency.'


def parse_args() -> ArgumentParser:

    parser = ArgumentParser()
    parser.add_argument(
        'llm_endpoint_url',
        help='LLM endpoint url'
    )

    return parser.parse_args()


def generate_qc_pairs(f1, f2):
    qc_pairs = []
    with open(f1, "r") as file1, open(f2, "r") as file2:
        for q, c in zip(file1, file2):
            qc_pairs.append((q.strip(), c.strip()))
    return qc_pairs


def generate_answers(f1):
    qa_pairs = []
    with open(f1, "r") as file1:
        for a in file1:
            qa_pairs.append(a.strip())
    return qa_pairs


if __name__ == '__main__':
    args = parse_args()
    llm_client = InferenceClient(base_url=args.llm_endpoint_url)

    client = QdrantClient(path="./qdrant")
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = QdrantVectorStore(
        client=client,
        collection_name="disk_collection",
        embedding=model
    )

    answers = generate_answers(
        "./db/train/train.answer")
    qc_pairs = generate_qc_pairs("/home/alina/hermes/db/test/test.question",
                                                            "/home/alina/hermes/db/test/test.code")[6000:]
    docs = [f"Question: {q} Code: {c}" for q, c in qc_pairs]
    qwen_answers = []
    for j in docs:
        query_text = j
        search_result = db.similarity_search_with_score(query=query_text, k=3)
        history_qc = [i.page_content for i, score in search_result]
        history_ans = [answers[i.metadata['_id']]
                       for i, score in search_result]

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': history_qc[0]},
            {'role': 'assistant', 'content': history_ans[0]},
            {'role': 'user', 'content': history_qc[1]},
            {'role': 'assistant', 'content': history_ans[1]},
            {'role': 'user', 'content': history_qc[2]},
            {'role': 'assistant', 'content': history_ans[2]},
            {'role': 'user', 'content': query_text}
        ]

        answer = llm_client.chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=100
        )

        print(answer.choices[0].message.content)
        print('-------')
        qwen_answers.append(answer.choices[0].message.content)
    with open ('qwen_rag.txt', 'a') as f:
        f.write("\n".join(qwen_answers) + "\n")

    


