from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from typing import Union
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class SemanticSimilarity:
    def __init__(self):
        self.device = 'cpu'  # for GPU usage or "cpu" for CPU usage
        self.checkpoint_110 = 'Salesforce/codet5p-110m-embedding'
        self.tokenizer_110 = AutoTokenizer.from_pretrained(
            self.checkpoint_110, trust_remote_code=True
        )
        self.model_110 = AutoModel.from_pretrained(
            self.checkpoint_110, trust_remote_code=True
        ).to(self.device)

    def _codet5_110_encode(self, comment):
        with torch.no_grad():
            inputs = self.tokenizer_110.encode(
                comment.lower(), return_tensors='pt'
            ).to(self.device)
            embed = self.model_110(inputs)[0]
            return embed.cpu().detach().numpy()

    def evaluate(self, orig: str, pred: str) -> float:
        return cosine_similarity(
            [self._codet5_110_encode(orig)], [self._codet5_110_encode(pred)]
        )[0][0]
    
def calc_test_score(
    train_sample: Union[str, list[str]],
    test_sample: list[Union[str, list[str]]],
) -> float:
    if train_sample == test_sample:
        return 1.0
    if not train_sample or not test_sample:
        return 0.0

    # Use metric BLEU-4 by default, for both train and test short samples
    # will switch to BLEU 1,2,3 according to max length.
    k = min(4, max(len(train_sample), len(test_sample)))
    weights = [1 / k] * k

    chencherry = SmoothingFunction().method1

    return sentence_bleu(
        [train_sample],
        test_sample,
        weights=weights,
        smoothing_function=chencherry,
    )


if __name__ == '__main__':
      

    qwen = pd.DataFrame(columns=['BLEU1', 'BLEU4', 'METEOR', 'ROUGE', 'CodeT5'])
    qwen_rag = pd.DataFrame(columns=['BLEU1', 'BLEU4', 'METEOR', 'ROUGE', 'CodeT5'])

    weights_1 = (1.0, 0.0, 0.0, 0.0)


    chencherry1= SmoothingFunction().method2
    ss = SemanticSimilarity()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


    with open('qwen.txt', 'r') as f1, open('qwen_rag.txt', 'r') as f2, open('/home/alina/QA_rag/db/test/test.answer', 'r') as f3:
            for q, rag, orig in zip(f1, f2, f3):
                    candidate = q.split()
                    candidate_rag = rag.split()
                    orig_answer = [orig.split()]
                    qwen.loc[len(qwen)] = [sentence_bleu(orig_answer, candidate, weights=weights_1, smoothing_function=chencherry1), 
                                        calc_test_score(orig.split(), candidate),
                                        meteor_score(orig_answer, candidate),
                                        scorer.score(orig, q)['rougeL'].fmeasure,
                                        ss.evaluate(orig, q)
                                        ]
                    qwen_rag.loc[len(qwen_rag)] = [sentence_bleu(orig_answer, candidate_rag, weights=weights_1, smoothing_function=chencherry1), 
                                                calc_test_score(orig.split(), candidate_rag),
                                                meteor_score(orig_answer, candidate_rag),
                                                scorer.score(orig, rag)['rougeL'].fmeasure,
                                                ss.evaluate(orig, rag)
                                                ]
    print(qwen)
    print("\033[1;34mОценка ответов модели с историческими документами:\033[0m")
    print(qwen_rag)
    print("\033[1;34mСтатистика оценок модели без исторических данных:\033[0m")
    print(qwen.describe())
    print("\033[1;34mСтатистика оценок модели с историческими данными:\033[0m")
    print(qwen_rag.describe())
