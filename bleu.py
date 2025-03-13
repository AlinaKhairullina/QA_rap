from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
qwen = pd.DataFrame(columns=['BLEU1', 'BLEU4'])
qwen_rag = pd.DataFrame(columns=['BLEU1', 'BLEU4'])
weights_4 = (0.25, 0.25, 0.25, 0.25)
weights_1 = (1.0, 0.0, 0.0, 0.0)
smooth = SmoothingFunction().method2
with open('qwen.txt', 'r') as f1, open('qwen_rag.txt', 'r') as f2, open('/home/alina/QA_rag/db/test/test.answer', 'r') as f3:
        for q, rag, orig in zip(f1, f2, f3):
                candidate = q.split()
                candidate_rag = rag.split()
                orig = [orig.split()]
                qwen.loc[len(qwen)] = [sentence_bleu(orig, candidate, weights=weights_4, smoothing_function=smooth), sentence_bleu(orig, candidate, weights=weights_1, smoothing_function=smooth)]
                qwen_rag.loc[len(qwen_rag)] = [sentence_bleu(orig, candidate_rag, weights=weights_4, smoothing_function=smooth), sentence_bleu(orig, candidate_rag, weights=weights_1, smoothing_function=smooth)]
      
print("\033[1;34mОценка ответов модели без исторических документов:\033[0m")
print(qwen)
print("\033[1;34mОценка ответов модели с историческими документами:\033[0m")
print(qwen_rag)
print("\033[1;34mСтатистика оценок модели без исторических данных:\033[0m")
print(qwen.describe())
print("\033[1;34mСтатистика оценок модели с историческими данными:\033[0m")
print(qwen_rag.describe())
