# 伪码 / 示例
from collections import Counter, defaultdict
import spacy
from nltk.corpus import wordnet


nlp = spacy.load("en_core_web_sm")

word_counts = Counter()
pos_counts = defaultdict(Counter)

# 假设你把所有语料拼成一个或多个文本文件，逐行处理
for txtfile in corpus_files:
    with open(txtfile, "r", encoding="utf8") as f:
        for line in f:
            doc = nlp(line.strip())
            for token in doc:
                if token.is_space or token.is_punct or token.like_num:
                    continue
                w = token.lemma_.lower()   # 或 token.text.lower()（是否用 lemma 由你决定）
                pos = token.pos_           # Universal POS (NOUN, VERB, PROPN, ...)
                word_counts[w] += 1
                pos_counts[w][pos] += 1

# 输出 huqie 风格文件：word freq pos
with open("huqie_en.txt", "w", encoding="utf8") as out:
    for w, cnt in word_counts.most_common():
        most_pos = pos_counts[w].most_common(1)[0][0] if pos_counts[w] else "x"
        out.write(f"{w} {cnt} {most_pos}\n")
