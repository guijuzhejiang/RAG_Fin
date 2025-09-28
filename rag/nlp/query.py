#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import math
import re
import logging
import copy
from elasticsearch_dsl import Q

from rag.nlp import rag_tokenizer, term_weight, synonym

class EsQueryer:
    def __init__(self, es):
        self.tw = term_weight.Dealer()
        self.es = es
        self.syn = synonym.Dealer()
        self.flds = ["ask_tks^10", "ask_small_tks"]

    @staticmethod
    def subSpecialChar(line):
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()

    @staticmethod
    def isChinese(line):
        """
        如果包含任何 CJK（中日韩）字符则认为是中文/日文/韩文文本，否则视为非中文（例如英文）。
        """
        # CJK Unified Ideographs + Hiragana + Katakana + Fullwidth forms 等范围
        return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\u3000-\u303f\uff00-\uffef]', line))

    @staticmethod
    def rmWWW(txt):
        # 1) 先移除词尾的 's 和 're（例如 AOC's -> AOC, they're -> they）
        txt = re.sub(r"(?<=\w)(?:'s|'re)\b", "", txt, flags=re.IGNORECASE)

        # 如果包含 CJK，则使用原来的中/日混合替换（保留你原来的规则）
        if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uff00-\uffef]', txt):
            patts = [
                (
                r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀)是*",
                ""),
                (
                r"(何時|何処|何故|どう|なぜ|どうして|何|どこ|どの|誰|どれ|いつ|どうやって|どちら|なに|どんな|だれ|だろう|でしょう|なあ|かな|いかが|ですか|ますか|でしょうか|ありますか|かね|教えてください)",
                "")
            ]
            for r, p in patts:
                txt = re.sub(r, p, txt, flags=re.IGNORECASE)
            return txt

        # 英文分支：按单词边界替换常见的问句/填充词（注意不要乱删单词内部）
        eng_patts = [
            (r"(^|\s)(what|who|how|which|where|why|when|whom|whose)(\'re|\'s)?\b", " "),
            # 常见虚词/礼貌用语（以空格或句首为界） — 只列常用的，避免过度清洗
            (
            r"(^|\s)(please|just|may|could|would|should|can|i|you|we|they|me|your|our|my|mine|yours|thanks|thank you)\b",
            " "),
            # 一些问句尾巴/语气词
            (r"(\b)(please\?|please\!|please\b)", " "),
            # 去掉单独的助动词等（按需增减）
            (r"(^|\s)(is|are|was|were|do|does|did|have|has|had|be|am|there)\b", " "),
        ]
        for r, p in eng_patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)

        # 额外清理多余空格
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def question(self, txt, tbl="qa", min_match="60%"):
        if len(txt) != 0:
            txt = re.sub(
                r"[ :\r\n\t,，。？\?/`!！&\^%%・、]+",
                " ",
                rag_tokenizer.tradi2simp(
                    rag_tokenizer.strQ2B(
                        txt.lower()))).strip()
            txt = EsQueryer.rmWWW(txt)

            if not self.isChinese(txt):
                tks = rag_tokenizer.tokenize(txt).split(" ")
                tks_w = self.tw.weights(tks)
                tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
                tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk]
                tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
                q = ["{}^{:.4f}".format(tk, w) for tk, w in tks_w if tk]
                for i in range(1, len(tks_w)):
                    q.append("\"%s %s\"^%.4f" % (tks_w[i - 1][0], tks_w[i][0], max(tks_w[i - 1][1], tks_w[i][1])*2))
                if not q:
                    q.append(txt)
                return Q("bool",
                         must=Q("query_string", fields=self.flds,
                                type="best_fields", query=" ".join(q),
                                boost=1)#, minimum_should_match=min_match)
                         ), list(set([t for t in txt.split(" ") if t]))

        def need_fine_grained_tokenize(tk):
            if len(tk) < 3:
                return False
            if re.match(r"^[0-9A-Za-z\.\+#_\*-]+$", tk):
                return False
            return True

        qs, keywords = [], []
        for tt in self.tw.split(txt)[:256]:  # .split(" "):
            if not tt:
                continue
            # keywords.append(tt)
            twts = self.tw.weights([tt])
            syns = self.syn.lookup(tt)
            if syns: keywords.extend(syns)
            logging.info(json.dumps(twts, ensure_ascii=False))
            tms = []
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                sm = rag_tokenizer.fine_grained_tokenize(tk).split(" ") if need_fine_grained_tokenize(tk) else []
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-・]+",
                        "",
                        m) for m in sm]
                sm = [EsQueryer.subSpecialChar(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]
                if len(sm) < 2:
                    sm = []

                keywords.append(re.sub(r"[ \\\"']+", "", tk))
                keywords.extend(sm)

                tk_syns = self.syn.lookup(tk)
                tk = EsQueryer.subSpecialChar(tk)
                if tk.find(" ") > 0:
                    tk = "\"%s\"" % tk
                if tk_syns:
                    tk = f"({tk} %s)" % " ".join(tk_syns)
                if sm:
                    tk = f"{tk} OR \"%s\" OR (\"%s\"~2)^0.5" % (
                        " ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

                if len(keywords) >= 12: break

            tms = " ".join([f"({t})^{w}" for t, w in tms])

            if len(twts) > 1:
                tms += f" (\"%s\"~4)^1.5" % (" ".join([t for t, _ in twts]))
            if re.match(r"[0-9a-z ]+$", tt):
                tms = f"(\"{tt}\" OR \"%s\")" % rag_tokenizer.tokenize(tt)

            syns = " OR ".join(
                ["\"%s\"^0.7" % EsQueryer.subSpecialChar(rag_tokenizer.tokenize(s)) for s in syns])
            if syns:
                tms = f"({tms})^5 OR ({syns})^0.7"

            qs.append(tms)

        flds = copy.deepcopy(self.flds)
        mst = []
        if qs:
            mst.append(
                Q("query_string", fields=flds, type="best_fields",
                  query=" OR ".join([f"({t})" for t in qs if t]), boost=1, minimum_should_match=min_match)
            )

        return Q("bool",
                 must=mst,
                 ), list(set(keywords))

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3,
                          vtweight=0.7):
        from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
        import numpy as np
        sims = CosineSimilarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        return np.array(sims[0]) * vtweight + \
            np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        def toDict(tks):
            d = {}
            if isinstance(tks, str):
                tks = tks.split(" ")
            for t, c in self.tw.weights(tks):
                if t not in d:
                    d[t] = 0
                d[t] += c
            return d

        atks = toDict(atks)
        btkss = [toDict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt))}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt))}
        s = 1e-9
        for k, v in qtwt.items():
            # if k in dtwt:
            if any(k in key for key in dtwt):
                s += v  # * dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v  # * v
        #d = 1e-9
        # for k, v in dtwt.items():
        #    d += v * v
        return s / q / max(1, math.sqrt(math.log10(max(len(qtwt.keys()), len(dtwt.keys())))))# math.sqrt(q) / math.sqrt(d)
