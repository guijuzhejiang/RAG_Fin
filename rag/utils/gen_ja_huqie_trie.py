import math
import re
import string
import datrie
from datetime import datetime


def key_(line):
    return str(line.lower().encode("utf-8"))[2:-1]

def rkey_(line):
    return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]
def loadDict_(fnm):
    try:
        of = open(fnm, "r", encoding='utf-8')
        count = 0
        while True:
            line = of.readline()
            if not line:
                break
            line = re.sub(r"[\r\n]+", "", line)
            line = re.split(r"[ \t]", line)
            k = key_(line[0])
            try:
                value = max(float(line[1]), 1)
                F = int(math.log(value / 1000000) + .5)
            except Exception as e:
                print(e)
            if k not in trie_ or trie_[k][0] < F:
                trie_[key_(line[0])] = (F, line[2])
            trie_[rkey_(line[0])] = 1
            count += 1
            if count % 10000 == 0:
                print(f"[HUQIE]:Build trie count:{count}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trie_.save(fnm + f"_{now}.trie")
        of.close()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    trie_ = datrie.Trie(string.printable)
    loadDict_("rag/res/ja_huqie.txt")