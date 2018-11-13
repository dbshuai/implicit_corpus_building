import sys
sys.path.append("../")
import yutils.str_utils as utils
import os

def read_files(path):
    sentences = []
    for file in os.listdir(path):
        file = os.path.join(path,file)
        with open(file,"r") as f:
            lines = f.readlines()
            print("reading data from "+file)
            for line in lines:
                if len(line)>2:
                    line = line.strip().lower()
                    info = line.split("   ")
                    if len(info) == 2:
                        sentence = info[1]
                        sentences.append(sentence)
    print(len(sentences))
    return sentences
def main():
    path = "original_data/phone/"
    sentences = read_files(path)
    seg_sentences = utils.segment_str(sentences)
    with open("text8.txt","w") as f:
        for sentence in seg_sentences:
            content = " ".join(sentence).lstrip()+"\n"
            f.write(content)

if __name__ == "__main__":
    main()
