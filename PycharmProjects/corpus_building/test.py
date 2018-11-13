if __name__ == "__main__":
    emb_dir = "/users4/gzluo/dataset/embedding/Tencent_AILab_ChineseEmbedding.txt"
    with open(emb_dir,"r") as f:
        for i,c in enumerate(f):
            if i<10:
                print(c)
