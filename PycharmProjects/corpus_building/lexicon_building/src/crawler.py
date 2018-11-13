import sys
import requests
import json
import time
from bs4 import BeautifulSoup
header = {
    'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    , 'Accept':
        'text/html,charset=GBK,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
}
bad_review_url = ["https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv863&productId=100000773889&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv2878&productId=100000727128&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv28791&productId=7652139&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv19886&productId=6708229&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv69005&productId=7081550&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv96435&productId=5089253&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv22915&productId=8735304&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv117056&productId=5089225&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                  "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv15037&productId=7479804&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1"]
def review_crawler(url):
    r = requests.get(url, headers=header)
    r.encoding = "gbk"
    bad_reviews = []
    index = r.text.find("{")
    json_data = r.text[index:-2]
    json_data = json.loads(json_data)
    comments = json_data["comments"]
    if len(comments)>0:
        for id in comments:
            review = id["content"]
            review = review.split("\n")
            review = "。".join(review).lstrip("。")
            print(review)
            bad_reviews.append(review)
        return bad_reviews
    else:
        return []

with open("../../data/original_data/bad_3_reviews.txt","w") as f:
    for url in bad_review_url[5:]:
        for i in range(2000):
            bad_review = review_crawler(url.format(i))
            if len(bad_review) > 0:
                for review in bad_review:
                    content = "差评" + "   " + review + "\n"
                    f.write(content)
                time.sleep(10)
            else:
                break
        time.sleep(20)

