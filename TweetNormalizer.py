from emoji import demojize
from nltk.tokenize import TweetTokenizer
import re
import functools
import html 

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


def preprocess_clean(tweet):
    '''
    进一步的预处理
    英文步骤包括：字符替换、空白字符删除、标点符号清洗；
    社交媒体特殊处理：@username -> '@USER' http链接 -> 'HTTPURL'
    '''
    # 建立替换表
    with open('change.txt') as f:
        mapping = {}
        for line in f:
            chars = line.strip('\n').split('\t')
            mapping[chars[0]] = chars[1]
        # 替换回车键至换行键
        mapping["\u000D"] = "\u000A"
        mapping["\u2028"] = "\u000A"
        mapping["\u2029"] = "\u000A"
        # 替换\t至空格
        mapping["\u0009"] = "\u0020"
    '''
    替换特殊字符以及删除不可见字符
    ''' 
    char_list = list(map(lambda x:mapping.get(x, x), tweet))
    for id_, x in enumerate(char_list):
        # 删除不可见字符 
        if "\u2000" <= x <= "\u200F" or "\u0000" <= x <= "\u001F" and x != "\n":
            char_list[id_]=''
    
    # 替换特殊字符 
    tweet = ''.join(char_list)

    def repl1(matchobj):
        return matchobj.group(0)[0]
    def repl2(matchobj):
        return matchobj.group(0)[-1]
    # 去掉段首尾换行
    tweet = tweet.strip()
    # 标点重复
    tweet=re.sub(r'([（《【‘“\(\<\[\{）》】’”\)\>\]\} ,;:·；：、，。])\1+',repl1,tweet)
    # 括号紧跟标点
    tweet=re.sub(r'[（《【‘“\(\<\[\{][ ,.;:；：、，。！？·]',repl1,tweet)
    tweet=re.sub(r'[ ,.;:；：、，。！？·][）》】\)\>\]\}]',repl2,tweet)
    # 括号内为空
    tweet=re.sub(r'([（《【‘“\(\<\[\{\'\"][\'\"）》】’”\)\>\]\}])','',tweet)
    # 三个。和.以上的转为...
    tweet = re.sub(r'[。.]{3,}', '...', tweet)
    # HTML网址清洗和username清洗
    tweet = re.sub("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "HTTPURL", tweet)      # url
    tweet = re.sub("@\S+", "@USER", tweet)      # username
    # html_entities 转译
    tweet = html.unescape(tweet)
    return tweet


if __name__ == "__main__":
    print(
        preprocess_clean(
            "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"
        )
    )