import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from collections import Counter

from collections import OrderedDict
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from html.parser import HTMLParser
from bs4 import BeautifulSoup

porter = PorterStemmer()
wnl = WordNetLemmatizer()
stop = stopwords.words('english')
stop.append("new")
stop.append("like")
stop.append("u")
stop.append("it'")
stop.append("'s")
stop.append("n't")
stop.append('mr.')
stop = set(stop)

# taken from http://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html

def tokenizer(text):

    tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]

    tokens = []
    for token_by_sent in tokens_:
        tokens += token_by_sent

    tokens = list(filter(lambda t: t.lower() not in stop, tokens))
    tokens = list(filter(lambda t: t not in punctuation, tokens))
    tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', u'\u2014', u'\u2026', u'\u2013'], tokens))

    filtered_tokens = []
    for token in tokens:
        token = wnl.lemmatize(token)
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

    return filtered_tokens


class MLStripper(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def get_keywords(tokens, num):
    return Counter(tokens).most_common(num)


def build_article_df(urls):
    articles = []
    for index, row in urls.iterrows():
        try:
            data=row['text'].strip().replace("'", "")
            data = strip_tags(data) #delete html tags
            soup = BeautifulSoup(data, features="html.parser")
            data = soup.get_text() #extracting all the text from row['text']
            data = data.encode('ascii', 'ignore').decode('ascii') #ignore errors(delete characters misunderstood)
            document = tokenizer(data) #we obtain a list with words lower case, without stop words, without numbers
            top_5 = get_keywords(document, 3) #top 3 words in article

            unzipped = list(zip(*top_5)) #  * is the 'splat' operator. It is used for unpacking a list into arguments. For example: foo(*[1, 2, 3]) is the same as foo(1, 2, 3)
            #s = [('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631),
 #('you', 554),  ('a', 546), ('my', 514), ('hamlet', 471), ('in', 451)]
            #print(list(zip(*s))):
            #[('the', 'and', 'to', 'of', 'i', 'you', 'a', 'my', 'hamlet', 'in'), (1143, 966, 762, 669, 631, 554, 546, 514, 471, 451)]
            kw= list(unzipped[0])
            #['the', 'and', 'to', 'of', 'i', 'you', 'a', 'my', 'hamlet', 'in']
            kw=",".join(str(x) for x in kw)
            articles.append((kw, row['title'], row['pubdate']))
        except Exception as e:
            print(e)
            #print data
            #break
            pass
        #break
    article_df = pd.DataFrame(articles, columns=['keywords', 'title', 'pubdate'])
    return article_df


#df = pd.read_csv('../examples/tocsv.csv')
df = pd.read_csv('tocsv.csv')
df.head()



#df = pd.read_csv('tocsv.csv')
data = []
for index, row in df.iterrows():
    data.append((row['Title'], row['Permalink'], row['Date'], row['Content']))
data_df = pd.DataFrame(data, columns=['title' ,'url', 'pubdate', 'text' ])


data_df.tail()
article_df = build_article_df(data_df)
article_df.head()
#building co-occurance matrix. To get there, we need to take a few steps
# to get our keywords broken up individually
keywords_array=[]
for index, row in article_df.iterrows():
    keywords=row['keywords'].split(',')
    for kw in keywords:
        keywords_array.append((kw.strip(' '), row['keywords']))

kw_df = pd.DataFrame(keywords_array).rename(columns={0:'keyword', 1:'keywords'})

kw_df.head()

document = kw_df.keywords.tolist()
names = kw_df.keyword.tolist()

document_array = []
for item in document:
    items = item.split(',')
    document_array.append((items))
# OrderedDict keeps track of order of insertion
occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)

# Find the co-occurrences:
for l in document_array:
    for i in range(len(l)):
        for item in l[:i] + l[i + 1:]:
            occurrences[l[i]][item] += 1

co_occur = pd.DataFrame.from_dict(occurrences )
co_occur.to_csv('out.csv')
co_occur.head()
