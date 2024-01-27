import pprint
import numpy as np
import json
import math
import sys
import nltk
from googleapiclient.discovery import build
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def createVectors(q,c,rel,nrel):
    vector={}
    N=len(c)
    vocab=sorted(list(c))
    for r in rel:
        vector[r]=np.zeros(N)
        for i,word in enumerate(vocab):
            vector[r][i]=calcWeight(word,r,c)
    for r in nrel:
        vector[r]=np.zeros(N)
        for i,word in enumerate(vocab):
            vector[r][i]=calcWeight(word,r,c)
    vector['query']=np.zeros(N)
    for i,word in enumerate(vocab): 
            vector['query'][i]=calcWeight(word,'query',c)
    return vector

def calcWeight(word,d,c):
    if d not in c[word]:
        return 0
    tf = c[word][d] #term frequency
    idf=math.log(10*len(c)/len(c[word]))
    return tf*idf

# Initialise score for words in intial query
def queryScore(query):
    score = {}
    words = query.split(' ')
    for word in words:
        score[word] = 1
    return score

def modifyQuery(query,vect,corpus,rel,nrel,score):
    new_query=""
    alpha=0.75
    beta=0.15
    vocab=sorted(list(corpus))
    lenr=len(rel)
    lennr=len(nrel)
    newQuery=vect['query']
    augment = []
    for r in rel:
        newQuery=np.add(newQuery, vect[r]*alpha/lenr) 
    for n in nrel:
        newQuery=np.subtract(newQuery, vect[n]*beta/lennr)
    count=0
    max_args=np.flip(np.argsort(newQuery))
    scores=np.flip(np.sort(newQuery))
    for i in max_args:
        if count==2:
            break
        word=vocab[i]
        if 'query' not in corpus[word]:
            count+=1
            score[word]=scores[i]
            augment.append(word)
        score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
    first_word = True
    for word in score.keys():
        if first_word:
            new_query = new_query + word
            first_word = False
        else:
            new_query = new_query + " " + word
    return new_query, score, augment
ps = PorterStemmer()

def main():
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.

    google_api_key=sys.argv[1]
    search_engine_id=sys.argv[2]
    target=float(sys.argv[3])
    query=sys.argv[4]
    service = build(
        "customsearch", "v1", developerKey=google_api_key
    )
    precision=0
    original_query = query
    score = queryScore(query)
    while precision<target:
        print("Parameters:")
        print("Client key   = ", google_api_key)
        print("Engine key   = ", search_engine_id)
        print("Query        = ", query)
        print("Precision    = ", target)
        print("Google Search Results:")
        print("======================")
        res = (
            service.cse()
            .list(
                q=query,
                cx=search_engine_id,
            )
            .execute()
        )
        if len(res['items']) < 10:
            return
        rel=[]
        nrel=[]
        corpus={}
        i = 0
        for r in res['items']:
            i = i + 1
            print("Result ", i)
            print("[")
            print("URL: ", r['link'])
            print("Title: ",r['title'])
            print("Summary: ",r['snippet'])
            print("]\n")
            # Tokenise
            tokens = word_tokenize(r['snippet']) + word_tokenize(r['title'])
            for t in tokens:
                # Stemming
                s = ps.stem(t)
                # Check if the token is not a number or a symbol
                if s.isalpha() and s not in stop_words:
                    try:
                        corpus[s][r['link']]+=1
                    except:
                        try:
                            corpus[s][r['link']]=1
                        except:
                            corpus[s]={}
            
            val= input('Relevant (Y/N)?')
            # Check if HTML file
            if 'fileFormat' not in r.keys():
                if val=='Y' or val=='y':
                    rel.append(r['link'])
                else:
                    nrel.append(r['link'])
        tokens = word_tokenize(query)
        for t in tokens:
            s=ps.stem(t)
            if s.isalpha():
                    try:
                        corpus[s]['query']+=1
                    except:
                        try:
                            corpus[s]['query']=1
                        except:
                            corpus[s]={}
        precision=len(rel)/(len(rel)+len(nrel))
        vect=createVectors(query,corpus,rel,nrel)
        query, score, augment=modifyQuery(query,vect,corpus,rel,nrel,score)
        print("======================")
        print("FEEDBACK SUMMARY")
        print("Query ", original_query)
        print("Precision ", precision)
        if len(rel) == 0:
            return
        if precision < target:
            print("Still below the desired precision of ", target)
            print("Indexing results ....")
            print("Indexing results ....")
            print("Augmenting by ", augment[0], " ", augment[1])
        else:
            print("Desired precision reached, done")
        print("======================")


if __name__ == "__main__":
    main()
