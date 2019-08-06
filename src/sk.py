from util import *
from plots import *

def word_clouds(data_list):
    #convert list of words to string
    positive = " ".join(data_list[0])
    plot_wordCloud(WordCloud().generate(positive))
    negative = " ".join(data_list[1])
    plot_wordCloud(WordCloud().generate(positive))
    neutral = " ".join(data_list[2])
    plot_wordCloud(WordCloud().generate(neutral))
    total = " ".join(data_list[0] + data_list[1] + data_list[2])
    plot_wordCloud(WordCloud().generate(total))
    
def freq_words(data_list, top):
    #returns most frequent words in list
    tweets = data_list[0] + data_list[1] + data_list[2]
    fwords = [word for word, word_count in Counter(tweets).most_common(top)]
    positive_fwords = [word for word, word_count in Counter(data_list[0]).most_common(top)]
    negative_fwords = [word for word, word_count in Counter(data_list[1]).most_common(top)]
    neutral_fwords = [word for word, word_count in Counter(data_list[2]).most_common(top)]
    
    print('Top 10 words from whole set:', fwords)
    print('Top 10 words from positive set:', positive_fwords)
    print('Top 10 words from negative set:', negative_fwords)
    print('Top 10 words from neutral set:', neutral_fwords)

def create_vectors(model2,all_tweets):
    tdict =  get_dict()
    all_vectors = []
    for tw in all_tweets:
        vectors = []
        words = tw.split()
        fscore = 0.0
        tlen = 0.0
        for word in words:
            try:
                word_vector = (model2[word])
            except KeyError:
                word_vector = np.random.rand(1000,)
            #extra features
            np.append(word_vector,tdict.get(word,0))
            tlen = tlen + float(len(word))
            vectors.append(word_vector)
        if (len(vectors) == 0):
            v = np.random.rand(1000,)
        else:
            np.append(vectors,tlen)
            v = sum(vectors)/len(vectors)
        all_vectors.append(v)
        vectors.clear()
    return all_vectors

def clean_data(x):
    x = re.sub(r'https?://[A-Za-z0-9./]+', ' ', x)
    x = re.sub('@[^\s]+', ' ', x)
    x = re.sub(r'#([^\s]+)', r'\1', x)
    x = re.sub(r'\W+', ' ', x)
    x = re.sub(r'[0-9]+', ' ', x)
    return x

def read_data(file):
    #read file and convert it to list of strings
    df = pd.read_csv(file, delimiter='\t', header = None, names = ['id','num','user','tweet'])
    rows = df.shape[0]
    sentiments=np.array(df['user'].values.tolist())
    tweets = np.array(df['tweet'].values.tolist())
    return [df['tweet'],sentiments]

def read_labels(file) :
    df = pd.read_csv(file, delimiter='\t', header = None, names = ['id','sentiment'])
    sentiments=np.array(df['sentiment'].values.tolist())
    return sentiments

def get_dict():
    ndict = make_dict('lexica/affin/affin.txt',{})
    ndict = make_dict('lexica/affin/valence_tweet.txt',ndict)
    ndict = make_dict('lexica/generic/generic.txt',ndict)
    ndict = make_dict('lexica/nrc/val.txt',ndict)
    ndict = make_dict('lexica/nrctag/val.txt',ndict)
    return ndict

def make_dict(file,newDict):
    with open('lexica/nrctag/val.txt', 'r') as f:
        for line in f:
            splitLine = line.split()
            key = newDict.get(str(splitLine[0]), None)
            if key == None:
                newDict[str(splitLine[0])] = splitLine[1]
            else:
                if key < splitLine[1]:
                    newDict[str(splitLine[0])] = splitLine[1]
            
    return newDict

def read_data2():
    #read file and conver it to list of strings
    df = pd.read_csv('twitter_data/train2017.tsv', delimiter='\t', header = None, names = ['id','num','sentiment','tweet'])
    rows = df.shape[0]
    positive_tweets = []
    negative_tweets = []
    neutral_tweets = []
    for i in range(0,rows):
        tweet = preprocess_tweets(df.iloc[i][3])
        if (df.iloc[i][2] == "positive"):
            positive_tweets = positive_tweets + tweet
        elif (df.iloc[i][2] == "negative"):
            negative_tweets = negative_tweets + tweet
        else:
            neutral_tweets = neutral_tweets + tweet
    
    return [positive_tweets,negative_tweets,neutral_tweets]

def preprocess_tweets(data_list) :
    #conver to lower case
    data_list = data_list.lower()
    #remove links
    data_list = re.sub(r'https?://[A-Za-z0-9./]+', ' ', data_list)
    #remove mentions
    data_list = re.sub('@[^\s]+', ' ', data_list)
    #remove hashtags
    data_list = re.sub(r'#([^\s]+)', r'\1', data_list)
    #remove panctuation
    data_list = re.sub(r'\W+', ' ', data_list)
    #remove numbers
    data_list = re.sub(r'[0-9]+', ' ', data_list)

    #tokenize
    stop = set(stopwords.words('english'))
    data = word_tokenize(data_list)

    #remove stop words
    data = [word for word in data if word not in (stop)]

    #stemming
    ps = PorterStemmer()
    data_list = []
    for w in data:
        data_list.append(ps.stem(w))
    return data_list

def read_data3(file):
    df = pd.read_csv(file, delimiter='\t', header = None, names = ['id','num','sentiment','tweet'])
    rows = df.shape[0]
    sentiments=np.array(df['sentiment'].values.tolist())

    tweets = []
    pos_neg=[]
    pos_neut=[]
    neg_neut=[]
    
    pos_neg_np=pd.DataFrame(columns={'tweet':[]})
    pos_neut_np=pd.DataFrame(columns={'tweet':[]})
    neg_neut_np=pd.DataFrame(columns={'tweet':[]})
    
    pos_negL=[]
    pos_neutL=[]
    neg_neutL=[]
    
    for i in range(0,rows):
        tweets = tweets + [df.iloc[i][3]]
        if df.iloc[i][2]=='positive' :
            pos_neg+= [df.iloc[i][3]]
            pos_neut+= [df.iloc[i][3]]
            pos_neg_np=pos_neg_np.append({'tweet' :df.iloc[i][3]},ignore_index=True)
            pos_neut_np=pos_neut_np.append({'tweet' :df.iloc[i][3]},ignore_index=True)
            pos_negL.append("positive")
            pos_neutL.append("positive")
        elif df.iloc[i][2]=='negative' :
            pos_neg+= [df.iloc[i][3]]
            neg_neut+= [df.iloc[i][3]]
            pos_neg_np=pos_neg_np.append({'tweet' :df.iloc[i][3]},ignore_index=True)
            neg_neut_np=neg_neut_np.append({'tweet' :df.iloc[i][3]},ignore_index=True)
            pos_negL.append("negative")
            neg_neutL.append("negative")
        else :
            pos_neut+= [df.iloc[i][3]]
            neg_neut+= [df.iloc[i][3]]
            pos_neut_np=pos_neut_np.append({'tweet' :df.iloc[i][3]},ignore_index=True)
            neg_neut_np=neg_neut_np.append({'tweet' :df.iloc[i][3]},ignore_index=True)
            pos_neutL.append("neutral")
            neg_neutL.append("neutral")
    
    pos_neg_np=pos_neg_np['tweet']
    pos_neut_np=pos_neut_np['tweet']
    neg_neut_np=neg_neut_np['tweet']
    return [tweets,df['tweet'],sentiments,pos_neg,pos_neut,neg_neut,pos_negL,pos_neutL,neg_neutL,pos_neg_np,pos_neut_np,neg_neut_np]


