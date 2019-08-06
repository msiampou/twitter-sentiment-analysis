from plots import *
from sk import *

def word_embeddings(all_tweets,labels,vocab,valid_labels):

    clean_tweet = all_tweets.apply(lambda x: clean_data(x))

    if os.path.isfile('we_model.pkl') :
        infile = open("we_model.pkl",'rb')
        model = pickle.load(infile)
        infile.close()
    else:
        model = gensim.models.Word2Vec(clean_tweet, size=1000, window=5, min_count=2, sg=1, hs=0, negative=10, workers=2, seed=34)
        model.train(clean_tweet, total_examples=len(clean_tweet), epochs=20)
        outfile = open("we_model.pkl",'wb')
        pickle.dump(model,outfile)    
        outfile.close()  
    
    if os.path.isfile('we_train.pkl') :
        infile = open("we_train.pkl",'rb')
        svc=pickle.load(infile)
        infile.close()
    else:
        all_vectors = create_vectors(model,clean_tweet)
        outfile = open("we_train.pkl",'wb')
        svc = svm.SVC(kernel='linear', C=1, probability=True)
        svc = svc.fit(all_vectors, labels)
        pickle.dump(svc,outfile)
        outfile.close()
    
    test = create_vectors(model,vocab)
    prediction = svc.predict(test) #predict on the validation set
    score = f1_score(prediction, valid_labels, average='micro')
    print(score)
    return score
    # print(model2.wv.most_similar(positive='demi'))
    # print(model2.similarity('hour', 'nite'))

def tf_idf(all_tweets,labels,vocab,valid_labels):
    
    tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    txt_fitted = tf.fit(all_tweets)
    txt_transformed = txt_fitted.transform(all_tweets)
    idf = tf.idf_
    
    rr = dict(zip(txt_fitted.get_feature_names(), idf))
    
    if os.path.isfile('td_idf_train.pkl'):
        infile = open("td_idf_train.pkl",'rb')
        svc=pickle.load(infile)
        infile.close()
    else:
        outfile = open("td_idf_train.pkl",'wb')
        svc = svm.SVC(kernel='linear', C=1, probability=True)
        svc = svc.fit(txt_transformed, labels) # xtrain_bow:bag of words features for train data, ytrain: train data labels
        pickle.dump(svc,outfile)
        outfile.close()
    
    test = txt_fitted.transform(vocab)
    prediction = svc.predict(test) #predict on the validation set
    score = f1_score(prediction, valid_labels, average='micro')
    print(score)
    return score


def bow(all_tweets,labels,vocab,valid_labels):
    vectorizer = CountVectorizer()
    bow_xtrain = vectorizer.fit_transform(all_tweets)
    
    if os.path.isfile('bow_train.pkl') :
        infile = open("bow_train.pkl",'rb')
        svc=pickle.load(infile)
        infile.close()
    else:
        outfile = open("bow_train.pkl",'wb')
        svc = svm.SVC(kernel='linear', C=1, probability=True)
        svc = svc.fit(bow_xtrain, labels) # xtrain_bow:bag of words features for train data, ytrain: train data labels
        pickle.dump(svc,outfile)
        outfile.close()

    test = vectorizer.transform(vocab)
    prediction = svc.predict(test) #predict on the validation set
    score = f1_score(prediction, valid_labels, average='micro')
    print(score)
    return score

if __name__ == "__main__":
    train_list = read_data('twitter_data/train2017.tsv')
    test_list = read_data('twitter_data/test2017.tsv')
    valid_labels = read_labels('twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt')
    df = train_list[0]
    all_tweets = np.array(df.values.tolist())
    labels = train_list[1]
    vocab = test_list[0]
    #bow(all_tweets,labels,vocab,valid_labels)
    #tf_idf(all_tweets,labels,vocab,valid_labels)
    #word_embeddings(df,labels,vocab,valid_labels)