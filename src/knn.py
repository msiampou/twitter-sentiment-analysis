from plots import *
from sk import *

def word_embeddings(all_tweets,labels,vocab,valid_labels):
    
    clean_tweet = all_tweets.apply(lambda x: clean_data(x))
    print(clean_tweet)
    
    if os.path.isfile('we_model_knn.pkl') :
        infile = open("we_model_knn.pkl",'rb')
        model = pickle.load(infile)
        infile.close()
    else:
        model = gensim.models.Word2Vec(clean_tweet, size=300, window=5, min_count=2)
        model.train(clean_tweet, total_examples=len(clean_tweet), epochs=20)
        outfile = open("we_model_knn.pkl",'wb')
        pickle.dump(model,outfile)    
        outfile.close()
    
    tsne_plot(model)

    if os.path.isfile('we_knn.pkl') :
        infile = open("we_knn.pkl",'rb')
        knn=pickle.load(infile)
        infile.close()
        print("yes")
    else:
        all_vectors = create_vectors(model,clean_tweet)
        print("done")
        outfile = open("we_knn.pkl",'wb')
        knn = KNeighborsClassifier(n_neighbors = 5)
        knn.fit(all_vectors, labels) 
        pickle.dump(knn,outfile)
        outfile.close()

    test = create_vectors(model,vocab)
    prediction=knn.predict(test) #predict on the validation set
    score=f1_score(prediction, valid_labels, average='micro')
    print(score)
    # print(model2.wv.most_similar(positive='demi'))
    # print(model2.similarity('hour', 'nite'))

def tf_idf(all_tweets,labels,vocab,valid_labels):
    
    tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    txt_fitted = tf.fit(all_tweets)
    txt_transformed = txt_fitted.transform(all_tweets)
    idf = tf.idf_

    rr = dict(zip(txt_fitted.get_feature_names(), idf))
    
    if os.path.isfile('td_idf_knn.pkl') :
        infile = open("td_idf_knn.pkl",'rb')
        knn=pickle.load(infile)
        infile.close()
    else:
        outfile = open("td_idf_knn.pkl",'wb')
        knn = KNeighborsClassifier(n_neighbors = 5)
        knn.fit(txt_transformed, labels) # xtrain_bow:bag of words features for train data, ytrain: train data labels
        pickle.dump(knn,outfile)
        outfile.close()
    
    test=txt_fitted.transform(vocab)
    prediction=knn.predict(test) #predict on the validation set
    score=f1_score(prediction, valid_labels, average='micro')
    print(score)
    
def bow(all_tweets,labels,vocab,valid_labels):
    vectorizer = CountVectorizer()
    bow_xtrain = vectorizer.fit_transform(all_tweets)
    
    if os.path.isfile('bow_knn.pkl') :
        infile = open("bow_knn.pkl",'rb')
        knn=pickle.load(infile)
        infile.close()
    else:
        outfile = open("bow_knn.pkl",'wb')
        knn = KNeighborsClassifier(n_neighbors = 5)
        knn.fit(bow_xtrain, labels) # xtrain_bow:bag of words features for train data, ytrain: train data labels
        pickle.dump(knn,outfile)
        outfile.close()

    test=vectorizer.transform(vocab)
    prediction=knn.predict(test) #predict on the validation set
    score=f1_score(prediction, valid_labels, average='micro')
   
if __name__ == "__main__":
    train_list = read_data('twitter_data/train2017.tsv')
    test_list = read_data('twitter_data/test2017.tsv')
    valid_labels=read_labels('twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt')
    df = train_list[0]
    all_tweets = np.array(df.values.tolist())
    labels = train_list[1]
    vocab = test_list[0]
    #bow(all_tweets,labels,vocab,valid_labels)
    #tf_idf(all_tweets,labels,vocab,valid_labels)
    #word_embeddings(df,labels,vocab,valid_labels)
