from plots import *
from sk import *

def word_embeddings(train_tweets,labels,test_tweets,valid_labels,pos_neg,pos_neut,neg_neut,pos_negL,pos_neutL,neg_neutL):
    train_tweets = train_tweets.apply(lambda x: clean_data(x))
    test_tweets = test_tweets.apply(lambda x: clean_data(x))
    
    pos_neg = pos_neg.apply(lambda x: clean_data(x))
    pos_neut = pos_neut.apply(lambda x: clean_data(x))
    neg_neut = neg_neut.apply(lambda x: clean_data(x))
    
    model2 = gensim.models.Word2Vec(pos_neg, min_count = 1, size = 1000, window = 5, sg = 1, workers = 2)
    model2.train(pos_neg, total_examples=len(pos_neg), epochs=20)
    
    all_vectors = create_vectors(model2,train_tweets)
    
    pn= create_vectors(model2,pos_neg)
    kPN = KNeighborsClassifier(n_neighbors = 5)
    kPN=kPN.fit(pn, pos_negL)
    
    test = create_vectors(model2,test_tweets)
    
    pn_test=kPN.predict_proba(test)
    pn_train=kPN.predict_proba(all_vectors)
    
    model2 = gensim.models.Word2Vec(pos_neut, min_count = 1, size = 1000, window = 5, sg = 1, workers = 2)
    model2.train(pos_neut, total_examples=len(pos_neut), epochs=20)
    
    all_vectors = create_vectors(model2,train_tweets)
    
    pnt= create_vectors(model2,pos_neut)
    test = create_vectors(model2,test_tweets)
    
    kPNT = KNeighborsClassifier(n_neighbors = 5)
    kPNT=kPNT.fit(pnt, pos_neutL)
    pnt_test=kPNT.predict_proba(test)
    pnt_train=kPNT.predict_proba(all_vectors)
    
    model2 = gensim.models.Word2Vec(neg_neut, min_count = 1, size = 1000, window = 5, sg = 1, workers = 2)
    model2.train(neg_neut, total_examples=len(neg_neut), epochs=20)
    
    all_vectors = create_vectors(model2,train_tweets)
    
    nn=  create_vectors(model2,neg_neut)
    kNN = KNeighborsClassifier(n_neighbors = 5)
    kNN=kNN.fit(nn, neg_neutL)
    
    test = create_vectors(model2,test_tweets)
    nn_test=kNN.predict_proba(test)
    nn_train=kNN.predict_proba(all_vectors)
    
    test=np.concatenate((nn_test,pn_test, pnt_test), axis=1)
    train = np.concatenate((nn_train,pn_train, pnt_train), axis=1)
    
    model2 = gensim.models.Word2Vec(train_tweets, min_count = 1, size = 1000, window = 5, sg = 1, workers = 2)
    model2.train(train_tweets, total_examples=len(train_tweets), epochs=20)
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn=knn.fit(train, labels)
    prediction=knn.predict(test)
    print(f1_score(prediction, valid_labels, average='micro'))

def tf_idf(vocab,labels,vocab2,valid_labels,pos_neg,pos_neut,neg_neut,pos_negL,pos_neutL,neg_neutL):
    tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    txt_fitted = tf.fit(vocab)
    txt_transformed = txt_fitted.transform(vocab)
    idf = tf.idf_
    
    rr = dict(zip(txt_fitted.get_feature_names(), idf))
    
    pn= txt_fitted.transform(pos_neg)
    pnt= txt_fitted.transform(pos_neut)
    nn= txt_fitted.transform(neg_neut)
    
    kPN = KNeighborsClassifier(n_neighbors = 5)
    kPNT = KNeighborsClassifier(n_neighbors = 5)
    kNN = KNeighborsClassifier(n_neighbors = 5)
    
    kPN=kPN.fit(pn, pos_negL)
    kPNT=kPNT.fit(pnt, pos_neutL)
    kNN=kNN.fit(nn, neg_neutL)
    
    test=txt_fitted.transform(vocab2)
    
    pn_test=kPN.predict_proba(test)
    pnt_test=kPNT.predict_proba(test)
    nn_test=kNN.predict_proba(test)
    
    pn_train=kPN.predict_proba(txt_transformed)
    pnt_train=kPNT.predict_proba(txt_transformed)
    nn_train=kNN.predict_proba(txt_transformed)
    
    test=np.concatenate((nn_test,pn_test, pnt_test), axis=1)
    train = np.concatenate((nn_train,pn_train, pnt_train), axis=1)
    
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn=knn.fit(train, labels)
    prediction=knn.predict(test)
    
    print(f1_score(prediction, valid_labels, average='micro'))

def bow(vocab,labels,vocab2,valid_labels,pos_neg,pos_neut,neg_neut,pos_negL,pos_neutL,neg_neutL):
    vectorizer = CountVectorizer()
    bow_xtrain = vectorizer.fit_transform(vocab)
    
    pn= vectorizer.transform(pos_neg)
    pnt= vectorizer.transform(pos_neut)
    nn= vectorizer.transform(neg_neut)
    
    kPN = KNeighborsClassifier(n_neighbors = 5)
    kPNT = KNeighborsClassifier(n_neighbors = 5)
    kNN = KNeighborsClassifier(n_neighbors = 5)
    
    kPN=kPN.fit(pn, pos_negL)
    kPNT=kPNT.fit(pnt, pos_neutL)
    kNN=kNN.fit(nn, neg_neutL)
    
    test=vectorizer.transform(vocab2)
    
    pn_test=kPN.predict_proba(test)
    pnt_test=kPNT.predict_proba(test)
    nn_test=kNN.predict_proba(test)
    
    pn_train=kPN.predict_proba(bow_xtrain)
    pnt_train=kPNT.predict_proba(bow_xtrain)
    nn_train=kNN.predict_proba(bow_xtrain)
    
    test=np.concatenate((nn_test,pn_test, pnt_test), axis=1)
    train = np.concatenate((nn_train,pn_train, pnt_train), axis=1)
    
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn=knn.fit(train, labels)
    prediction=knn.predict(test)
    print(f1_score(prediction, valid_labels, average='micro'))

if __name__ == "__main__":
    data_list = read_data3('twitter_data/train2017.tsv')
    valid_labels=read_labels('twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt')
    data_list2 = read_data3('twitter_data/test2017.tsv')
    # bow(data_list[0],data_list[2],data_list2[0],valid_labels,data_list[3],data_list[4],data_list[5],data_list[6],data_list[7],data_list[8])
    # tf_idf(data_list[0],data_list[2],data_list2[0],valid_labels,data_list[3],data_list[4],data_list[5],data_list[6],data_list[7],data_list[8])
    # word_embeddings(data_list[1],data_list[2],data_list2[1],valid_labels,data_list[9],data_list[10],data_list[11],data_list[6],data_list[7],data_list[8])
