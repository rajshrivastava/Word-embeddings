#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:47:13 2019
SENTIMENT CLASSIFICATION USING HYBRID PREDICTION MODEL
@author: Raj Kumar Shrivastava
"""
import pandas as pd
import numpy as np
import pickle
import time
import random
import re
import math
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import tensorflow as tf

class NeuralNet():
    def __init__(self, config):                           #initializing hyperparameters
        self.n = config['n']
        self.m = config['m']
        self.skipgram_epochs = config['skipgram_epochs']
        self.classifier_epochs = config['classifier_epochs']
        self.window = config['window_size']
        self.mini_batch_size_skipgram=config['mini_batch_size_skipgram']
        self.mini_batch_size_classifier=config['mini_batch_size_classifier']
        self.k = config['k']
        self.alpha_pred = config['alpha_pred']
        self.skipgram_eta = config['skipgram_eta']
        self.classifier_eta = config['classifier_eta']
        self.dropout = 0.5
        pass
    
    def create_corpus(self, mails, polarity):            #data cleansing
        mails=list(mails)
        stop_words=['k','m','t','d','e','f','g','h','i','u','r','I','im',\
                    'ourselves', 'hers', 'between', 'yourself', 'again', \
                    'there', 'about', 'once', 'during', 'out', 'very', \
                    'having', 'with', 'they', 'own', 'an', 'be', 'some', \
                    'for', 'do', 'its', 'yours', 'such', 'into', 'of', \
                    'most', 'itself', 'other', 'off', 'is', 's', 'am', \
                    'or', 'who', 'as', 'from', 'him', 'each', 'the', \
                    'themselves', 'until', 'below', 'are', 'we', 'these', \
                    'your', 'his', 'through', 'don', 'nor', 'me', 'were', \
                    'her', 'more', 'himself', 'this', 'should', 'our', \
                    'their', 'while', 'above', 'both', 'to', 'ours', 'had', \
                    'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',\
                    'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',\
                    'yourselves', 'then', 'that', 'because', 'what', 'over', \
                    'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you',\
                    'herself', 'has', 'just', 'where', 'too', 'only', 'myself',\
                    'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',\
                    'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',\
                    'how', 'further', 'was', 'here', 'than'];
        
        filtered_mails=[]
        pols=[]
        total_words=0
        word_counts_all = {}   #dictionary to store all words and their counts
        porter = PorterStemmer()
        for i, mail in enumerate(mails):     #each mail
            if(type(mail)==float):
                continue
            mail = mail.lower()
            mail = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",mail).split()) #remove puncs
            mail = re.sub(r'(.)\1+', r'\1\1', mail)     #remove repeating characterssss

            #mail_words=mail.split()
            stemmed_words = [porter.stem(word) for word in word_tokenize(mail)]
            filtered_words=[]
            
            for word in stemmed_words:
                if (word not in stop_words) and word.isdigit() == False:
                    filtered_words.append(word)
                    if word in word_counts_all:
                        word_counts_all[word] += 1   #for existing words
                    else:
                        word_counts_all[word] = 1     #for new words
                    total_words += 1
                
            if len(filtered_words)>=2:
                filtered_mails.append(filtered_words)
                pols.append(polarity.iloc[i])
        
        word_counts_reduced = {}    #dictionary to store words occuring atleast n times and their counts
        for key, value in word_counts_all.items():
            if(value>4):
                word_counts_reduced[key] = value
        
        print("Total words = " , total_words)        
        print("Vocabulary size (Unprocessed) = ",len(word_counts_all.keys()))
        self.voc_size = len(word_counts_reduced.keys())
        print("Vocabulary size = ",self.voc_size)
        
        self.words_list = sorted(list(word_counts_reduced.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))        
        print("...created dictionary.")
        
        return filtered_mails, pols
        
    def skipgram_dataset(self, corpus):
        targets = []
        contexts= []
        for sentence in corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):                
                rand_window= np.random.randint(1,self.window+1)  #returns a number between [low, high)
                lower = max(i-rand_window, 0)
                upper = min(sent_len, i+rand_window+1)
                for j in range(lower, upper):
                    if j!=i and j>=0 and j<=sent_len-1:
                        if word in self.word_index.keys() and sentence[j] in self.word_index.keys():
                            targets.append(self.word_index[word])
                            contexts.append(self.word_index[sentence[j]])
        
        return targets, contexts
    
    def train_skipgram(self, targets, contexts):
        length = len(targets)
        print("Length of training data: ", length)        
        print("Minibatch size: ", self.mini_batch_size_skipgram)
        
        #graph architecture
        
        self.w1   = tf.Variable(tf.random_uniform([self.voc_size+1, self.n], -1.0, 1.0), name='w1')

        self.nce_w2   = tf.Variable(tf.truncated_normal([self.voc_size, self.n], stddev=1.0/math.sqrt(self.n)), name='nce_w2')
        self.nce_b2   = tf.Variable(tf.zeros([self.voc_size]), name='nce_b2')
        
        X = tf.placeholder(tf.int32, shape = [None])
        Y = tf.placeholder(tf.int32, shape = [None, 1])  
        
        h = tf.nn.embedding_lookup(self.w1, X) 
        print("h: ",h.get_shape())
        
        loss = tf.reduce_mean(tf.nn.nce_loss(weights = self.nce_w2,
                                             biases = self.nce_b2,
                                             labels = Y,    #[batch_size, num_true]
                                             inputs = h,    #[batch_size, dim]
                                             num_sampled = 64,
                                             num_classes = self.voc_size))
        
        optimizer = tf.train.AdamOptimizer(self.skipgram_eta).minimize(loss)
        #optimizer = tf.train.GradientDescentOptimizer(self.eta).minimize(loss)
        
        #correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #graph architecture end####
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        # train on mini batches
        print("Training started at ", time.ctime(time.time()) )
        for epo in range(self.skipgram_epochs):
            temp = list(zip(targets, contexts))
            random.shuffle(temp) 
            targets, contexts = zip(*temp)
            
            mini_batches_x = [ targets[k:k+self.mini_batch_size_skipgram] for k in range(0, length, self.mini_batch_size_skipgram)]
            mini_batches_y = [ contexts[k:k+self.mini_batch_size_skipgram] for k in range(0, length, self.mini_batch_size_skipgram)]
            
            for mini_count in (range(len(mini_batches_x))):
                batch_x = mini_batches_x[mini_count]
                batch_y = mini_batches_y[mini_count]  
                batch_y = np.array(batch_y).reshape(len(batch_y),1)
                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                
                #print("Minibatch: ", mini_count+1," of Epoch: ", epo+1,"\t| Time =",time.ctime(time.time()) )
                
            testdata_x = batch_x
            testdata_y = batch_y      
            #minibatch_loss, minibatch_accuracy = sess.run([loss, accuracy], feed_dict={X: testdata_x, Y: testdata_y})
            #print("Iteration", epo+1, "\t| Loss =", minibatch_loss, "\t| Accuracy =", minibatch_accuracy, "\t| Time =",time.ctime(time.time()))
            minibatch_loss = sess.run(loss, feed_dict={X: testdata_x, Y: testdata_y})
            print("Iteration", epo+1, "\t| Loss =", minibatch_loss, "\t| Time =",time.ctime(time.time()))
        print("Skipgram training completed at ", time.ctime(time.time()) )
        
        np.save('skipgram_weight1.npy', sess.run(self.w1))
        sess.close()
        
    def save_model(self):
        
        out_file=open('word_index.txt','wb')
        pickle.dump(self.word_index,out_file)
        out_file.close()        
        
        out_file=open('index_word.txt','wb')
        pickle.dump(self.index_word,out_file)
        out_file.close() 
        
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)
    config={'n':70, 'm':20, 'window_size':7, 'mini_batch_size_skipgram':16384,'mini_batch_size_classifier':4096,\
            'skipgram_epochs':3, 'classifier_epochs':7, 'alpha_pred':0.8, 'skipgram_eta':0.03, 'classifier_eta':0.006, 'k':25}   #final 
     
    model = NeuralNet(config)
    print("Reading data...")
    data=pd.read_csv('generated_dataset.csv')   #each row should contain: [index, mail, label]
    mails    = data.iloc[:, 0]   #full
    polarity = data.iloc[:, 1]   #full
    print("No. of emails: ", len(mails))
    
    print("Generating corpus...", time.ctime(time.time()))    
    filtered_mails, polarity= model.create_corpus(mails, polarity)
    
    print("Generating skipgram training_data...", time.ctime(time.time()))
    targets, contexts = model.skipgram_dataset(filtered_mails)
        
    print("Training skipgram...", time.ctime(time.time()))
    model.train_skipgram(targets, contexts)      #training skipgram neural network
    
    print("Saving model...")
    model.save_model()              #saving the parameters to file
    
    print("PROCESS COMPLETED.")