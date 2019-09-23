import numpy as np
import pickle
import sys
from scipy.spatial.distance import cosine as similar

class NeuralNet():      
    def __init__(self):        # LOAD weights and vocab_index for w2v
        self.w1   = np.load('skipgram_weight1.npy')
                
        in_file   = open('skipgram_word_index.txt','rb')
        self.word_index = pickle.load(in_file)
        in_file.close()

        in_file   = open('skipgram_index_word.txt','rb')
        self.index_word = pickle.load(in_file)
        in_file.close()
        
        self.voc_size=len(self.word_index)
        #print(self.voc_size)
    
    def cos_sim1(self, word1, word2):        #input a word, return cosine similarity
        index1=self.word_index[word1]
        index2=self.word_index[word2]
        vec1=(self.w1)[index1]
        vec2=(self.w1)[index2]
        nr=np.dot(vec1,vec2)
        dr=np.linalg.norm(vec1)*np.linalg.norm(vec2)
        return (nr/dr)

    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def word_sim(self, word, top_n):
        w1_index = self.word_index[word]
        v_w1 = (self.w1)[w1_index]
        word_sim = {}
        for i in range(self.voc_size):
            v_w2 = (self.w1)[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den
            word = self.index_word[i]
            word_sim[word] = theta
        words_sorted = sorted(word_sim.items(), key=lambda kv:kv[1], reverse=True)
        i=1
        for word, sim in words_sorted:
            if(i<=top_n):
                print(word, sim)
                i += 1
            else:
                break
    
    def similarity(self):
        print("Enter two words: ")
        w1=input()
        w2=input()
        #w1=sys.argv[1]
        #w2=sys.argv[2]
        print("Cosine similarity between ",w1, " and", w2, ": ", self.cos_sim1(w1,w2))         
    
    def closest(self):
        print("Enter a word: ")
        w1=input()
        #w1=sys.argv[1]
        print(self.word_sim(w1,20))          
          
#--- Driver --------------------------------------------------------------+
if __name__=='__main__':
    model = NeuralNet()
    
    #model.similarity()    #for similarity between two words
    
    model.closest()       #for top n=10 most closest context words
    
    