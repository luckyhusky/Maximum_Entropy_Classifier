# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from corpus import ReviewCorpus, Document
from random import seed, shuffle
import numpy as np
import math
from scipy.misc import logsumexp
from scipy.optimize import minimize

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import RSLPStemmer

import xlwt

class MaxEnt(Classifier):

    def __init__(self, model=None):
        self.label = {} # store all the label in the training data set
        self.features = {}  # store all the features in the training data
        self.rowLen = 0
        self.columnLen = 0
        self.featureFunction = {} # Using how many times the words exist in the label as feature function 
        self.devAccu = [] # store dev data set accuracy after each upgrade

    def get_model(self):
        return self.maxEntModel

    def set_model(self, model):
        self.maxEntModel = model

    model = property(get_model, set_model)

    def train(self, instances, dev_instances):
        """Construct a statistical model from labeled instances."""
        for instance in instances:
            l = instance.label
            if l == "":
                continue
            #  Get all label/class of the model
            if l not in self.label:
                self.featureFunction[l] = {}
                self.label[l] = self.rowLen
                self.rowLen += 1

            featureDict = self.featureFunction[l]
            #  Get features for all the dataset
            for feature in instance.features():
                featureDict[feature] = featureDict.get(feature, 0) + 1
                

        # find the feauture which has been exist document in that label for more than average times 
        for l, fDict in self.featureFunction.iteritems():
            mean = np.mean(fDict.values())
            std = np.std(fDict.values())
            maximum = np.max(fDict.values())
            for f, v in fDict.iteritems():
                if v > mean - std and v < mean + std and f not in self.features and maximum > 1200:
                    self.features[f] = self.columnLen
                    self.columnLen += 1
                elif maximum < 1200:
                    self.features[f] = self.columnLen
                    self.columnLen += 1


        # create 2-D array to store 
        self.maxEntModel = np.zeros((self.rowLen, self.columnLen))

        batch_size = 30
        learning_rate = 0.001
        mini_batches = list(self.chop_up(instances, batch_size))
        is_converged = False
        pre_accu = 0
        self.count = 1

        #  If not converged, then upgrade the lambda
        while not is_converged:
            for mini_batch in mini_batches:
                cur_accu = self.train_sgd(mini_batch, dev_instances, learning_rate, batch_size) / (self.count * batch_size)
                if(abs(cur_accu - pre_accu) <= 0.001):
                    is_converged = True
                    break
                else:
                    print(self.count)
                    self.count += 1
                    pre_accu = cur_accu
        print(self.count)
        # save dev_accuracy into an excel sheet
        self.save_dev_accuracy()
               
    def chop_up(self, train, length):
        """chop the training set into mini_batches."""
        for i in xrange(0, len(train), length):
            yield train[i : i + length]

    def save_dev_accuracy(self):
        excel = xlwt.Workbook(encoding = "utf-8")
        dev_sheet = excel.add_sheet("dev_sheet")
        i = 0
        for n in self.devAccu:
            dev_sheet.write(i, 0, n)
            i += 1
        excel.save("dev_accuracy.xls")

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient
        """
        self.obs = np.zeros((self.rowLen, self.columnLen))
        self.exp = np.zeros((self.rowLen, self.columnLen))

        for data in train_instances:
            gradient = self.calc_gradient(data)
            self.maxEntModel += gradient * learning_rate
        
        accu = self.dev_accuracy(dev_instances)
        self.devAccu.append(accu)
        return np.sum(np.gradient(gradient))

    def dev_accuracy(self, dev_instances):
        correct = [self.classify(x) == x.label for x in dev_instances]
        return float(sum(correct)) / len(correct)

    def calc_gradient(self, data):
        l = data.label
        row_index = self.label[l]
        posterior = self.calc_posterior(data)
            
        for f in data.features():
            if f not in self.features:
                continue
            col_index = self.features.get(f)
            #  calculate the observed 
            self.obs[row_index][col_index] += 1
            #  calculate the expected
            for row in range(len(posterior)):
                self.exp[row][col_index] += posterior[row]

        return self.obs - self.exp

    def calc_posterior(self, instance):
        l = instance.label
        score = np.zeros(self.rowLen)
        # posterior is a list store data point's each posterior in every lable 
        posterior = []

        for f in instance.features():
            if f not in self.features:
                continue
            col_index = self.features.get(f)

            # calculate the score for all the label
            for row in range(self.rowLen):
                score[row] += self.maxEntModel[row][col_index]

        for row in score:
            posterior.append(math.exp(row - logsumexp(score)))
        return posterior

    def classify(self, instance):
        result = self.calc_posterior(instance)
        for k, v in self.label.iteritems():
            if np.argmax(result) == v:
                return k

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()


class BagOfWordsTokenized(Document):
    def features(self):
        return word_tokenize(self.data)


class BagOfWordsStemmed(Document):
    def features(self):
        """Stem the word"""
        st = RSLPStemmer()    
        return [st.stem(word) for word in self.data.split()]


class NGram(Document):
    def features(self):
        """Use N gram to extract feature """
        n = 1
        data = self.data.split()
        st = RSLPStemmer()    
        data = [st.stem(word) for word in self.data.split()]
        out = []
        for i  in range(n, len(self.data.split()) - n + 1):
            out.append(data[i - n:i])
            out.append(data[i + 1:i + n])
        return [' '.join(x) for x in out]


def split_review_corpus(document_class):
    """Split the yelp review corpus into training, dev, and test sets"""
    reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
    seed(hash("reviews"))
    shuffle(reviews)
    return (reviews[:10000], reviews[10000:11000], reviews[11000:14000])


def accuracy(classifier, test):
    """Find the performance of model"""
    correct = [classifier.classify(x) == x.label for x in test]
    return float(sum(correct)) / len(correct)

if __name__ == '__main__':
    train, dev, test = split_review_corpus(BagOfWords)
    print("ready to train")
    classifier = MaxEnt()
    classifier.train(train, dev)
    print(accuracy(classifier, test))

