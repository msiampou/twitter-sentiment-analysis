import os
import re
import csv
import sys
import gensim
import random
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn import svm
import pickle as pickle
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
