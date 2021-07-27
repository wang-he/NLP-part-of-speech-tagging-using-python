# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:39:21 2021

@author: Alan
"""
#open the file
with open('C:\\Users\\USER\\Downloads\\Shakespeare.txt')as f:
      files= f.read()
      #print(files)
#Restore some common abbreviation
import re
pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I) # to find the 's following the pronouns. re.I is refers to ignore case
pat_s = re.compile("(?<=[a-zA-Z])\'s")# to find the 's following the letters
pat_s2 = re.compile("(?<=s)\'s?")# to find the ' following the words ending by s
pat_not = re.compile("(?<=[a-zA-Z])n\'t")# to find the abbreviation of not
pat_would = re.compile("(?<=[a-zA-Z])\'d")# to find the abbreviation of would
pat_will = re.compile("(?<=[a-zA-Z])\'ll")# to find the abbreviation of will
pat_am = re.compile("(?<=[I|i])\'m")# to find the abbreviation of am
pat_are = re.compile("(?<=[a-zA-Z])\'re")# to find the abbreviation of are
pat_ve = re.compile("(?<=[a-zA-Z])\'ve") # to find the abbreviation of have
#After searching, it is found that there are some special abbreviations in the source file
pat_the = re.compile("(?<=[(^a-zA-Z)th/Th])\'(\s)") #th' means the
pat_verb = re.compile("(lov|mak|ceiv|ow|gav|deserv|rud)(\'st)", re.I)#restore verb

#substitute
files = pat_is.sub(r"\1 is", files)
#’s /s'represents the possessive form, which is directly removed when restoring, leaving the verb prototype
files = pat_s.sub("", files)
files = pat_s2.sub("", files)
files = pat_not.sub(" not", files) # turn 't into not
files = pat_would.sub(" would", files) # turn 'd into would
files = pat_will.sub(" will", files) # turn 'll into will
files = pat_am.sub(" am", files) # turn 'm into am
files = pat_are.sub(" are", files) # turn 're into are
files = pat_ve.sub(" have", files)  # turn' ve into have
files = pat_the.sub("e ",files) # turn th' into the
files = pat_verb.sub(r"\1e", files) # add e to after the verb less e 
files = files.replace("'t"," it ") # Turned "‘t" into "it" 
files = files.replace("'Tis"," It is ") # turn 'Tis' into 'it is'
files = files.replace("'er","ver ") # turn 'er' into 'ver'
files = files.replace("'st","") # remove 'st
files = files.replace('\'', '') # remove '

# We don't consider lowercase and uppercase of words, uniformly converted to lowercase.
files=files.lower()
#Filter out all non-word characters
noneng_word=re.compile(r"\W+")
files=re.sub(noneng_word," ",files)
#Complete file before split
print(files)


#Split words
words=files.split()
print(words)


# Count the number of words in the article
count=0
# for loop word in the collection after split
for word in words:
    # if it is a word, count+1
    count=+ len(words)
print("The number of words in the whole article is:",count)


# The frequency of the top 20 words in the document 
import pandas as pd
fre=pd.Series(words).value_counts()
print("The frequency of the top 20 words are:")
print(fre[:20])
#print(fre)


# data visualization
import matplotlib.pyplot as plt
#count the unique word
unique_count=0
for word in fre:
    unique_count=+ len(fre)
print(unique_count)
# count is total number of words; unique_count is total number of unique words
x=("count","unique_count")
y=(count,unique_count)
plt.bar(x,y)
plt.title('frequency')
plt.show()


# Word Cloud: A word cloud showing the keywords of the article
#pip install wordcloud
from wordcloud import WordCloud
# import files to WordCloud and generate wordcloud
wordcloud = WordCloud().generate(files)
# show wordcloud. The two parameters of imshow, the first is text, which stores the text of the image, and the second is interpolation, bilinear.
plt.imshow(wordcloud, interpolation="bilinear")
# turn off axis
plt.axis("off")
plt.show()


# Build a dictionary and arrange it in alphabetical order
import pandas as pd
# define a dictionary function, in which we have two parameters: one is words after split, the other is minimum word frequency.
def build_dict(sp_words, min_word_freq=0):
    word_freq = pd.Series(sp_words).value_counts()   #show frequency of each word. pd.series is a tool of pandas, sp_words is a parameter.
    word_freq = filter(lambda x: x[1]>=min_word_freq, word_freq.items()) # if the collection of x is greater than min_word_freq, we keep the collection, items are elements list of the collection
    word_freq_sorted = sorted(word_freq, key=lambda x: (x[0], x[1]))     # sort by frequency of words, default is ascending order,alphabetically sort.
    words, _ = list(zip(*word_freq_sorted))     # output the result as a list.
    word_idx = dict(zip(words, range(len(words))))      # The dict() function can be used to convert the zip object into a dictionary. The former produces the key, and the latter produces the value.
    word_idx['<unk>'] = len(words)      #unk means unknown, unknown word
    return word_idx
# call the build_dict function.word is words after split, 1 is True.
dic=build_dict(words,1)
print("The content of dictionary:",dic)


# Count the number of words in the dictionary
dic_c=0
for word in dic:
    dic_c=+len(dic)
print("The number of words in the dictionary is: ",dic_c)


# Select three sentences in the document, perform part-of-speech tagging, identify part-of-speech
#pip install nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('brown')
import nltk  
t1="You had a father, let your son say so."
t2="That Time will come and take my love away "
t3="Nature bequest gives nothing but doth lend. "
# put t1,t2,t3 into a corpus, named dict
dict= list(set(nltk.word_tokenize(t1)+nltk.word_tokenize(t2)+nltk.word_tokenize(t3)))
# use pos_tag in nltk to do part of speech tagging.
tag=nltk.pos_tag(dict)
print(tag)


from nltk.corpus import brown
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
#brown news as a training set training model
brown_tagged_sents = brown.tagged_sents(categories='news')
default_tagger = nltk.DefaultTagger('NN')
#training set and test set is 9:1
train_data=brown_tagged_sents[:int(len(brown_tagged_sents)*0.9)]
test_data=brown_tagged_sents[int(len(brown_tagged_sents)*0.9):]
#Unigram is a uninary tagger, setting backoff= default_tagger represents the default tagger if not found
unigram_tagger=UnigramTagger(train_data,backoff=default_tagger)
print("The result of unigram is:",unigram_tagger.evaluate(test_data))
#Bigram is a binary tagger, setting backoff=unigram_tagger represents the unigram tagger if not found
bigram_tagger=BigramTagger(train_data,backoff=unigram_tagger)
print("The result of bigram is:",bigram_tagger.evaluate(test_data))
#Trigram is a trinary tagger, setting backoff=bigram_tagger represents the bigram tagger if not found
trigram_tagger=TrigramTagger(train_data,backoff=bigram_tagger)
print("The result of trigram is:",trigram_tagger.evaluate(test_data))

#Save tagger
from pickle import dump
# Save the tagger unigram to the file unigram.pkl
output = open('unigram_tagger.pkl','wb')
dump(unigram_tagger,output,-1)
output.close()
#Load tagger
from pickle import load
# Import from file
input = open('unigram_tagger.pkl','rb')
tagger = load(input)
input.close()
#Use tagger
#part of speech tagging for three sentences above
tagged=tagger.tag(dict)
print("The tags of three sentences are:",tagged)

# Visualization of part-of-speech tagging
all_noun=[word for word ,pos in tagged if pos in ['NN','NN-TL']] # see the pos tagging who is the NN.
all_verb=[word for word ,pos in tagged if pos in ['VB','HVD','VBN','MD']] # see the pos tagging who is the VB
all_pron=[word for word ,pos in tagged if pos in ['DT','PPSS','PP$']]
all_adv=[word for word ,pos in tagged if pos in ['RB']]
all_conj=[word for word ,pos in tagged if pos in ['CC']]
p0=len(dict)  # p0 is total number of words in the three sentences
p1=len(all_noun)/p0  # the percentage of the number of noun in total
p2=len(all_verb)/p0  # the percentage of the number of verb in total
p3=len(all_pron)/p0  # the percentage of the number of pron in total
p4=len(all_adv)/p0
p5=len(all_conj)/p0
p6=1-p1-p2-p3-p4-p5  # the percentage of the number of others in total
labels=['noun','verb','pron','adv','conj','others']
X=[p1,p2,p3,p4,p5,p6] 
colors  = ["#5bae23","#dfecd5","#cad3c3","#9fa39a","#b2cf87","#96c24e"]  
fig = plt.figure()  
# Draw a pie chart (data, the label corresponding to the data, color, and the percentage to two decimal places)
plt.pie(X,labels=labels,colors=colors,autopct='%1.2f%%') 
plt.title("Tag Pie chart")
plt.show()  

