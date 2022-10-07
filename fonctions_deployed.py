
#!pip install openai
import os
import openai
import requests 
import re
import pandas as pd 
from PIL import Image
from bs4 import BeautifulSoup
import nltk
import math
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import re
import pandas as pd 
from PIL import Image
import glob
import torch
import warnings

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer

OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY#os.getenv("OPENAI_API_KEY")


# # define functions
#Realise les fonctions TF-IDF
def tf(word, document):   
    return document.count(word) / (len(document) * 1.)

def n_containing(word, liste_document):   
    return sum(1. for document in liste_document if word in document)

def idf(word, liste_document):   
    return math.log(len(liste_document) / (1. + n_containing(word, liste_document)))

def tfidf(word, document, liste_document):   
    return tf(word, document) * idf(word, liste_document)


def get_tokens(text):
    lowers = text.lower()
    no_punctuation = lowers.translate(str.maketrans('', '', string.punctuation))    
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def stemmed_info(text):
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]  
    
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    
    return stemmed


# summarization
def summary_gpt3(input_text):
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=f"Summarize this for a second-grade student:\n\n: {input_text}\n\nSummary:",
    temperature=0.7,
    max_tokens=550,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
    
    summary = response["choices"][0]["text"]
    return summary.replace("\n","").strip()


# Ttile
def title_gpt3(input_text):
    response = openai.Completion.create(
        engine="davinci",
        prompt= input_text + "\n\nheadline:",
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
        #n=3,
        #best_of=1,
    )
    return response["choices"][0]["text"]
# topics:
def topics_gpt3(input_text):
    response = openai.Completion.create(
        engine="davinci",
        prompt= input_text + "\n\nKeywords & topics:",
        temperature=0,  
        max_tokens=16,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
        #n=3,
        #best_of=1,
    )
    return response["choices"][0]["text"]
def getdata(url): 
    r = requests.get(url) 
    
    return r.text
def most_similar_file(array,documentkeywords):
    similarity = list()
    sim = 0.0
    users_interests = documentkeywords
    for i in range(len(documentkeywords)):
        
        for j in range(len(array)):
           
            for k in range(1,10):
        
                if(array[j] == users_interests[i][k][0]):
                    
                    
                    sim = sim + 1.0*users_interests[i][k][1]
                    
             
            similarity.append((i,sim))
        
            sim = 0.0
                 
    sorted_sim = sorted(similarity,
                  key = lambda x:x[1],
                  reverse = True)
    return(sorted_sim)


# # Search the links with query


def sum_fun (query):
    links = []
    try:
        from googlesearch import search
    except ImportError:
        print("No module named 'google' found")
     
    # to search
 
    for j in search(query, tld="co.in", num=20, stop=10, pause=2):
        #TLD: TLD stands for the top-level domain which means we want to search our results on google.com or google. in 
        #print(j)
        links.append(j)

    # # scraping the web  page

    textList = []
    textComp=[]
    for i in links :
        htmldata = getdata(i) 
        soup = BeautifulSoup(htmldata, 'html.parser') 
        for data  in  (soup.find_all("p")) :
            if (data.find("span") in data) :
                continue
        
            else :
                if len(stemmed_info(data.get_text())) < 10:
                    continue
                else :
                    textList.append (stemmed_info(data.get_text()))
                    textComp.append (data.get_text())
         

    #print(textComp)


    # # Searching relevant paraghrah 
    # The inverse document frequency is a measure of the importance of the term in the whole corpus. In the TF-IDF scheme, 
    #it aims to give greater weight to less frequent terms, considered to be more discriminating 
    #It consists of calculating the logarithm of the inverse of the proportion of documents in the corpus that contain the term

    #define the quantity of the output key words 
    N = 10
    l=0 
    #define a list to save the key words
    documentkeywords = list()
    #print(len(textList))
    for i, document in enumerate(textList):
       
        #print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, document, textList) for word in document}
        
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        #print(sorted_words)
        #for word, score in sorted_words[:N]:
            
            #print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
        words = list()
        if (len(scores)<N):
            l= len(textList)-1
            #print(l)
            continue
            
        for i in range(N):
            
            words.append(sorted_words[i])
            
        
        documentkeywords.append(words)



    #Recommandation part
       
    num_array = list()
    doc=""
        
        
    for i , a in enumerate(query.split()):
        num_array.append(str(a))
        doc= doc + str(a) + " "

        
    print ('ARRAY: ',num_array)
    print('par ordre decroissant de similarite (numero_document, similarite)')
    print(most_similar_file(stemmed_info(doc),documentkeywords)) 
    print("wanted paragraph :")
    res = most_similar_file(stemmed_info(doc),documentkeywords)
    final_text=[]
    for i in res:
        if i[1] != 0.0 :
            final_text.append(re.sub(r'\[(.*?)\]','',textComp[i[0]]))
            
            #print (re.sub(r'\[(.*?)\]','',textComp[i[0]]))
            #print("------------------------------")
    final_text1=[]
    for i in final_text : 
        if i not in final_text1: 
            final_text1.append(i) 
    #print(final_text1)


    # Takes the input paragraph and splits it into a list of sentences
    from sentence_splitter import SentenceSplitter, split_text_into_sentences
    sentence_list=[]
    splitter = SentenceSplitter(language='en')
    for i in final_text1:
        sentence_list.append(splitter.split(i))

    # Do a for loop to iterate through the list of sentences and paraphrase each sentence in the iteration
    paraphrase0 = []

    for i in final_text1:
        #print("test:",i)
        a =summary_gpt3(i)
        #print("sum :",a)
        
        paraphrase0.append(a)
    paraphrase2 = [' '.join(x for x in paraphrase0) ]
    paraphrased_text = str(paraphrase2).strip('[]').strip("'")
    print(paraphrased_text)
    return(paraphrased_text)

#get_response("what toxins can cause seizures in dogs",1)



#PEAGUS
# # paraphrase text using transformers 
#PEGASUS is an acronym for Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence models

# https://huggingface.co/tuner007/pegasus_paraphrase

##import torch
##from transformers import PegasusForConditionalGeneration, PegasusTokenizer
##
##model_name = 'tuner007/pegasus_paraphrase'
##torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
##tokenizer = PegasusTokenizer.from_pretrained(model_name)
##model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
##
##def get_response(input_text,num_return_sequences):
##  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
##  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
##  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
##  return tgt_text


# Do a for loop to iterate through the list of sentences and paraphrase each sentence in the iteration
##paraphrase = []
##
##for i in sentence_list:
##    for j in i :
##        a = get_response(j,1)
##        paraphrase.append(a)

# # save the final text in a document

##from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
##Story=[]
##doc = SimpleDocTemplate("article_text.pdf")
##styleSheet = getSampleStyleSheet()
##bt = styleSheet['BodyText']
##btL = ParagraphStyle('BodyTextTTLower',parent=bt,textTransform='lowercase')
##btU = ParagraphStyle('BodyTextTTUpper',parent=bt,textTransform='uppercase')
##Story.append(Paragraph('''<font color="green"> text  extraction file </font>''',style=btU))
##
## 
##  
##        
##bogustext = ("\n  %s  " %paraphrased_text) 
##p = Paragraph(bogustext, style=btL)
##Story.append(p)   
##     
##doc.multiBuild(Story) 
##       


# # extract images and save them in a folder and in a document

##
##
##from bing_image_downloader import downloader
##downloader.download("2022 Russian invasion Ukraine", limit=100,  output_dir='dataset', timeout=60, verbose=True)

#scaping images from web pages
##img_tags = soup.find_all('img')
##urls= []
##
##urls = [img['src'] for img in img_tags]
##
##for i,url in enumerate (urls):
##
##        img = Image.open(requests.get(url, stream = True).raw)
##        if ((img.width < 50) or ((img.height)<50)):
##            
##            continue 
##        else :
##            
##            img.save(f'images\image {i}.png')
##
    
    
##
##from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle 
##from reportlab.platypus.flowables import ParagraphAndImage, Image
##from reportlab.lib.testutils import testsFolder
##story=[]
##doc = SimpleDocTemplate("article_image.pdf")
##styleSheet = getSampleStyleSheet()
##normal = ParagraphStyle(name='normal',fontName='Times-Roman',fontSize=12,leading=1.2*12,parent=styleSheet['Normal'])
##bt = styleSheet['BodyText']
##tit =styleSheet['Heading3']
##btL = ParagraphStyle('BodyTextTTLower',parent=bt,textTransform='lowercase')
##btU = ParagraphStyle('BodyTextTTUpper',parent=bt,textTransform='uppercase')
##story.append(Paragraph('''<font color="green">image extraction file </font>''',style= tit))
##images ='images'
##
##i=0
##for filename in glob.glob('dataset/2022 Russian invasion Ukraine/*.jpg'): 
##    
##    txt= "image "+ str(i) + "\n"
##    story.append(Paragraph(txt,style=btL))
##    story.append(Paragraph("",style=btL))
##    story.append(Image(filename ,width=390,height=200) )
##    i = i+1 

##doc.multiBuild(story) 
##       

