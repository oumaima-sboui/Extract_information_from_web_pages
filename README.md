# Extract information from web pages
Globally, this project aims to extract from an entry query the informations existing in the paragraphs of the web pages
## The process is illustrated in this steps:
* Scrap the links related to the  words of the query 
* For each link, with the python library BeautifulSoup,scrap all paraghs from the web page
* Define the key words of each paraghraps and sorted theirs scores from the most  to least relevant in purpose to specify the wanted paraph using  TF-IDF scheme (The inverse document frequency is a measure of the importance of the term in the whole corpus.It aims to give greater weight to less frequent terms, considered to be more discriminating) 
* Takes the input paragraph and splits it into a list of sentences
* Paraphrase each sentence with gpt3 model.There are as well in comment the PEAGUS model (it paraphrases the text using transformers)
* Save the final text in a pdf document
* Extract images and save them in a folder and in a document
* The result is displayed in a web  page with flask
