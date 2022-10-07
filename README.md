# Extract_information_from_web_pages
Globally, this project aims to extract from an entry query the information existing in the paragraphs of the web pages
##The process is illustrated in this step:
Scrap the links related to the query 
For each link with the python library BeautifulSoup scrap all paraghs in the web page
Define the key words of each paraghraps and sorted theirs scores from the most  to least relevant in purpose to specify the wanted paraph
Takes the input paragraph and splits it into a list of sentences
Paraphrase each sentence with gpt3 model
The result is displayed in a web  page with flask
