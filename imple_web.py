import os
from flask import Flask, render_template, abort,request,redirect,url_for,flash
import cv2
import glob
from flask import Flask, request,render_template,jsonify,redirect,url_for
from flask_ngrok import run_with_ngrok
import sys
import codecs
import html 

import re
import  Project_Sarmad_v1

app = Flask(__name__)
#run_with_ngrok(app)

pat = re.compile(r"[A-Za-z0-9!;,?.&' ]+") 
  

@app.route('/', methods=['POST','GET'])
def rs():
    if request.method == 'POST':
        if request.form.get('SUBMIT') == "submit":
            q=request.form.get('keyword')
           
             
           
               #return render_template('firstt.html', name = "enter a valid query")
            resu=Project_Sarmad_v1.sum_fun(q)
            res=re.sub("[^a-zA-Z0-9\n\.\,\']", " ", resu)
               
           
           

          
             
        else :
           resu = ""
    
        return render_template('firstt.html', name = resu)
        
          
    elif request.method == 'GET':
       return render_template('firstt.html')  
##@app.route('/', methods = ['GET', 'POST'])
##def htmlPage():
##   if request.method == 'GET':
##       q=request.form['query']   
##       resu=Project_Sarmad_v1.sum_fun(str(q))
##       #"what toxins can cause seizures in dogs"
##       print(resu)
##               
##   return render_template('firstt.html')   

    
if __name__ == '__main__':
    app.run()
####    
