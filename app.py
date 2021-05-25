from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open('iri.pkl','rb'))


app=Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/submit', methods=['POST','GET'])
def home():
    if (request.method=='POST'):
        

    file=request.form['myfile']
    #file='0_0.txt'
    #pred=model.predict(file)
    #print(file)
    pickle.dump(file,open('res.pkl','wb'))
    
    return render_template('after.html', data=model)

if __name__=="__main__":
    app.run(debug=True)