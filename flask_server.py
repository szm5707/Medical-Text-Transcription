from flask import Flask, render_template, request, jsonify
import classifyMedicalTerms as cmt
import retrieve_definition as rdef

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classifyText')
def classifyText():
    text = request.args.get('a', 0, type=str)
    print text
    wordlist = text.split(' ')
    prediction = []
    for word in wordlist:
        if(cmt.predict(word)[1]) == 'Medical Term':
            print "sending", str(word), "to Oxford Dictionary...."
            oxResponse = rdef.retrieve_definition(word)
            print oxResponse[0]
            word = oxResponse[0]
        prediction.append(word)
    final_string = ""
    for item in prediction:
        final_string += item + " "
    return jsonify(result = str(final_string))

if __name__ == '__main__':
    app.debug = True
    app.run()
