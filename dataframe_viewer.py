#!/usr/bin/env python
'''
pandas dataframe viewer
'''

from flask import Flask
import webbrowser
import data_grab

app = Flask(__name__)

file1 = 'comp1.txt'
file2 = 'comp2.txt'

train_df = data_grab.get_selects('train', feature_list)

webbrowser.open('http://127.0.0.1:5000')

@app.route("/")
def make_html():

    return raw_html

app.route('/analysis/<filename>')
def analysis(filename):
    # x = pd.DataFrame(np.random.randn(20, 5))
    x = train_df
    return render_template("analysis.html", name=filename, data=x.to_html())


if __name__ == "__main__":
    app.run()
