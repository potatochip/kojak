#!/usr/bin/env python

from flask import Flask, make_response, render_template
from cStringIO import StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import webbrowser
import data_grab
import pandas as pd


app = Flask(__name__)

train_df = data_grab.get_selects('train')

webbrowser.open('http://127.0.0.1:5000')


@app.route("/")
def make_html():
    x = train_df.to_html()
    return x


# @app.route('/')
# def index():
#     return """\
# <html>
# <body>
# <img src="/plot.png">
# </body>
# </html>"""


@app.route('/plot.png')
def plot():
    df = data_grab.get_selects('train', ['inspection_date'])
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    df = df.reindex(pd.date_range(min(df.index), max(df.index)), fill_value=0)

    df.plot()
    canvas = FigureCanvas(plt.gcf())
    output = StringIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response


# app.route('/analysis/<filename>')
# def analysis(filename):
#     # x = pd.DataFrame(np.random.randn(20, 5))
#     x = train_df
#     return render_template("analysis.html", name=filename, data=x.to_html())


if __name__ == "__main__":
    app.run()
