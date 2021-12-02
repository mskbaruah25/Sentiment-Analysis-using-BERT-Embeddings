
from flask import Flask, render_template, request, redirect
import utils

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        return render_template("index.html")

    if request.method == 'POST':
        result = request.form.to_dict(flat=True)
        review = result.get("review_text")
        sim_msg = utils.sentiment_analysis(review)
        result["sim_msg"] = sim_msg
        return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)