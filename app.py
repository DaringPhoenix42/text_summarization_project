from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Force the pipeline to use the PyTorch backend
summarizer = pipeline("summarization", model="t5-small", framework="pt")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    original_text = ""
    error = None

    if request.method == "POST":
        original_text = request.form.get("text", "")
        if not original_text.strip():
            error = "Please enter some text to summarize."
        else:
            try:
                result = summarizer(original_text, max_length=100, min_length=30, do_sample=False)
                summary = result[0]['summary_text']
            except Exception as e:
                error = f"An error occurred during summarization: {e}"
    return render_template("index.html", summary=summary, original_text=original_text, error=error)

if __name__ == "__main__":
    app.run(debug=True)
