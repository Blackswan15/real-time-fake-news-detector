from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

app = FastAPI()

# Load and train model
df1 = pd.read_csv("data/fake_or_real_news.csv")[['title', 'label']]
df2 = pd.read_csv("data/True.csv")
df3 = pd.read_csv("data/Fake.csv")


df2['label'] = 'REAL'
df3['label'] = 'FAKE'

combined_df = pd.concat([
    df1,
    df2[['title', 'label']],
    df3[['title', 'label']]
], ignore_index=True)

fake_df = combined_df[combined_df['label'] == 'FAKE']
real_df = combined_df[combined_df['label'] == 'REAL']
min_len = min(len(fake_df), len(real_df))
balanced_df = pd.concat([
    fake_df.sample(n=min_len, random_state=42),
    real_df.sample(n=min_len, random_state=42)
]).sample(frac=1, random_state=42)

X = balanced_df['title']
y = balanced_df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

trainx, testx, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
trainx_vector = vectorizer.fit_transform(trainx)
clf = LinearSVC()
clf.fit(trainx_vector, ytrain)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, headline: str = Form(...)):
    vector = vectorizer.transform([headline])
    result = clf.predict(vector)[0]
    conclusion = "✅ REAL News" if result == 0 else "❌ FAKE News"
    return templates.TemplateResponse("form.html", {"request": request, "result": conclusion, "headline": headline})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
