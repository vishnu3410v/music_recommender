# 🎵 Music Collaborative Filtering Recommender System

This project implements a collaborative filtering recommender system using **implicit feedback data** from users' listening history. It uses the [`implicit`](https://github.com/benfred/implicit) library and matrix factorization techniques (ALS) to recommend music artists based on user preferences.

---

## 🔍 Overview

- ✅ Built using the **Last.fm dataset**
- ✅ Utilizes **ALS (Alternating Least Squares)** for collaborative filtering
- ✅ Returns top-N artist recommendations for any user
- ✅ Handles artist name lookup via a dedicated utility
- ✅ Modular, clean, and extensible Python codebase

---

## 📁 Project Structure
```
music-collaborative-filtering/
│
├── musiccollaborativefiltering/
│ ├── recommender.py # ImplicitRecommender class (ALS-based)
│ └── data.py # Data loading and ArtistRetriever class
│
├── lastfmdata/ # Sample user-artist and artist files (place here)
│ ├── user_artists_sample.dat
│ └── artists_sample.dat
│
├── run.py # Script to train model and get recommendations
├── requirements.txt # Dependencies
├── .gitignore # Files to ignore
└── README.md # This file
```

---

## 📦 Installation

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/music-collaborative-filtering.git
cd music-collaborative-filtering
```
Install dependencies:

```bash
pip install -r requirements.txt
```
## 📂 Dataset Format
Place your dataset files in the `lastfmdata/` directory:

`user_artists.dat`
Tab-separated file with:

```nginx
userID    artistID    weight
```
`artists.dat`
Tab-separated file with:

```bash

id    name
```
## 🚀 Usage
Run the recommender system with:

```bash
python run.py
```
It will:

Load data

Fit the ALS model

Print top 5 artist recommendations for a given user (e.g., `user_id = 2`)

### 📊 Example Output
```yaml
Top 5 recommendations for user 2:

1. Coldplay — Score: 2.435
2. Radiohead — Score: 2.302
3. Daft Punk — Score: 2.178
4. Muse — Score: 1.962
5. The Beatles — Score: 1.893
```
🧠 Tech Stack
`Python 3.8+`

`pandas`

`scipy`

`implicit`(ALS model)

`logging` (for traceability)


### ✅ Future Enhancements

🔄 Add TF-IDF or BM25 weighting

🧪 Add evaluation metrics (MAP@K, precision@K)

🌐 Add a Flask or Streamlit front-end

🧊 Handle cold-start users with fallback logic


### 📄 License
This project is open-source under the MIT License.


# 🙋‍♂️ Author
### Vishnu Munukutla
##### B.Tech in AI & ML @ Dayananda Sagar University

Passionate about building ML systems for real-world impact.

