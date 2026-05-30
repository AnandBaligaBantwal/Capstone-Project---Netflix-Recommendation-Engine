# Netflix Recommendation Engine (Capstone Project)

An end-to-end recommendation engine built using the classic Netflix Prize dataset structure. This project parses raw user-movie interaction logs, performs demographic genre interest profiling, implements a memory-efficient sparse Collaborative Filtering system via Item-Based Cosine Similarity, and outputs personalized watchlists optimized by genre constraints.

## 🚀 Features

* **Custom Data Parser:** Processes raw text data structured with episodic movie header rows (`movieId:`) followed by user rating logs, handling large-scale sparse transactional files efficiently.
* **User Interest Profiling:** Dynamically explodes multi-genre movie labels to map micro-preferences and average baseline ratings per user across individual categories.
* **Optimized Matrix Operations:** Utilizes Compressed Sparse Row (`scipy.sparse.csr_matrix`) indexing to manage high-dimensional user-item interactions seamlessly without memory-related crashes.
* **Item-Item Collaborative Filtering:** Implements a pairwise `cosine_similarity` model over movie vectors to score unobserved items.
* **Hybrid Genre Constraint Recommender:** Generates localized watchlists by parsing the top prediction vectors and isolating the absolute highest-scoring film per distinct genre for a diversified user dashboard.

---

## 🛠️ Project Structure

```text
├── Netflix Recommendation Engine -Captstone Project.py  # Main capstone project pipeline
├── movie_title.csv                                      # Movie metadata containing IDs, Titles, and Genres
├── combined_data_1.txt                                  # Raw user transaction history (MovieID, UserID, Rating)
├── processed_data.csv                                   # Combined transactional matrix logs
└── README.md                                            # Documentation
