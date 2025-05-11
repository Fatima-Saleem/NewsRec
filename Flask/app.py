from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import torch
import os
import torch.nn as nn

app = Flask(__name__)

# Define StudentGenerator
class StudentGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_size):
        super(StudentGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 60)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + 60, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_size),
            nn.Tanh()        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels)
        input_data = torch.cat((noise, label_embed), dim=1)
        return self.model(input_data)

# Load models and data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent_dim = 200
num_classes = 26
output_size=21824
# Load Student Model
student_generator=StudentGenerator(latent_dim, num_classes,output_size).to(device) 
#student_generator = torch.load('ackd_student_generator.pth', map_location=device).to(device)
student_generator.load_state_dict(torch.load('ackd_best_student_generator7.pth'))
student_generator.eval()

interaction_matrix = pd.read_csv('interaction_matrix_with_embeddings.csv', index_col=0)
news_df_new = pd.read_csv('news_df_new.csv')
new_user = pd.read_csv('newUsers.csv')
# Category to label mapping
category_to_label = {
    'lifestyle': 0, 'health': 1, 'news': 2, 'sports': 3, 'weather': 4,
    'entertainment': 5, 'autos': 6, 'travel': 7, 'foodanddrink': 8,
    'tv': 9, 'finance': 10, 'movies': 11, 'video': 12, 'music': 13,
    'kids': 14, 'middleeast': 15, 'northamerica': 16
}

MAX_USER_ID_LENGTH = 20  

@app.route("/", methods=["GET", "POST"])
def home():
    error_message = None

    if request.method == "POST":
        user_id = request.form.get("user_id", "").strip()

        if not user_id:
            error_message = "User ID cannot be empty."
        elif not user_id.isalnum():
            error_message = "Invalid user ID. Only letters and numbers are allowed."
        elif len(user_id) > MAX_USER_ID_LENGTH:
            error_message = f"User ID is too long. Maximum {MAX_USER_ID_LENGTH} characters allowed."
        else:
            if user_id in interaction_matrix.index or user_id in new_user['UserID'].values:
                return redirect(url_for('recommend', user_id=user_id))
            else:
                return redirect(url_for('select_categories', user_id=user_id))

    return render_template("home.html", error_message=error_message)


# Route: Select Categories
@app.route("/select_categories/<user_id>", methods=["GET", "POST"])
def select_categories(user_id):
    if request.method == "POST":
        selected_categories = request.form.getlist('categories')
        if selected_categories:
            new_user_data = {'UserID': user_id, 'Categories': ', '.join(selected_categories)}
            pd.DataFrame([new_user_data]).to_csv(
                'newUsers.csv', mode='a', index=False, header=not os.path.exists('newUsers.csv')
            )
        return redirect(url_for('recommend', user_id=user_id, categories=','.join(selected_categories)))
    return render_template("select_categories.html", categories=list(category_to_label.keys()))

@app.route("/recommend/<user_id>", methods=["GET"])
def recommend(user_id):
    categories = request.args.get("categories", None)
    recommendations = []

    try:
        # Generate recommendations based on categories
        if categories:
            selected_labels = [category_to_label.get(cat.strip(), -1) for cat in categories.split(',')]
            selected_labels = [label for label in selected_labels if label != -1]

            if not selected_labels:
                return jsonify({"error": "No valid categories found"}), 400

            for label in selected_labels:
                noise = torch.randn(1, latent_dim).to(device)
                cluster_label = torch.tensor([label]).to(device)
                predicted_interactions = (
                    student_generator(noise, cluster_label).cpu().detach().numpy().flatten()
                )
                recommendations.append(pd.DataFrame({
                    'NewsID': interaction_matrix.columns,
                    'PredictedScore': predicted_interactions
                }))
            articles_with_scores = pd.concat(recommendations, ignore_index=True).drop_duplicates()
        else:
            # Generate user-specific recommendations
            noise = torch.randn(1, latent_dim).to(device)
            user_label = torch.tensor([hash(user_id) % num_classes]).to(device)
            predicted_interactions = (
                student_generator(noise, user_label).cpu().detach().numpy().flatten()
            )
            articles_with_scores = pd.DataFrame({
                'NewsID': interaction_matrix.columns,
                'PredictedScore': predicted_interactions
            })

        # Merge with article details and sort
        articles_with_scores = articles_with_scores.sort_values(by="PredictedScore", ascending=False)
        articles_with_details = articles_with_scores.merge(
            news_df_new, on="NewsID", how="inner"
        )

        # Filter and limit results
        valid_articles = articles_with_details.dropna(subset=["Title", "Abstract"])
        valid_articles = valid_articles[valid_articles["PredictedScore"] > 0]
        top_articles = valid_articles.head(20).to_dict(orient="records")

        # Pass user_id to template
        return render_template("articles.html", user_id=user_id, articles=top_articles)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/logout")
def logout():
    # No session is used, just redirect to homepage
    return redirect(url_for('home'))
if __name__ == "__main__":
    app.run(debug=True)
