import csv
import numpy as np
import subprocess
import time
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity



def load_data(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

def compute_statistics(data):
    total_movies = set()
    total_users = set()
    ratings_count = {str(i): 0 for i in range(1, 6)}

    user_4321_stats = {'count': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, 'average': 0}
    movie_3_stats = {'count': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, 'average': 0}

    total_rating = 0

    for movie, user, rating, _ in data:
        total_movies.add(movie)
        total_users.add(user)

        ratings_count[rating] += 1
        total_rating += float(rating)

        if user == "4321":
            user_4321_stats['count'] += 1
            user_4321_stats[rating] += 1
            user_4321_stats['average'] += float(rating)

        if movie == "3":
            movie_3_stats['count'] += 1
            movie_3_stats[rating] += 1
            movie_3_stats['average'] += float(rating)

    # Compute the average
    user_4321_stats['average'] /= user_4321_stats['count']
    movie_3_stats['average'] /= movie_3_stats['count']

    return len(total_movies), len(total_users), ratings_count, total_rating/len(data), user_4321_stats, movie_3_stats


def create_user_movie_matrix(data):
    user_ids = [int(row[1]) for row in data]
    movie_ids = [int(row[0]) for row in data]
    ratings = [float(row[2]) for row in data]
    num_users = max(user_ids) + 1
    num_movies = max(movie_ids) + 1
    matrix = csr_matrix((ratings, (user_ids, movie_ids)), shape=(num_users, num_movies))
    return matrix


def get_top_k_nearest_neighbors(matrix, idx, metric="dot", k=5):
    if metric == "dot":
        similarities = matrix.dot(matrix[idx].T).toarray().flatten()
    elif metric == "cosine":
        # Using scikit-learn's cosine_similarity function
        similarities = cosine_similarity(matrix, matrix[idx].reshape(1, -1)).flatten()
    similarities[idx] = -np.inf
    nearest_neighbors = np.argsort(similarities)[-k:][::-1]
    return nearest_neighbors


def predict_user_user(matrix, user_idx, movie_idx, similarity_metric, method, k=5):
    nearest_users = get_top_k_nearest_neighbors(matrix, user_idx, metric=similarity_metric, k=k)
    ratings = matrix[nearest_users, movie_idx].toarray().flatten()
    if method == "mean":
        prediction = np.mean(ratings)
    elif method == "weighted_mean":
        similarities = matrix[nearest_users].dot(matrix[user_idx].T).toarray().flatten()
        prediction = np.dot(ratings, similarities) / (np.sum(np.abs(similarities)) + 1e-10)
    prediction += 3
    return min(max(prediction, 1), 5)  # Ensure rating is in [1, 5]


def get_movie_average_rating(matrix, movie_id):
    """Return the average rating of a movie, excluding zero ratings."""
    movie_ratings = matrix[:, movie_id]
    # If there are no ratings for the movie, return a default value (e.g., 3).
    if movie_ratings.nnz == 0:
        return 3.0  # Or whatever default value you deem appropriate
    
    avg_rating_imputed = movie_ratings.data.mean()
    avg_rating = avg_rating_imputed + 3  # Add back the 3 that was subtracted during imputation
    return avg_rating


def generate_predictions_for_dev_set(dev_file, matrix, similarity_metric, method, k=5):
    predictions = []
    dev_data = load_data(dev_file)
    for movie, user in dev_data:
        if matrix[int(user), :].nnz == 0:  # Check if user vector is all zeros
            predictions.append(get_movie_average_rating(matrix, int(movie)))
        else:
            user_idx, movie_idx = int(user), int(movie)
            pred_rating = predict_user_user(matrix, user_idx, movie_idx, similarity_metric, method, k)
            predictions.append(pred_rating)
    return predictions



def create_user_movie_csc_matrix(data):
    user_ids = [int(row[1]) for row in data]
    movie_ids = [int(row[0]) for row in data]
    ratings = [float(row[2]) for row in data]
    num_users = max(user_ids) + 1
    num_movies = max(movie_ids) + 1
    matrix = csr_matrix((ratings, (user_ids, movie_ids)), shape=(num_users, num_movies))
    return matrix.tocsc()

def compute_association_matrix(matrix_csc, metric="dot"):
    if metric == "dot":
        return matrix_csc.T.dot(matrix_csc)
    elif metric == "cosine":
        normalized_matrix = matrix_csc.copy().astype(np.float64)
        # Normalize the columns
        col_norms = np.array(np.sqrt(normalized_matrix.power(2).sum(axis=0)))
        col_indices = np.where(col_norms != 0)
        normalized_matrix[:, col_indices] /= col_norms[col_indices]
        return normalized_matrix.T.dot(normalized_matrix)

def predict_movie_movie(matrix_csc, association_matrix, user_idx, movie_idx, method, k=5):
    nearest_movies = get_top_k_nearest_neighbors(association_matrix, movie_idx, k=k)
    user_ratings = matrix_csc[user_idx, nearest_movies].toarray().flatten()
    
    if method == "mean":
        prediction = np.mean(user_ratings)
    elif method == "weighted_mean":
        similarities = association_matrix[movie_idx, nearest_movies].toarray().flatten()
        prediction = np.dot(user_ratings, similarities) / (np.sum(np.abs(similarities)) + 1e-10)
    
    return min(max(prediction, 1), 5)

def generate_movie_movie_predictions_for_dev_set(dev_file, matrix_csc, association_matrix, method, k=5):
    predictions = []
    dev_data = load_data(dev_file)
    for movie, user in dev_data:
        user_idx, movie_idx = int(user), int(movie)
        pred_rating = predict_movie_movie(matrix_csc, association_matrix, user_idx, movie_idx, method, k)
        predictions.append(pred_rating)
    return predictions


def evaluate_predictions(predictions_file):
    result = subprocess.run(["python", "eval/eval_rmse.py", "eval/dev.golden", predictions_file], capture_output=True, text=True)
    output = result.stdout.strip()
    try:
        rmse = float(output)
        return rmse
    except ValueError:
        print(f"Unexpected output from evaluation script: {output}")
        return None


def main():
    data = load_data("src/data/train.csv")
    total_movies, total_users, ratings_count, avg_rating, user_4321_stats, movie_3_stats = compute_statistics(data)

    print(f"Total movies: {total_movies}")
    print(f"Total users: {total_users}")
    print(f"Number of times any movie was rated '1': {ratings_count['1']}")
    print(f"Number of times any movie was rated '3': {ratings_count['3']}")
    print(f"Number of times any movie was rated '5': {ratings_count['5']}")
    print(f"Average movie rating across all users and movies: {avg_rating:.2f}")
    
    print("\nFor user ID 4321:")
    print(f"Number of movies rated: {user_4321_stats['count']}")
    print(f"Number of times the user gave a '1' rating: {user_4321_stats['1']}")
    print(f"Number of times the user gave a '3' rating: {user_4321_stats['3']}")
    print(f"Number of times the user gave a '5' rating: {user_4321_stats['5']}")
    print(f"Average movie rating for this user: {user_4321_stats['average']:.2f}")

    print("\nFor movie ID 3:")
    print(f"Number of users rating this movie: {movie_3_stats['count']}")
    print(f"Number of times the user gave a '1' rating: {movie_3_stats['1']}")
    print(f"Number of times the user gave a '3' rating: {movie_3_stats['3']}")
    print(f"Number of times the user gave a '5' rating: {movie_3_stats['5']}")
    print(f"Average rating for this movie: {movie_3_stats['average']:.2f}")


    user_movie_matrix = create_user_movie_matrix(data)
    user_movie_matrix.data -= 3

    # Compute top-5 nearest neighbors for user 4321
    user_4321_dot_nn = get_top_k_nearest_neighbors(user_movie_matrix, 4321, metric="dot")
    user_4321_cosine_nn = get_top_k_nearest_neighbors(user_movie_matrix, 4321, metric="cosine")

    # Compute top-5 nearest neighbors for movie 3 (using transpose for movie-based CF)
    movie_3_dot_nn = get_top_k_nearest_neighbors(user_movie_matrix.T, 3, metric="dot")
    movie_3_cosine_nn = get_top_k_nearest_neighbors(user_movie_matrix.T, 3, metric="cosine")

    print("Top 5 NNs of user 4321 (Dot Product):", user_4321_dot_nn)
    print("Top 5 NNs of user 4321 (Cosine Similarity):", user_4321_cosine_nn)
    print("Top 5 NNs of movie 3 (Dot Product):", movie_3_dot_nn)
    print("Top 5 NNs of movie 3 (Cosine Similarity):", movie_3_cosine_nn)


    results = []  # Store results (RMSE, Runtime) for each combination

    methods = ["mean", "weighted_mean"]
    metrics = ["dot", "cosine"]
    k_values = [10, 100, 500]

    # Printing header for the results
    print("Rating Method\tSimilarity Metric\tK\tRMSE\tRuntime(sec)")

    for method in methods:
        for metric in metrics:
            for k in k_values:
                
                # Skipping weighted dot product combination
                if method == "weighted_mean" and metric == "dot":
                    continue
                
                start_time = time.time()
                
                predictions = generate_predictions_for_dev_set("src/data/dev.csv", user_movie_matrix, metric, method, k)
                runtime = time.time() - start_time
                
                file_name = f"predictions_{method}_{metric}_k{k}.txt"
                with open(file_name, "w") as f:
                    for pred in predictions:
                        f.write(f"{pred}\n")
                
                rmse = evaluate_predictions(file_name)
                
                # Printing results right after the computation for the current combination
                print(f"{method}\t{metric}\t{k}\t{rmse:.4f}\t{runtime:.2f}")
                
                results.append({
                    "Rating Method": method,
                    "Similarity Metric": metric,
                    "K": k,
                    "RMSE": rmse,
                    "Runtime (sec)": runtime
                })


    # Printing results in the desired format
    print("Rating Method\tSimilarity Metric\tK\tRMSE\tRuntime(sec)")
    for result in results:
        print(f"{result['Rating Method']}\t{result['Similarity Metric']}\t{result['K']}\t{result['RMSE']}\t{result['Runtime (sec)']:.2f}")



if __name__ == "__main__":
    main()
