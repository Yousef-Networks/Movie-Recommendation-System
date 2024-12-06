import pandas as pd
import streamlit as st
import json
import re
import requests
import time
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_image = image_to_base64("C:\\Users\\HP\\Downloads\\samuel-regan-asante-wMkaMXTJjlQ-unsplash.jpg")


# TMDb API key
API_KEY = "663323692c0edc1c6f1a0f4a7781a336"  # Replace with your TMDb API key

# Function to get movie info from TMDb API using TMDb ID
def get_movie_info_tmdb(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        return poster_path
    else:
        return None

# Collaborative Filtering
movies_cf = pd.read_csv("C:\\Users\\HP\\Desktop\\Mmovies\\movies.csv")
ratings = pd.read_csv("C:\\Users\\HP\\Desktop\\Mmovies\\ratings.csv")
links = pd.read_csv("C:\\Users\\HP\\Desktop\\Mmovies\\links.csv")
tags = pd.read_csv("C:\\Users\\HP\\Desktop\\Mmovies\\tags.csv")

user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_movie_ratings)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

def get_collaborative_recommendation(user_id):
    n = 5  # Number of movies to recommend
    if user_id not in user_similarity_df.index:
        return "User ID not found in similarity matrix."
    
    user_ratings = user_movie_ratings.loc[user_id]
    movies_not_rated = user_ratings[user_ratings == 0].index
    
    if len(movies_not_rated) > 0:
        similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).index
        recommended_movies = []
        predicted_ratings = pd.Series(index=user_movie_ratings.columns, data=0)
        similarity_sum = 0
        for user in similar_users:
            user_ratings = user_movie_ratings.loc[user]
            unrated_movies = user_ratings[user_ratings == 0].index
            new_recommendations = unrated_movies.difference(recommended_movies)
            recommended_movies.extend(new_recommendations)
            similarity = user_similarity_df.loc[user_id, user]
            similarity_sum += similarity
            predicted_ratings += similarity * user_movie_ratings.loc[user]
            print(f"User {user}'s ratings: {user_movie_ratings.loc[user]}")
            print(f"Similarity with user {user}: {similarity}")
            print(f"Predicted ratings after adding user {user}: {predicted_ratings}")

            print(f"Similarity sum: {similarity_sum}")
            print(f"Final predicted ratings: {predicted_ratings}")
            if len(recommended_movies) >= n:
                break
        
        # Normalize predicted ratings
        predicted_ratings /= similarity_sum
        
        recommended_movies_info = movies_cf[movies_cf['movieId'].isin(recommended_movies)][:n][['movieId', 'title']]
        recommended_movies_info['predicted_rating'] = predicted_ratings[recommended_movies_info['movieId']].values
        return recommended_movies_info
    else:
        return "All movies have been rated by the user."




# Content-Based Filtering
df1 = pd.read_csv("C:\\Users\\HP\\Desktop\\Mmovies\\tmdb_5000_movies.csv")
df2 = pd.read_csv("C:\\Users\\HP\\Desktop\\Mmovies\\tmdb_5000_credits.csv")
df = pd.merge(df1, df2, on='title')
df.dropna(inplace=True)

def extract_names(json_str):
    items = json.loads(json_str)
    names = [item["name"].replace(" ", "") for item in items]
    return names

# Extracting features from JSON columns
df["genres"] = df["genres"].apply(extract_names)
df["keywords"] = df["keywords"].apply(extract_names)
df["cast"] = df["cast"].apply(extract_names)
df["crew"] = df["crew"].apply(extract_names)
df["release_date"] = df["release_date"].apply(lambda x: [str(x)[:4]])

# Combining text features
df["text"] = (df["release_date"].astype(str) + " " +
              df["overview"].astype(str) + " " +
              df["genres"].astype(str) + " " +
              df["keywords"].astype(str) + " " +
              df["cast"].astype(str) + " " +
              df["crew"].astype(str) + " " +
              df["tagline"].astype(str) + " " +
              df["original_title"].astype(str))

# Cleaning text
def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

df["text"] = df["text"].apply(clean_text)

# Stemming text
stemmer = PorterStemmer()
df["text"] = df["text"].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Vectorizing text
cv = CountVectorizer(max_features=10000, stop_words="english")
data = cv.fit_transform(df["text"]).toarray()

# Cosine similarity
similarity = cosine_similarity(data)
sim_df = pd.DataFrame(similarity, index=df['title'], columns=df['title'])

def make_recommendation(movie_name):
    similar_scores = sim_df[movie_name].sort_values(ascending=False)[1:11]
    recommended_movies = similar_scores.index.tolist()
    tmdb_ids = [df[df['title'] == movie]['id'].values[0] for movie in recommended_movies]  # Fetching tmdbId
    return list(zip(recommended_movies, tmdb_ids))

def fetch_tmdb_details(tmdb_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=663323692c0edc1c6f1a0f4a7781a336"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            return poster_path
        else:
            return None
    except Exception as e:
        st.error("Error fetching movie details")
        st.error(e)
        return None
# Set page configuration
st.set_page_config(page_title="Aflamak", page_icon=":film_frames:", layout="wide")

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None



# Streamlit App
# Add CSS for styling with fixed background image for all pages
st.markdown(f"""
    <style>
        body {{
            background-image: url('data:image/png;base64,{background_image}');
            background-size: contain;
            background-repeat: repeat;
            background-attachment: fixed; /* Keeps the background fixed while scrolling */
            font-family: Arial, sans-serif;
            margin: 0; /* Removes default margin */
            padding: 0; /* Removes default padding */
        }}
        /* Style for the entire app container to ensure background image visibility */
        .stApp {{
            background-color: transparent; /* Make the Streamlit app background transparent */
        }}
        /* Styling for Sign In container */
        .signin-container {{
            background-color: rgba(255, 255, 255, 0.8); /* Add some opacity for better readability */
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 100px auto; /* Centers the container */
            max-width: 400px;
        }}
        .stButton > button {{
            background-color: #007bff;
            color: #fff;
            border-radius: 5px;
        }}
        .stTextInput > div > div > input {{
            border-color: #007bff;
            border-radius: 5px;
        }}
        /* Styling for .st-emotion-cache-13k62yr */
        .st-emotion-cache-13k62yr {{
            position: absolute;
            color: rgb(250, 250, 250);
            inset: 0px;
            color-scheme: dark;
            overflow: hidden;
        }}
    </style>
""", unsafe_allow_html=True)

 #st.image("C:\\Users\\HP\\Pictures\\Screenshots\\Screenshot (54).png", use_column_width=True)  # 
 # Define color palette
# Define color palette
primary_color = "#4D5255"  # Dark blue
secondary_color = "#3498db"  # Light blue
text_color = "#333"
background_color = "#ecf0f1"  # Light grey background
# Page layout
page = st.sidebar.radio("", ["Home", "About Us", "Sign In", "Recommender","History","Create Account","Search&Rate"], index=0)


if page == "Sign In":
    st.subheader("Sign In")
    
    # Input field for existing user ID
    existing_user_id = st.number_input("Enter Existing User ID", min_value=1)
    
    # Check if the existing user ID exists
    if existing_user_id in user_movie_ratings.index:
        if st.button("Sign In"):
            st.session_state.user_id = existing_user_id
            st.experimental_rerun()  # Refresh the page to reflect changes
            st.write("")  # Empty space for separation after successful sign-in
    else:
        st.warning("User ID does not exist.")

    # Display success message after successful sign-in
    if "user_id" in st.session_state:
        st.success(f"Welcome, User {st.session_state.user_id}! You have successfully signed in.")


    
    
    













  




# Check for user authentication
if page == "Recommender" and st.session_state.user_id is None:
    st.warning("Please sign in to access this feature.")
    st.stop()  # Stop execution if user is not signed in

  



     


elif page == "Home":
    st.markdown("""
    <div class="home-container" style="background-color: #222F20; padding: 50px; border-radius: 15px;">
        <h1 style="text-align: center; color: #E3B786; font-weight: bold; font-size: 3em; margin-bottom: 30px;">🎬 Welcome to Aflamak - The Perfect Guide for your next Movie! 🍿</h1>
        <p style="text-align: center; color: #E3B786; font-size: 1.5em;">Discover and explore movies tailored just for you. Whether you're in the mood for the latest blockbusters or hidden gems, Aflamak has got you covered.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 100px; margin-bottom: 80px;">
        <h2 style="color: #ecf0f1; font-weight: bold; font-size: 2.8em; margin-bottom: 50px;">🌟 Big Hit Movies 🌟</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display popular movies in a horizontal flexbox layout
    st.markdown("""
<style>
                
    .movie-container {{
        display: flex;
        flex-direction: row;
        overflow-x: auto; 
        margin-bottom: 1rem;       
        padding: 30px ;
        gap: 50px; 
    }}
                
    .movie-card {{
        background-color: #222F20;
        border-radius: 30px;
        padding: 50px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: transform .3s;
        overflow: hidden;
        width: 1600px; 
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }}
    .movie-card:hover {{
        transform: translateY(-10px);
    }}
    .movie-image {{
        border-radius: 20px;
        margin-right: 40px; 
        width: 250px;
        height: auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform .3s;
    }}
    .movie-image:hover {{
        transform: scale(1.05);
    }}
    .movie-details {{
        flex-grow: 1;
    }}
    .movie-caption {{
        font-size: 24px;
        text-align: left;
        color: #E3B786;
        margin-bottom: 15px;
        font-weight: bold;
    }}
    .movie-rating {{
        font-size: 22px;
        color: #E3B786;
        text-align: left;
        margin-bottom: 10px;
    }}
    .movie-overview {{
        font-size: 20px;
        color: #E3B786;
        text-align: left;
        margin-bottom: 20px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 600px;
    }}
    .tmdb-link {{
        text-align: left;
        margin-top: 25px;
        color: #fff;
        font-size: 18px;
        transition: color .3s;
    }}
    .tmdb-link:hover {{
        color: #f39c12;
    }}
</style>
""".format(card_color="#2c3e50"), unsafe_allow_html=True)

    popular_movies = ["Inception", "The Dark Knight", "The Shawshank Redemption", "Pulp Fiction", "Fight Club"]
    
    st.markdown('<div class="movie-container">', unsafe_allow_html=True)
    
    for movie in popular_movies:
        movie_row = df[df['original_title'] == movie]
        if not movie_row.empty:
            tmdb_id = movie_row['id'].values[0]
            poster_path = fetch_tmdb_details(tmdb_id)
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/original{poster_path}"
                rating = movie_row['vote_average'].values[0]
                overview = movie_row['overview'].values[0]
                
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster_url}" class="movie-image">
                        <div class="movie-details">
                            <div class="movie-caption">🎥 <strong>{movie}</strong></div>
                            <div class="movie-rating">⭐ {rating}/10</div>
                            <div class="movie-overview">{overview[:150]}</div>
                            <div class="tmdb-link"><a href="https://www.themoviedb.org/movie/{tmdb_id}/" target="_blank">🔗 TMDb Link</a></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


elif page == "Create Account":
    new_user_id = st.number_input("Enter New User ID", min_value=1)
    submit_button = st.button("Submit")
    
    if submit_button:
        # Check if the new user ID already exists
        if new_user_id in ratings['userId'].unique():
            st.warning("User ID already exists. Please enter a different User ID.")
        else:
            # Define default values for movie ID and rating
            default_movie_id = 1
            default_rating = 0.0
            
            # Create a new row for the new user with default values
            new_user_row = {
                'userId': new_user_id,
                'movieId': default_movie_id,
                'rating': default_rating,
                'timestamp': int(time.time())  # Current timestamp
            }
            
            # Convert the new user row to a DataFrame
            new_user_df = pd.DataFrame([new_user_row])
            
            # Concatenate the new user DataFrame with the existing ratings DataFrame
            ratings = pd.concat([ratings, new_user_df], ignore_index=True)
            
            # Save the updated ratings dataset to CSV
            ratings.to_csv("C:\\Users\\HP\\Desktop\\Mmovies\\ratings.csv", index=False)
            
            # Confirm that the new user row is added
            st.write("New user row:", new_user_row)
            
            st.success(f"New User {new_user_id} created successfully!")
elif page == "About Us":
   
    
    st.markdown("""
    <div style="padding: 50px; background-color: #222F20; border-radius: 15px; margin-top: 50px; color: #ecf0f1;">
        <h2 style="font-size: 2.5em; color: #E3B786; margin-bottom: 30px;">Our Mission</h2>
        <p style="font-size: 1.3em;">
        Aflamak is a personalized movie recommendation platform designed to help users discover movies tailored to their tastes and preferences. Our mission is to enhance your movie-watching experience by providing you with movie recommendations that resonate with you.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 50px; background-color: #222F20; border-radius: 15px; margin-top: 50px; color: #ecf0f1;">
        <h2 style="font-size: 2.5em; color: #E3B786; margin-bottom: 30px;">Features</h2>
        <ul style="font-size: 1.3em; padding-left: 20px;">
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">🎬</span> Personalized movie recommendations</li>
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">🔍</span> Search and explore movies</li>
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">📊</span> Collaborative and content-based filtering</li>
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">🌐</span> Integration with TMDb API</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 50px; background-color: #222F20; border-radius: 15px; margin-top: 50px; color: #ecf0f1;">
        <h2 style="font-size: 2.5em; color: #E3B786; margin-bottom: 30px;">Technologies Used</h2>
        <ul style="font-size: 1.3em; padding-left: 20px;">
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">🔧</span> Python</li>
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">🔧</span> Streamlit</li>
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">🔧</span> Pandas</li>
            <li style="margin-bottom: 20px;"><span style="color: #f39c12; font-size: 1.8em; margin-right: 15px;">🔧</span> Scikit-learn</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 50px; background-color: #222F20; border-radius: 15px; margin-top: 50px; color: #ecf0f1;">
        <h2 style="font-size: 2.5em; color: #E3B786; margin-bottom: 30px;">Motivation</h2>
        <p style="font-size: 1.3em;">
        As a movie enthusiast, I often found it challenging to discover new movies that align with my interests. Aflamak was born out of this frustration with the goal of making movie recommendations more personalized and enjoyable.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 50px; background-color: #222F20; border-radius: 15px; margin-top: 50px; color: #ecf0f1;">
        <h2 style="font-size: 2.5em; color: #E3B786; margin-bottom: 30px;">Contact Me</h2>
        <p style="font-size: 1.3em;">
        I'd love to hear your feedback and suggestions! Feel free to reach out to me.
        </p>
        <a href="mailto:youssef.abdelhaleem@gmail.com" style="font-size: 1.3em; color: #fff; text-decoration: none;">youssef.abdelhaleem@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)

elif page == "History":
    st.sidebar.subheader("History")

    # Check if the user is signed in
    if st.session_state.user_id is None:
        st.warning("Please sign in to view your history.")
        st.stop()  # Stop execution if user is not signed in

    # Fetch the user's ratings history
    user_history = ratings[ratings['userId'] == st.session_state.user_id]

    # Check if the user has provided any ratings
    if user_history.empty:
        st.info("You haven't provided any ratings yet.")
    else:
        # Merge user ratings with movie titles
        user_history = pd.merge(user_history, movies_cf, on='movieId', how='inner')

        st.write("Your Rating History:")
        st.write(user_history[['title', 'rating']])

        # Create an expander to group the ratings and sliders (optional)
        with st.expander("Edit Ratings (Expander)", expanded=True):
            for index, row in user_history.iterrows():
                # Create a container for each rating slider and its text
                rating_container = st.container()

                # Use methods to add content within the container
                rating_container.write(f"Rate for {row['title']}:")
                new_rating = rating_container.slider("", min_value=0.0, max_value=5.0, value=row['rating'], step=0.1, format="%.1f", key=str(index))

                # Update the rating in the original DataFrame if the user provides a new rating
                if new_rating != row['rating']:
                    movie_id = row['movieId']
                    rating_index = ratings[(ratings['userId'] == st.session_state.user_id) & (ratings['movieId'] == movie_id)].index
                    if not rating_index.empty:
                        ratings.loc[rating_index, 'rating'] = new_rating
                        st.success(f"Rating of {new_rating}/5 updated for {row['title']}.")
                    else:
                        st.error("Could not find the rating to update.")

        # Save the updated ratings to the dataset
        ratings.to_csv('ratings.csv', index=False)  # Assuming the dataset is stored in a CSV file















        
elif page == "Search&Rate":
    st.title("Search & Rate")

    # Check if the user is signed in
    if st.session_state.user_id is None:
        st.warning("Please sign in to access this feature.")
        st.stop()  # Stop execution if user is not signed in

    # Sidebar for selecting genre
    genres = movies_cf['genres'].str.split('|', expand=True).stack().dropna().unique()
    selected_genre = st.sidebar.selectbox("Select a genre", genres)

    # Filter movies based on selected genre
    filtered_movies = movies_cf[movies_cf['genres'].str.contains(selected_genre, na=False)]

    # Display filtered movies
    st.write(f"Showing {len(filtered_movies)} movies in the {selected_genre} genre:")
    st.write(filtered_movies[['title', 'genres']])  # Removed 'overview' and 'vote_average'

    # Allow users to rate movies
    st.header("Rate Movies")
    movie_title = st.text_input("Enter the title of the movie you want to rate:")
    rating = st.slider("Rate this movie (0-5)", min_value=0, max_value=5, step=1, value=5)

    if st.button("Submit Rating"):
        # Update rating in the DataFrame
        movie_index = filtered_movies[filtered_movies['title'].str.lower() == movie_title.lower()].index
        if len(movie_index) > 0:
            # Add the user ID to the ratings DataFrame
            user_id = st.session_state.user_id
            new_rating_row = {'userId': user_id, 'movieId': filtered_movies.loc[movie_index, 'movieId'].iloc[0], 'rating': float(rating),'timestamp':int(time.time())}
            x=pd.DataFrame([new_rating_row])
            ratings=pd.concat([ratings,x],ignore_index=True)
            ratings.to_csv("C:\\Users\\HP\\Desktop\\Mmovies\\ratings.csv", index=False)
            st.success(f"Rating of {rating}/5 submitted for {movie_title}.")
        else:
            st.error("Movie not found. Please enter a valid movie title.")





elif page == "Recommender":
    st.sidebar.subheader("Recommendation Techniques")
    option = st.sidebar.radio("", ["Recommended For You", "Similar to what you want"])
    
    st.markdown("""
<style>
    .recommendation-header {
        color: #fff;
        font-weight: bold;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .movie-container {
        display: flex;
                
        flex-direction: row;
        flex-wrap: wrap;
        justify-content: flex-start;
        gap: 30px;
    }
    .movie-card {
        background-color: #222F20;
        border-radius: 30px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: transform .3s;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        margin-bottom:1rem;
        align-items: center;
        width: 250px;  /* Fixed width for each card */
    }
    .movie-card:hover {
        transform: translateY(-10px);
    }
    .movie-image {
        border-radius: 20px;
        margin-bottom: 10px; 
        width: 100%;  /* Full width for the image */
        height: auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform .3s;
    }
    .movie-image:hover {
        transform: scale(1.05);
    }
    .movie-details {
        text-align: center;
        margin-top: 10px;  /* Space between poster and title */
    }
    .movie-caption {
        font-size: 18px;
        color: #E3B786;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .movie-rating {
        color: #E3B786;  /* Change color to black */
    }
</style>
""", unsafe_allow_html=True)

    if option == "Recommended For You":
     if st.sidebar.button("Get Recommendations"):
        user_ratings = user_movie_ratings.loc[st.session_state.user_id].dropna()
        user_ratings = user_ratings[user_ratings != 0]  # Filter out zero-rated movies
        print("User ratings:")
        print(user_ratings)  # Debug statement
        rating_count = len(user_ratings)
        print("User rating count:", rating_count)  # Debug statement
        if rating_count < 5:
            st.warning("Please rate at least 5 movies to get recommendations.")
        else:
            recommendations = get_collaborative_recommendation(st.session_state.user_id)
            if not isinstance(recommendations, str):
                st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
                st.markdown('<h2 class="recommendation-header">Top 5 Recommendations based on your ratings:</h2>', unsafe_allow_html=True)
                
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                
                for index, row in recommendations.iterrows():
                    tmdb_id = links[links['movieId'] == row['movieId']]['tmdbId'].values[0]
                    poster_path = get_movie_info_tmdb(tmdb_id)
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/original{poster_path}"
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster_url}" class="movie-image">
                            <div class="movie-details">
                                <div class="movie-caption">🎥 <strong>{row['title']}</strong></div>
                                <a href="https://www.themoviedb.org/movie/{tmdb_id}/" target="_blank">🔗 TMDb Link</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(recommendations)


    elif option == "Similar to what you want":
        movie_name = st.sidebar.text_input("Enter a movie name", "The Hangover")
        if st.sidebar.button("Get Recommendations"):
            recommendations = make_recommendation(movie_name)
            st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
            st.markdown(f'<h2 class="recommendation-header">Top 10 Recommendations for {movie_name}:</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
            
            for i, (movie, tmdb_id) in enumerate(recommendations, 1):
                poster_path = fetch_tmdb_details(tmdb_id)
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/original{poster_path}"
                    st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster_url}" class="movie-image">
                        <div class="movie-details">
                            <div class="movie-caption">🎥 <strong>{movie}</strong></div>
                            <a href="https://www.themoviedb.org/movie/{tmdb_id}/" target="_blank">🔗 TMDb Link</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
