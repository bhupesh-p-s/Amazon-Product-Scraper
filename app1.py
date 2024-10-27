import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from textblob import TextBlob
from PIL import Image
import pytesseract

# Set up the path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust this path as needed

# Function to extract product details from a product listing
def get_product_details(product):
    try:
        title = product.find("span", attrs={'class': 'a-size-medium a-color-base a-text-normal'}).text.strip()
    except AttributeError:
        title = "Not Available"

    try:
        price = product.find("span", attrs={'class': 'a-price-whole'}).text.strip()
    except AttributeError:
        price = "Not Available"

    try:
        rating = product.find("span", attrs={'class': 'a-icon-alt'}).text.strip()
    except AttributeError:
        rating = "Not Available"

    try:
        review_count = product.find("span", attrs={'class': 'a-size-base'}).text.strip()
    except AttributeError:
        review_count = "Not Available"

    return title, price, rating, review_count

# Function to scrape a single page
def scrape_page(URL, headers):
    webpage = requests.get(URL, headers=headers)
    soup = BeautifulSoup(webpage.content, "html.parser")
    products = soup.find_all("div", attrs={'data-component-type': 's-search-result'})
    
    product_data = []
    for product in products:
        details = get_product_details(product)
        product_data.append(details)

    return product_data

# Main Streamlit application
def scraping():
    st.title("Product Scraper: Amazon")

    # Input for text or image
    input_type = st.radio("Select Input Type:", ("Text Input", "Image Upload"))

    if input_type == "Text Input":
        search_query = st.text_input("Enter product name (e.g., laptops, phones, etc.):")
    else:
        uploaded_image = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])
        search_query = ""
        if uploaded_image is not None:
            # Open the image for OCR
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Perform OCR on the image
            search_query = pytesseract.image_to_string(image).strip()
            st.success(f"Extracted Text: {search_query}")

    pages_to_scrape = st.number_input("Number of pages to scrape:", min_value=1, max_value=10, value=5)

    # Button for scraping
    if st.button('Scrape'):
        if not search_query:
            st.warning("Please provide a valid search query.")
            return

        with st.spinner("Scraping..."):
            # User agent and headers
            HEADERS = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
                'Accept-Language': 'en-US, en;q=0.9'
            }

            # Use the recognized search query
            base_url = f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}"  # Replace spaces with '+' for URL
            all_product_data = []

            for page_num in range(1, pages_to_scrape + 1):
                URL = f"{base_url}&page={page_num}"
                st.write(f"Scraping page {page_num}...")
                product_data = scrape_page(URL, HEADERS)
                all_product_data.extend(product_data)
                time.sleep(2)  # Be polite to avoid being blocked

            # Create DataFrame and store in session state
            df = pd.DataFrame(all_product_data, columns=["Product Title", "Product Price", "Product Rating", "Review Count"])
            st.session_state.df = df  # Store DataFrame in session state

            st.success("Scraping completed!")
            st.dataframe(df)

            # Extract brands
            def extract_brand_with_regex(title):
                match = re.match(r'^[A-Za-z]+', title)
                if match:
                    first_word = match.group(0).lower()
                    if first_word not in ["laptop", "laptops", "slim"]:
                        return match.group(0)
                return "Unknown"

            df['Extracted_Brand_Regex'] = df['Product Title'].apply(extract_brand_with_regex)
            unique_brands = df['Extracted_Brand_Regex'].unique()
            unique_brands = [brand for brand in unique_brands if brand != "Unknown"]

            # Store unique brands in session state
            st.session_state.unique_brands = unique_brands

            # Display unique brands
            st.write("Unique Brands Extracted:", unique_brands)

def clustering():
    # Load data from session state (assuming data was previously stored during scraping)
    if 'df' not in st.session_state or st.session_state.df.empty:
        st.warning("No data available for clustering. Please scrape data first.")
        return
    
    df = st.session_state.df

    # TF-IDF Vectorization
    df['text'] = df['Extracted_Brand_Regex'].fillna('') + ' ' + df['Review Count'].fillna('')
    df = df[df['text'].str.strip() != '']  # Remove rows with empty text

    if df['text'].empty:
        st.warning("No valid text data for TF-IDF vectorization.")
    else:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        st.write("TF-IDF matrix shape:", tfidf_matrix.shape)

        # Clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(tfidf_matrix)
        df['KMeans_Cluster'] = kmeans_labels

        st.write("KMeans Clustering Labels:")
        st.dataframe(df[['Extracted_Brand_Regex', 'KMeans_Cluster']].head())

        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg_clustering.fit_predict(tfidf_matrix.toarray())
        df['Agglomerative_Cluster'] = agg_labels

        st.write("Agglomerative Clustering Labels:")
        st.dataframe(df[['Extracted_Brand_Regex', 'Agglomerative_Cluster']].head())

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(tfidf_matrix.toarray())
        df['DBSCAN_Cluster'] = dbscan_labels
        
        st.write("DBSCAN Clustering Labels:")
        st.dataframe(df[['Extracted_Brand_Regex', 'DBSCAN_Cluster']].head())

        # Silhouette Score
        silhouette_kmeans = silhouette_score(tfidf_matrix, kmeans_labels)
        silhouette_agg = silhouette_score(tfidf_matrix, agg_labels)
        silhouette_dbscan = silhouette_score(tfidf_matrix, dbscan_labels) if len(set(dbscan_labels)) > 1 else None

        st.write(f"Silhouette Score - KMeans: {silhouette_kmeans:.3f}")
        st.write(f"Silhouette Score - Agglomerative: {silhouette_agg:.3f}")
        st.write(f"Silhouette Score - DBSCAN: {silhouette_dbscan:.3f}" if silhouette_dbscan else "Silhouette Score - DBSCAN: Not Applicable")

        # Calinski-Harabasz Index
        ch_kmeans = calinski_harabasz_score(tfidf_matrix.toarray(), kmeans_labels)
        ch_agg = calinski_harabasz_score(tfidf_matrix.toarray(), agg_labels)
        ch_dbscan = calinski_harabasz_score(tfidf_matrix.toarray(), dbscan_labels) if len(set(dbscan_labels)) > 1 else None

        st.write(f"Calinski-Harabasz Index - KMeans: {ch_kmeans:.3f}")
        st.write(f"Calinski-Harabasz Index - Agglomerative: {ch_agg:.3f}")
        st.write(f"Calinski-Harabasz Index - DBSCAN: {ch_dbscan:.3f}" if ch_dbscan else "Calinski-Harabasz Index - DBSCAN: Not Applicable")

        # Davies-Bouldin Index
        db_kmeans = davies_bouldin_score(tfidf_matrix.toarray(), kmeans_labels)
        db_agg = davies_bouldin_score(tfidf_matrix.toarray(), agg_labels)
        db_dbscan = davies_bouldin_score(tfidf_matrix.toarray(), dbscan_labels) if len(set(dbscan_labels)) > 1 else None

        st.write(f"Davies-Bouldin Index - KMeans: {db_kmeans:.3f}")
        st.write(f"Davies-Bouldin Index - Agglomerative: {db_agg:.3f}")
        st.write(f"Davies-Bouldin Index - DBSCAN: {db_dbscan:.3f}" if db_dbscan else "Davies-Bouldin Index - DBSCAN: Not Applicable")

        # Store the clustering results in session state for later analysis
        st.session_state.df = df

        # Summary display
        st.write("Final Clustering Summary:")
        st.dataframe(df[['Extracted_Brand_Regex', 'KMeans_Cluster', 'Agglomerative_Cluster', 'DBSCAN_Cluster']].head())
        
        # Provide a bar chart for KMeans clusters to visualize distribution
        kmeans_counts = df['KMeans_Cluster'].value_counts()
        st.write("KMeans Cluster Distribution:")
        st.bar_chart(kmeans_counts)

# Sentiment Analysis
def sentiment_analysis():
    if 'df' not in st.session_state or st.session_state.df.empty:
        st.warning("No data available for sentiment analysis. Please scrape data first.")
        return
    
    df = st.session_state.df
    unique_brands = st.session_state.unique_brands if 'unique_brands' in st.session_state else []

    # Brand extraction function
    def extract_brand(title):
        for brand in unique_brands:
            if re.search(r'\b' + brand + r'\b', title, re.IGNORECASE):
                return brand
        return "Unknown"

    df['Brand'] = df['Product Title'].apply(extract_brand)

    # Numeric rating extraction
    def extract_numeric_rating(rating_text):
        match = re.search(r"(\d+\.\d+|\d+) out of 5 stars", rating_text)
        if match:
            return float(match.group(1))
        return None

    df['Rating'] = df['Product Rating'].apply(extract_numeric_rating)

    # Sentiment labeling based on rating
    def rating_to_sentiment(rating):
        if rating >= 4:
            return 'Positive'
        elif 2.5 <= rating < 4:
            return 'Neutral'
        else:
            return 'Negative'

    df['Rating_Sentiment'] = df['Rating'].apply(rating_to_sentiment)
    st.write("Product Ratings with Sentiment:")
    st.dataframe(df[['Product Title', 'Rating', 'Rating_Sentiment']].head())

    # Title sentiment analysis
    def title_sentiment(title):
        analysis = TextBlob(title)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    df['Title_Sentiment'] = df['Product Title'].apply(title_sentiment)
    st.write("Product Titles with Sentiment:")
    st.dataframe(df[['Product Title', 'Title_Sentiment']].head())

    # Combine sentiments
    def combined_sentiment(row):
        if row['Rating_Sentiment'] == 'Negative' or row['Title_Sentiment'] == 'Negative':
            return 'Negative'
        elif row['Rating_Sentiment'] == 'Positive' and row['Title_Sentiment'] == 'Positive':
            return 'Positive'
        else:
            return 'Neutral'

    df['Final_Sentiment'] = df.apply(combined_sentiment, axis=1)

    # Display final sentiment with brand and title
    st.write("Final Sentiment Analysis:")
    st.dataframe(df[['Product Title', 'Brand', 'Rating', 'Final_Sentiment']])

    # Group by brand and analyze sentiment
    sentiment_summary = df.groupby('Brand')['Final_Sentiment'].value_counts().unstack(fill_value=0)
    st.write("Sentiment Summary by Brand:")
    st.dataframe(sentiment_summary)

    # Plotting the sentiment summary
    st.bar_chart(sentiment_summary)

    st.session_state.df = df  # Update session state with modified dataframe

# Product Recommendations
def product_recommendations():
    st.subheader("Product Recommendations Based on Similarity")

    if 'df' not in st.session_state or st.session_state.df.empty:
        st.warning("Please scrape product data first!")
        return

    df = st.session_state.df
    st.write("Product Recommendations:")
    recommended_products = df[df['Rating'] >= 4][['Product Title', 'Product Price', 'Product Rating']]
    st.dataframe(recommended_products)

# Main Navigation
def main():
    st.title("Amazon Product Analysis Tool")
    nav_choice = st.sidebar.radio("Navigate to:", ("Scraping", "Clustering", "Sentiment Analysis", "Product Recommendations"))

    if nav_choice == "Scraping":
        scraping()

    elif nav_choice == "Clustering":
        clustering()

    elif nav_choice == "Sentiment Analysis":
        sentiment_analysis()

    elif nav_choice == "Product Recommendations":
        product_recommendations()

if __name__ == "__main__":
    main()

