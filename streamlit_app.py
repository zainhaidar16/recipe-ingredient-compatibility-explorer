import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Load and preprocess data
@st.cache_data
def load_data():
    with open("data/train.json", "r") as f:
        train_data = json.load(f)
    with open("data/test.json", "r") as f:
        test_data = json.load(f)
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    test_df["cuisine"] = "Unknown"
    df = pd.concat([train_df, test_df], ignore_index=True)
    return df

# Precompute co-occurrence for all ingredients
@st.cache_data
def precompute_cooccurrence(df):
    pairs = {}
    for recipe in df["ingredients"]:
        for i in range(len(recipe)):
            for j in range(i + 1, len(recipe)):
                pair = tuple(sorted([recipe[i], recipe[j]]))
                pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

# Enhanced network graph with dark background
def plot_network(selected_ingredients, cooccurrence):
    G = nx.Graph()
    for ing in selected_ingredients:
        G.add_node(ing, size=20)
    for (ing1, ing2), weight in cooccurrence.items():
        if ing1 in selected_ingredients and ing2 in selected_ingredients:
            G.add_edge(ing1, ing2, weight=weight)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    avg_weight = sum(d["weight"] for _, _, d in G.edges(data=True)) / len(G.edges()) if G.edges() else 1
    edge_width = max(1, min(5, avg_weight / 100))
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=edge_width, color="#888"),
                            hoverinfo="text", hovertext=[f"Weight: {d['weight']}" for _, _, d in G.edges(data=True)])
    node_x, node_y = zip(*pos.values())
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()),
                            marker=dict(size=[d["size"] for _, d in G.nodes(data=True)], color="#1f77b4",
                                        line=dict(width=2, color="#fff")),
                            textposition="top center", hoverinfo="text", textfont=dict(size=12, color="#ffffff"))
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title="Ingredient Compatibility Network", showlegend=False,
                                     plot_bgcolor="#2c2f33", paper_bgcolor="#2c2f33",
                                     font=dict(color="#ffffff"), margin=dict(b=40, l=40, r=40, t=40)))
    return fig

# Complementary ingredients bar chart with dark background
def plot_complementary_ingredients(df, selected_ingredients):
    all_related = []
    for recipe in df["ingredients"]:
        if any(ing in recipe for ing in selected_ingredients):
            all_related.extend([ing for ing in recipe if ing not in selected_ingredients])
    top_complements = Counter(all_related).most_common(5)
    ingredients, counts = zip(*top_complements)
    fig = go.Figure([go.Bar(x=ingredients, y=counts, marker_color="#f39c12",
                            text=counts, textposition="auto", textfont=dict(color="#ffffff"))])
    fig.update_layout(title="Top Complementary Ingredients", xaxis_title="Ingredient",
                      yaxis_title="Frequency", plot_bgcolor="#2c2f33", paper_bgcolor="#2c2f33",
                      font=dict(color="#ffffff"), xaxis_tickangle=-45)
    return fig

# Adventurous pairing
def get_adventurous_pairing(df, selected_ingredients):
    vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
    ingredient_matrix = vectorizer.fit_transform(df["ingredients"]).toarray()
    ingredient_names = vectorizer.get_feature_names_out()
    
    selected_indices = [i for i, ing in enumerate(ingredient_names) if ing in selected_ingredients]
    if not selected_indices:
        return None
    
    selected_vector = ingredient_matrix[:, selected_indices].mean(axis=1).reshape(1, -1)
    similarities = cosine_similarity(selected_vector, ingredient_matrix.T)[0]
    
    freq = Counter([ing for sublist in df["ingredients"] for ing in sublist])
    candidates = [(ing, sim) for ing, sim in zip(ingredient_names, similarities)
                  if ing not in selected_ingredients and freq[ing] < 50 and sim > 0.1]
    return max(candidates, key=lambda x: x[1])[0] if candidates else None

# Download function
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="recipes.csv">Download CSV</a>'
    return href

# Main app
def main():
    st.set_page_config(page_title="Recipe Compatibility Explorer", layout="wide")

    # Header
    # Commented out online image due to potential loading issues; use local image if available
    # st.image("logo.png", width=100, caption="Your Logo")  # Uncomment and replace with your local image path
    st.title("Recipe Ingredient Compatibility Explorer")
    st.markdown("Discover how your ingredients pair and find recipe inspiration!")

    # Sidebar
    st.sidebar.title("Controls")
    df = load_data()
    all_ingredients = sorted(set([ing for sublist in df["ingredients"] for ing in sublist]))
    cooccurrence = precompute_cooccurrence(df)

    search_query = st.sidebar.text_input("Search Ingredients", "", help="Type to filter ingredients (e.g., 'onion')")
    if search_query:
        filtered_ingredients = [ing for ing in all_ingredients if search_query.lower() in ing.lower()]
    else:
        filtered_ingredients = all_ingredients
    
    selected_ingredients = st.sidebar.multiselect("Select Ingredients", filtered_ingredients,
                                                  default=["chicken", "garlic", "olive oil"],
                                                  help="Choose 2+ ingredients for best results.")
    cuisine_options = ["All"] + sorted(df["cuisine"].unique())
    cuisine = st.sidebar.selectbox("Filter by Cuisine", cuisine_options,
                                   help="Filter recipes by cuisine type (e.g., 'Italian').")
    
    tab1, tab2 = st.tabs(["Explorer", "About"])

    with tab1:
        if len(selected_ingredients) < 1:
            st.warning("Please select at least one ingredient to begin!")
            return

        filtered_df = df if cuisine == "All" else df[df["cuisine"] == cuisine]
        filtered_cooccurrence = {k: v for k, v in cooccurrence.items() if all(ing in selected_ingredients for ing in k)}

        with st.expander("Network & Complements", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ingredient Compatibility Network")
                if len(selected_ingredients) >= 2:
                    with st.spinner("Rendering network..."):
                        fig_network = plot_network(selected_ingredients, filtered_cooccurrence)
                    st.plotly_chart(fig_network, use_container_width=True)
                else:
                    st.info("Select 2 or more ingredients to see the network!")
            
            with col2:
                st.subheader("Top Complementary Ingredients")
                with st.spinner("Analyzing complements..."):
                    fig_complements = plot_complementary_ingredients(filtered_df, selected_ingredients)
                st.plotly_chart(fig_complements, use_container_width=True)

        with st.expander("Suggestions", expanded=True):
            st.subheader("Adventurous Pairing Suggestion")
            with st.spinner("Finding a unique pairing..."):
                adventurous = get_adventurous_pairing(filtered_df, selected_ingredients)
            if adventurous:
                st.success(f"Try something new: **{adventurous}** pairs surprisingly well with your selection!")
            else:
                st.write("No adventurous pairing found. Try different ingredients!")

            st.subheader("Recipe Suggestions")
            recipes = filtered_df[filtered_df["ingredients"].apply(lambda x: all(ing in x for ing in selected_ingredients))]
            if not recipes.empty:
                st.dataframe(recipes[["id", "cuisine", "ingredients"]].head(5), use_container_width=True)
                st.markdown(download_csv(recipes[["id", "cuisine", "ingredients"]].head(5)), unsafe_allow_html=True)
            else:
                st.write("No exact matches found. Try different ingredients or cuisine!")

            if st.button("Random Suggestion"):
                random_ing = all_ingredients[st.session_state.get("random_idx", 0) % len(all_ingredients)]
                st.session_state.random_idx = st.session_state.get("random_idx", 0) + 1
                st.info(f"Random suggestion: **{random_ing}**")

    with tab2:
        st.title("About This App")
        st.markdown("""
        ### Recipe Ingredient Compatibility Explorer
        This app analyzes ingredient pairings from over 50,000 recipes, combining training and test data from Kaggle’s "What’s Cooking" dataset. 
        It visualizes how ingredients connect, suggests complementary flavors, and provides recipe ideas based on your selections.

        #### Features
        - **Interactive Network**: See how your ingredients pair using a dynamic graph with hover details.
        - **Complementary Insights**: Discover the most common ingredients that pair with your choices.
        - **Adventurous Pairing**: Get a rare but viable ingredient suggestion using cosine similarity.
        - **Recipe Finder**: Find and export recipes matching your ingredients, filtered by cuisine.
        - **Random Suggestion**: Explore new ingredients with a click.

        #### Tech Stack
        - **Streamlit**: Interactive UI.
        - **Pandas**: Data manipulation.
        - **NetworkX & Plotly**: Graph visualization.
        - **Scikit-learn**: Cosine similarity for adventurous pairings.

        #### Data Source
        [What’s Cooking dataset](https://www.kaggle.com/kaggle/recipe-ingredients-dataset) from Kaggle.

        #### Project Details
        - **Version**: 1.0
        - **Last Updated**: March 08, 2025
        - **Built by**: [Your Name]
        - **GitHub**: [Insert GitHub Link]
        - **Contact**: [Your Email]
        """)

    # Footer
    st.markdown("---")
    st.markdown("© 2025 Zain The Analyst | Contact: contact@zaintheanalyst.com ", unsafe_allow_html=False)

if __name__ == "__main__":
    if "random_idx" not in st.session_state:
        st.session_state.random_idx = 0
    main()