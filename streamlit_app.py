import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import json
from collections import Counter

# Load and preprocess data
@st.cache_data
def load_data():
    with open("data/train.json", "r") as f:
        train_data = json.load(f)
    with open("data/test.json", "r") as f:
        test_data = json.load(f)
    # Combine train and test, adding cuisine where missing in test
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    test_df["cuisine"] = "Unknown"  # Placeholder for test data
    df = pd.concat([train_df, test_df], ignore_index=True)
    return df

# Build co-occurrence matrix
def get_cooccurrence(df, ingredients):
    pairs = {}
    for recipe in df["ingredients"]:
        for i in range(len(recipe)):
            for j in range(i + 1, len(recipe)):
                pair = tuple(sorted([recipe[i], recipe[j]]))
                pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

# Network graph
def plot_network(selected_ingredients, cooccurrence):
    G = nx.Graph()
    for ing in selected_ingredients:
        G.add_node(ing)
    for (ing1, ing2), weight in cooccurrence.items():
        if ing1 in selected_ingredients and ing2 in selected_ingredients:
            G.add_edge(ing1, ing2, weight=weight)
    
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2, color="#888"))
    node_x, node_y = zip(*pos.values())
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()), 
                            marker=dict(size=25, color="#1f77b4", line=dict(width=2, color="#fff")), 
                            textposition="top center", hoverinfo="text")
    fig = go.Figure(data=[edge_trace, node_trace], 
                    layout=go.Layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", 
                                     margin=dict(b=0, l=0, r=0, t=0)))
    return fig

# Complementary ingredients bar chart
def plot_complementary_ingredients(df, selected_ingredients):
    all_related = []
    for recipe in df["ingredients"]:
        if any(ing in recipe for ing in selected_ingredients):
            all_related.extend([ing for ing in recipe if ing not in selected_ingredients])
    top_complements = Counter(all_related).most_common(5)
    ingredients, counts = zip(*top_complements)
    fig = go.Figure([go.Bar(x=ingredients, y=counts, marker_color="#ff7f0e")])
    fig.update_layout(title="Top Complementary Ingredients", xaxis_title="Ingredient", yaxis_title="Frequency")
    return fig

# Main app
def main():
    # Set page config for wide layout and title
    st.set_page_config(page_title="Recipe Compatibility Explorer", layout="wide")

    # Sidebar
    st.sidebar.title("Controls")
    df = load_data()
    all_ingredients = sorted(set([ing for sublist in df["ingredients"] for ing in sublist]))
    selected_ingredients = st.sidebar.multiselect("Select Ingredients", all_ingredients, 
                                                  default=["chicken", "garlic", "olive oil"],
                                                  help="Choose 2+ ingredients for best results.")
    cuisine_options = ["All"] + sorted(df["cuisine"].unique())
    cuisine = st.sidebar.selectbox("Filter by Cuisine", cuisine_options, 
                                   help="Filter recipes by cuisine type.")
    
    # Tabs for Main Content and About
    tab1, tab2 = st.tabs(["Explorer", "About"])

    with tab1:
        if len(selected_ingredients) < 1:
            st.warning("Please select at least one ingredient to begin!")
            return

        # Filter data by cuisine
        filtered_df = df if cuisine == "All" else df[df["cuisine"] == cuisine]

        # Title and layout
        st.title("Recipe Ingredient Compatibility Explorer")
        st.markdown("Discover how your ingredients pair and find recipe inspiration!")

        # Two-column layout
        col1, col2 = st.columns(2)

        with col1:
            # Network Graph
            st.subheader("Ingredient Compatibility Network")
            cooccurrence = get_cooccurrence(filtered_df, selected_ingredients)
            if len(selected_ingredients) >= 2:
                fig_network = plot_network(selected_ingredients, cooccurrence)
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.info("Select 2 or more ingredients to see the network!")

        with col2:
            # Complementary Ingredients
            st.subheader("Top Complementary Ingredients")
            fig_complements = plot_complementary_ingredients(filtered_df, selected_ingredients)
            st.plotly_chart(fig_complements, use_container_width=True)

        # Recipe Suggestions
        st.subheader("Recipe Suggestions")
        recipes = filtered_df[filtered_df["ingredients"].apply(lambda x: all(ing in x for ing in selected_ingredients))]
        if not recipes.empty:
            st.dataframe(recipes[["id", "cuisine", "ingredients"]].head(5).style.set_properties(**{
                "background-color": "#f9f9f9", "border-color": "#ddd", "padding": "5px"
            }), use_container_width=True)
        else:
            st.write("No exact matches found. Try different ingredients or cuisine!")

    with tab2:
        st.title("About This App")
        st.markdown("""
        ### Recipe Ingredient Compatibility Explorer
        This app analyzes ingredient pairings from over 50,000 recipes, combining training and test data from Kaggle’s "What’s Cooking" dataset. 
        It visualizes how ingredients connect, suggests complementary flavors, and provides recipe ideas based on your selections.

        #### Features
        - **Interactive Network**: See how your ingredients pair using a dynamic graph.
        - **Complementary Insights**: Discover the most common ingredients that pair with your choices.
        - **Recipe Finder**: Get recipes matching your ingredients, filtered by cuisine.

        #### Tech Stack
        - **Streamlit**: For the interactive UI.
        - **Pandas**: Data manipulation.
        - **NetworkX & Plotly**: Graph visualization.
        - **Python**: Core logic and analysis.

        #### Data Source
        The app uses the [What’s Cooking dataset](https://www.kaggle.com/kaggle/recipe-ingredients-dataset) from Kaggle, 
        combining both `train.json` and `test.json` for a richer analysis.

        Built by [Your Name] as a portfolio project to showcase data analysis, visualization, and app development skills.
        """)

if __name__ == "__main__":
    main()