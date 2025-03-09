# Recipe Ingredient Compatibility Explorer

## Overview

The Recipe Ingredient Compatibility Explorer is a Streamlit application that helps you discover how different ingredients pair together and find recipe inspiration. The app analyzes ingredient pairings from over 50,000 recipes, combining training and test data from Kaggle’s "What’s Cooking" dataset. It visualizes how ingredients connect, suggests complementary flavors, and provides recipe ideas based on your selections.

## Features

- **Interactive Network**: See how your ingredients pair using a dynamic graph with hover details.
- **Complementary Insights**: Discover the most common ingredients that pair with your choices.
- **Adventurous Pairing**: Get a rare but viable ingredient suggestion using cosine similarity.
- **Recipe Finder**: Find and export recipes matching your ingredients, filtered by cuisine.
- **Random Suggestion**: Explore new ingredients with a click.

## Tech Stack

- **Streamlit**: Interactive UI.
- **Pandas**: Data manipulation.
- **NetworkX & Plotly**: Graph visualization.
- **Scikit-learn**: Cosine similarity for adventurous pairings.

## Data Source

The app uses the [What’s Cooking dataset](https://www.kaggle.com/competitions/whats-cooking/overview) from Kaggle.

## Project Details

- **Version**: 1.0
- **Last Updated**: March 08, 2025
- **Built by**: Zain The Analyst
- **GitHub**: [https://github.com/zainhaidar16](https://github.com/zainhaidar16)
- **Contact**: contact@zaintheanalyst.com

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/zainhaidar16/recipe-ingredient-compatibility-explorer.git
    cd recipe-ingredient-compatibility-explorer
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the dataset and place it in the `data` directory:
    - `train.json`
    - `test.json`

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run streamlit_app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the app.

## How to Use the App

1. **Select Ingredients**: Use the sidebar to search and select ingredients. Choose at least two ingredients for the best results.
2. **Filter by Cuisine**: Optionally, filter recipes by cuisine type.
3. **Explore**: Use the "Explorer" tab to visualize ingredient compatibility networks, see top complementary ingredients, and get adventurous pairing suggestions.
4. **Recipe Suggestions**: Find recipes that match your selected ingredients and download them as a CSV file.
5. **Random Suggestion**: Click the "Random Suggestion" button to explore new ingredients.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or feedback, please contact Zain The Analyst at contact@zaintheanalyst.com.