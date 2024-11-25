import os
import streamlit as st
import requests
import polars as pl
from classes.api_error import APIError


#############################################################################
#   Global variable
#############################################################################
API_URL = os.environ["API_URL"]

MAX_RESULTS = 20

DTYPES = {
    "code": pl.Utf8,
    "url": pl.Utf8,
    "last_modified_t": pl.Int64,
    "product_name": pl.Utf8,
    "packaging_tags": pl.Utf8,
    "categories_tags": pl.Utf8,
    "ingredients_tags": pl.Utf8,
    "ingredients_analysis_tags": pl.Utf8,
    "allergens": pl.Utf8,
    "traces_tags": pl.Utf8,
    "additives_tags": pl.Utf8,
    "nutriscore_grade": pl.Utf8,
    "food_groups_tags": pl.Utf8,
    "states_tags": pl.Utf8,
    "ecoscore_grade": pl.Utf8,
    "nutrient_levels_tags": pl.Utf8,
    "popularity_tags": pl.Utf8,
    "main_category": pl.Utf8,
    "image_url": pl.Utf8,
    "image_small_url": pl.Utf8,
    "energy_100g": pl.Float32,
    "fat_100g": pl.Float32,
    "saturated-fat_100g": pl.Float32,
    "cholesterol_100g": pl.Float32,
    "sugars_100g": pl.Float32,
    "proteins_100g": pl.Float32,
    "salt_100g": pl.Float32,
    "fruits-vegetables-nuts-estimate-from-ingredients_100g": pl.Float32,
    "last_modified_year": pl.Int32,
    "preprocessed_nutriscore_grade": pl.Utf8,
    "preprocessed_ecoscore_grade": pl.Utf8,
    "preprocessed_product_name": pl.Utf8,
    "preprocessed_packaging_tags": pl.Utf8,
    "preprocessed_packaging_tags_lemmatized": pl.Utf8,
    "preprocessed_categories_tags": pl.Utf8,
    "preprocessed_categories_tags_lemmatized": pl.Utf8,
    "preprocessed_ingredients_tags": pl.Utf8,
    "preprocessed_ingredients_tags_lemmatized": pl.Utf8,
    "preprocessed_ingredients_analysis_tags": pl.Utf8,
    "preprocessed_ingredients_analysis_tags_lemmatized": pl.Utf8,
    "preprocessed_nutrient_levels_tags": pl.Utf8,
    "preprocessed_nutrient_levels_tags_lemmatized": pl.Utf8,
    "preprocessed_main_category": pl.Utf8,
    "preprocessed_main_category_lemmatized": pl.Utf8,
    "preprocessed_popularity_tags": pl.Utf8,
    "preprocessed_popularity_tags_lemmatized": pl.Utf8,
    "cluster_text": pl.Utf8,
}

COLUMNS_TO_KEEP = [
    "code",
    "url",
    "product_name",
    "allergens",
    "traces_tags",
    "image_url",
    "nutriscore_grade",
    "ecoscore_grade",
]


#############################################################################
#   Functions
#############################################################################


def load_database():
    lazy_df = pl.scan_csv(
        "./data/production/database_text_api.csv",
        # columns=COLUMNS_TO_KEEP,
        schema_overrides=DTYPES,
    )
    return lazy_df


def search(search_term):
    lazy_df = load_database()
    return (
        lazy_df.filter(
            pl.col("code").str.contains(search_term)
            | pl.col("product_name").str.to_lowercase().str.contains(search_term.lower())
        )
        .head(MAX_RESULTS)
        .collect()
        .to_pandas()
    )


def get_similar_products_text(product_code, allergen=None, top_n=10):
    body = {"code": product_code, "top_n": top_n, "allergen": allergen}
    response = requests.post(f"{API_URL}/product/find_similar_products_text", json=body)
    if response.status_code == 200:
        return response.json()
    else:
        raise APIError(response.status_code, f"API Error: {response.status_code}")


def get_similar_products_image(product_code, top_n=10):
    body = {"code": product_code, "top_n": top_n}
    response = requests.post(f"{API_URL}/product/find_similar_products_image", json=body)
    if response.status_code == 200:
        return response.json()
    else:
        raise APIError(response.status_code, f"API Error: {response.status_code}")


#############################################################################
#   CSS
#############################################################################


#############################################################################
#   IHM
#############################################################################
if __name__ == "__main__":
    # Configuration de la page
    st.set_page_config(
        page_title="Recherche de Produits Alimentaires",
        page_icon="🍲",
        layout="wide",
    )

    ## CSS
    st.markdown(
        """
    <style>
    .button-search {
        display: inline-block;
        color: white;
        background-color: #007BFF;
        border: none;
        padding: 8px 16px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
        border-radius: 4px;
        margin-left: 10px;
    }
    .button-search:hover {
        background-color: #0056b3;
    }
    .product-grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .product-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px;
        width: calc(90% - 20px); /* 25% de la largeur de la grille moins les marges */
        height: 250px;
        text-align: center;
        box-sizing: border-box;
    }
    .product-card img {
        width: 120px;
        height: 120px;
        object-fit: cover;
        border-radius: 5px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("🚀 Foodstuffs Recommendation")

        # Ajouter une image de fond
    st.sidebar.image("assets/images/logo_fr.webp")
    st.sidebar.title("À propos 📚")
    st.sidebar.info(
        "Cette application vous aide à trouver des produits alimentaires similaires qui ne contiennent pas d'ingrédients pouvant provoquer des allergies."
    )

    # Titre de l'application
    st.title("Recherche de Produits Alimentaires 🍲")

    st.markdown("---")

    # Avertissement pour l'utilisateur
    st.warning(
        "🚨 **Attention**: Veuillez vérifier attentivement le contenu des ingrédients et les traces d'allergènes avant de consommer les produits recommandés. 🚨"
    )

    filtered_df = pl.DataFrame(data=[]).to_pandas()  # Init empty dataframe

    # Champs de recherche
    search_term = st.text_input(
        "Recherchez un produit ou un code dans la base de données.",
        label_visibility="collapsed",
    )

    # Filtrer le DataFrame en fonction du terme de recherche
    if st.button("Rechercher"):
        if len(search_term) < 3:
            st.warning(
                "Veuillez entrer au moins 3 caractères pour effectuer une recherche."
            )
        else:
            with st.spinner("Recherche en cours..."):
                filtered_df = search(search_term)

    cols_per_row = 4

    if not filtered_df.empty:
        # Créer des lignes de produits
        rows = [
            filtered_df.iloc[i : i + cols_per_row]
            for i in range(0, len(filtered_df), cols_per_row)
        ]

        st.markdown('<div class="product-grid">', unsafe_allow_html=True)

        # Afficher chaque ligne
        for row in rows:
            cols = st.columns(cols_per_row)
            for col, (_, product) in zip(cols, row.iterrows()):
                with col:
                    st.markdown(
                        f"""
                        <div class="product-card">
                            <img src="{product['image_url']}" alt="{product['product_name']}">
                            <a href="{product['url']}" target="_blank">{product['product_name']}</a>
                            <p>Code: {product['code']}</p>
                            <p>Nutriscore: {product['nutriscore_grade']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("Aucun produit correspondant trouvé.")

    st.markdown("---")

    # Créer des onglets pour faire la recherche par texte et par image
    text, image = st.tabs(['Text', 'Image'])

    with text:

        # Formulaire pour entrer le code produit et l'allergie
        st.subheader("Recherchez des alternatives à un produit basé sur caractérisques textuelles.")
        product_code = st.text_input("Code du produit:")

        # Dictionnaire des allergènes les plus courants

        allergens = {
            "None": "Aucun",
            "en:milk": "Lait",
            "en:peanuts": "Arachides",
            "en:gluten": "Gluten",
            "en:eggs": "Œufs",
            "en:soybeans": "Soja",
            "en:nuts": "Fruits à coque",
            "en:fish": "Poisson",
            "en:sulphur-dioxide-and-sulphites": "Sulfites",
            "en:celery": "Céleri",
            "en:sesame-seeds": "Sésame",
            "en:crustaceans": "Crustacés",
            "en:molluscs": "Mollusques",
            "en:lupin": "Lupin",
        }

        # Liste déroulante pour sélectionner un allergène
        allergy_key = st.selectbox(
            "Sélectionnez un allergène à banir (optionnel):",
            options=list(allergens.keys()),
            format_func=lambda x: allergens[x],
        )

        # Bouton pour soumettre le formulaire
        if st.button("Rechercher 🔍"):
            if product_code:
                allergy_value = allergens[allergy_key]
                with st.spinner("Recherche de produits similaires..."):
                    similar_products = get_similar_products_text(product_code, allergy_key)
                    if similar_products:
                        st.success("Produits similaires trouvés:")
                        st.markdown('<div class="product-grid">', unsafe_allow_html=True)

                        # Affichage des produits sous forme de cartes
                        for key, product in similar_products.items():
                            st.markdown(
                                f"""
                                <div class="product-card">
                                    <img src="{product['image_url']}" alt="{product['product_name']}">
                                    <a href="{product['url']}" target="_blank">{product['product_name']}</a>
                                    <p>Nutriscore: {product['nutriscore_grade']}</p>
                                    <p>Allergènes: {product['allergens']}</p>
                                    <p>Code: {product['code']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Aucun produit similaire trouvé ou erreur API.")
            else:
                st.error("Veuillez entrer à la fois le code produit et l'allergie.")

    with image:
        # Formulaire pour entrer le code produit et l'allergie
        st.subheader(
            "Recherchez des alternatives à un produit basé sur caractérisques visuelles."
        )
        product_code = st.text_input("Code du produit:", key="product_code_image")

        if st.button("Rechercher 🔍", key="image_recherche"):
            if product_code:
                with st.spinner("Recherche de produits similaires..."):
                    similar_products = get_similar_products_image(
                        int(product_code)
                    )
                    if similar_products:
                        st.success("Produits similaires trouvés:")
                        st.markdown(
                            '<div class="product-grid">', unsafe_allow_html=True
                        )

                        # Affichage des produits sous forme de cartes
                        for key, product in similar_products.items():
                            st.markdown(
                                f"""
                                <div class="product-card">
                                    <img src="{product['image_url']}" alt="{product['product_name']}">
                                    <a href="{product['url']}" target="_blank">{product['product_name']}</a>
                                    <p>Code: {product['code']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Aucun produit similaire trouvé ou erreur API.")
            else:
                st.error("Veuillez entrer le code produit.")
