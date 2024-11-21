from fastapi import APIRouter, HTTPException, Response

import json

router = APIRouter(
    prefix="/product",
    responses={404: {"description": "Not found"}},
)


@router.get("/find_similar_products_text")
async def find_similar_products():
    """
    Recherche des produits similaires dans le même cluster en évitant ceux contenant un allergène spécifique.

    Parameters:
        df (DataFrame): Le DataFrame contenant les données produits.
        product_name (str): Nom du produit de référence.
        allergen (str): Allergène à éviter, si spécifié.
        top_n (int): Nombre de produits similaires à retourner.
        encoding_method (function): La méthode d'encodage des données catégorielles.

    Returns:
        DataFrame: Les produits similaires triés par similarité de cosinus.
    """
    return {"message": "find_similar_products route 🎉"}
