import streamlit as st
import json
from openai import OpenAI
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import folium
from streamlit_folium import st_folium
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

client = OpenAI()


OSM_FEATURES = {
    "amenities": [
        "hospital",
        "school",
        "restaurant",
        "cafe",
        "bank",
        "pharmacy",
        "parking",
        "fuel",
        "police",
        "post_office",
        "library",
        "university",
        "bar",
        "cinema",
        "theatre",
        "marketplace",
        "place_of_worship",
    ],
    "boundaries": ["administrative", "national_park", "protected_area", "maritime"],
    "buildings": [
        "residential",
        "commercial",
        "industrial",
        "retail",
        "warehouse",
        "church",
        "school",
        "hospital",
        "train_station",
        "stadium",
    ],
    "barriers": ["wall", "fence", "gate", "bollard", "lift_gate", "cycle_barrier"],
    "highways": [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "footway",
        "cycleway",
        "path",
    ],
    "landuse": [
        "residential",
        "commercial",
        "industrial",
        "retail",
        "forest",
        "farmland",
        "grass",
        "recreation_ground",
    ],
    "natural": [
        "water",
        "wood",
        "tree",
        "scrub",
        "heath",
        "grassland",
        "beach",
        "cliff",
        "peak",
        "volcano",
    ],
    "waterways": ["river", "stream", "canal", "drain", "ditch"],
}


SYSTEM_PROMPT = """
You are a geospatial query interpreter. Convert natural language queries into a JSON structure that handles OpenStreetMap features and spatial operations. Use this structure:
{
    "operation": string,  
    "locations": {       
        "primary": string[],  
        "exclude": string[]   
    },
    "features": {        
        "primary": string,    
        "specific": string,   
        "additional": []      
    },
    "modifiers": {
        "spatial": [],    
        "distance": {     
            "value": number,
            "unit": string
        },
        "filters": {}     
    }
}

Example queries and responses:
1. "Give me Victoria and NSW"
{
    "operation": "union_areas",
    "locations": {
        "primary": ["Victoria, Australia", "New South Wales, Australia"],
        "exclude": []
    }
}

2. "Show me Florida without airports"
{
    "operation": "subtract_areas",
    "locations": {
        "primary": ["Florida, USA"],
        "exclude": []
    },
    "features": {
        "primary": "amenities",
        "specific": "aeroway",
        "additional": []
    }
}

If you cannot interpret the query, respond with:
{"error": "Cannot interpret query. Please rephrase."}
"""


# TODO: Limit the search area
def get_combined_area(locations: List[str]) -> gpd.GeoDataFrame:
    """Get and combine multiple areas"""
    combined_gdf = None

    if locations is None:
        logger.warning("No locations found.")
        return None

    for location in locations:
        try:
            gdf = ox.geocode_to_gdf(location)
            if combined_gdf is None:
                combined_gdf = gdf
            else:
                combined_gdf.geometry = combined_gdf.geometry.union(
                    gdf.geometry.iloc[0]
                )
        except Exception as e:
            st.warning(f"Error processing location {location}: {str(e)}")

    return combined_gdf


def get_features_to_exclude(location: str, features: Dict) -> gpd.GeoDataFrame:
    """Get features that should be excluded from the area"""
    tags = build_osm_tags(features)
    try:
        return ox.features_from_place(location, tags=tags)
    except Exception as e:
        st.warning(f"Error getting features to exclude: {str(e)}")
        return None


def process_spatial_query(query_json: dict) -> gpd.GeoDataFrame:
    """Process the spatial query based on LLM interpretation"""
    logger.debug("check")
    try:
        if query_json.get("error"):
            return None

        if query_json["operation"] == "union_areas":
            return get_combined_area(query_json["locations"]["primary"])

        elif query_json["operation"] == "subtract_areas":

            base_area = get_combined_area(query_json["locations"]["primary"])

            if base_area is None:
                return None

            exclude_area = get_combined_area(query_json["locations"]["exclude"])
            base_area.geometry = base_area.geometry.difference(exclude_area)

            return base_area

        else:

            location = query_json["locations"]["primary"][0]

            tags = build_osm_tags(query_json.get("features", {}))

            if query_json["operation"] == "get_features":
                result = ox.features_from_place(location, tags=tags)
            elif query_json["operation"] == "get_area":
                result = ox.geocode_to_gdf(location)
            elif query_json["operation"] == "get_distance":
                result = ox.geocode_to_gdf(location)
            else:
                return None

            result = apply_spatial_modifiers(result, query_json.get("modifiers", {}))
            return result

    except Exception as e:
        st.error(f"Error processing spatial query: {str(e)}")
        return None


def get_llm_interpretation(query: str) -> dict:
    """Get LLM interpretation of the natural language query"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Error interpreting query: {str(e)}"}


def build_osm_tags(features: Dict) -> Dict:
    """Build OSM tags dictionary based on feature specifications"""
    tags = {}

    if features.get("primary") and features.get("specific"):
        primary = features["primary"].lower()
        specific = features["specific"].lower()

        if primary in OSM_FEATURES:
            if specific in OSM_FEATURES[primary]:
                tags[primary] = specific
            else:
                tags[primary] = True

    for feature in features.get("additional", []):
        if isinstance(feature, dict):
            tags.update(feature)

    return tags


def apply_spatial_modifiers(gdf: gpd.GeoDataFrame, modifiers: Dict) -> gpd.GeoDataFrame:
    """Apply spatial modifications to the GeoDataFrame"""
    logger.debug(modifiers)

    if not modifiers:
        return gdf

    result = gdf.copy()

    if "distance" in modifiers and modifiers["distance"].get("value"):
        distance = modifiers["distance"]["value"]
        unit = modifiers["distance"].get("unit", "meters")

        if unit == "kilometers":
            distance *= 1000
        elif unit == "miles":
            distance *= 1609.34

        result.geometry = result.geometry.buffer(distance)

    for spatial_mod in modifiers.get("spatial", []):
        if spatial_mod == "without":

            if "exclude_geometry" in modifiers:
                result.geometry = result.geometry.difference(
                    modifiers["exclude_geometry"]
                )

    return result


def main():
    st.title("Natural Language Geocoding")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.text_input("Enter your spatial query:", key="query_input")

    if query:

        interpretation = get_llm_interpretation(query)

        if interpretation.get("error"):
            st.error(interpretation["error"])
        else:

            with st.expander("Query Interpretation"):
                st.json(interpretation)

            result = process_spatial_query(interpretation)

            if result is not None and not result.empty:

                map = folium.Map(location=[0, 0], zoom_start=2)

                folium.GeoJson(
                    result.__geo_interface__,
                    style_function=lambda x: {
                        "fillColor": "blue",
                        "color": "blue",
                        "fillOpacity": 0.3,
                    },
                ).add_to(map)

                bounds = result.total_bounds
                map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                st_folium(map)
            else:
                st.warning("No results found for your query.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if __name__ == "__main__":
    main()
