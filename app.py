# app.py
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

# TODO: No hard code
# OSM feature categories
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

# TODO: Improve. This is like the worst prompt I've ever seen :((
SYSTEM_PROMPT = """
You are a geospatial query interpreter. Convert natural language queries into a JSON structure that handles all OpenStreetMap primary features. Use this structure:
{
    "operation": string,  # get_features, get_area, get_distance, get_poi
    "location": string,   # primary location to search
    "features": {        # OSM features to search for
        "primary": string,    # main category (amenities, boundaries, buildings, etc.)
        "specific": string,   # specific feature type
        "additional": []      # additional features to consider
    },
    "modifiers": {
        "spatial": [],    # spatial modifications (within, without, left_of, etc.)
        "distance": {     # distance specifications
            "value": number,
            "unit": string
        },
        "filters": {}     # additional filters
    }
}

Available feature categories:
- amenities (hospital, school, restaurant, etc.)
- boundaries (administrative, national_park, etc.)
- buildings (residential, commercial, etc.)
- barriers (wall, fence, gate, etc.)
- highways (motorway, primary, residential, etc.)
- landuse (residential, commercial, forest, etc.)
- natural (water, wood, beach, etc.)
- waterways (river, stream, canal, etc.)

If you cannot interpret the query, respond with:
{"error": "Cannot interpret query. Please rephrase."}
"""

client = OpenAI()


def get_llm_interpretation(query: str) -> dict:
    """Get LLM interpretation of the natural language query"""
    try:
        response = client.chat.completions.create(
            # For some reason only gpt-4 works, 4o and 4o-mini doesn't
            model="gpt-4",
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

        # Handle different feature types
        if primary in OSM_FEATURES:
            if specific in OSM_FEATURES[primary]:
                tags[primary] = specific
            else:
                tags[primary] = True

    # Add additional features
    for feature in features.get("additional", []):
        if isinstance(feature, dict):
            tags.update(feature)

    return tags


def apply_spatial_modifiers(gdf: gpd.GeoDataFrame, modifiers: Dict) -> gpd.GeoDataFrame:
    """Apply spatial modifications to the GeoDataFrame"""
    if not modifiers:
        return gdf

    result = gdf.copy()

    # Handle distance modifications
    if "distance" in modifiers and modifiers["distance"].get("value"):
        distance = modifiers["distance"]["value"]
        unit = modifiers["distance"].get("unit", "meters")

        if unit == "kilometers":
            distance *= 1000
        elif unit == "miles":
            distance *= 1609.34

        result.geometry = result.geometry.buffer(distance)

    # Handle spatial relations
    for spatial_mod in modifiers.get("spatial", []):
        if spatial_mod == "without":
            # Implement exclusion
            if "exclude_geometry" in modifiers:
                result.geometry = result.geometry.difference(
                    modifiers["exclude_geometry"]
                )

    return result


def process_spatial_query(query_json: dict) -> gpd.GeoDataFrame:
    """Process the spatial query based on LLM interpretation"""
    try:
        if query_json.get("error"):
            return None

        # Get base location
        location = query_json["location"]

        # Build OSM tags
        tags = build_osm_tags(query_json.get("features", {}))

        # Get features based on operation type
        if query_json["operation"] == "get_features":
            # Get specific features within the area
            result = ox.features_from_place(location, tags=tags)

        elif query_json["operation"] == "get_area":
            # Get base area
            result = ox.geocode_to_gdf(location)

        elif query_json["operation"] == "get_distance":
            # Handle distance-based queries
            result = ox.geocode_to_gdf(location)

        else:
            return None

        # Apply any spatial modifiers
        result = apply_spatial_modifiers(result, query_json.get("modifiers", {}))

        return result

    except Exception as e:
        st.error(f"Error processing spatial query: {str(e)}")
        return None


def main():
    st.title("Natural Language Geocoding")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat input
    query = st.text_input("Enter your spatial query:", key="query_input")

    if query:

        # Get LLM interpretation
        interpretation = get_llm_interpretation(query)

        if interpretation.get("error"):
            st.error(interpretation["error"])
        else:
            # Show interpretation (for debugging)
            with st.expander("Query Interpretation"):
                st.json(interpretation)

            # Process spatial query
            result = process_spatial_query(interpretation)

            if result is not None and not result.empty:
                # Display map
                m = folium.Map(location=[0, 0], zoom_start=2)

                # Convert to GeoJSON and add to map
                folium.GeoJson(
                    result.__geo_interface__,
                    style_function=lambda x: {
                        "fillColor": "blue",
                        "color": "blue",
                        "fillOpacity": 0.3,
                    },
                ).add_to(m)

                # Fit map bounds to the data
                bounds = result.total_bounds
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                # Display map
                st_folium(m)
            else:
                st.warning("No results found for your query.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if __name__ == "__main__":
    main()
