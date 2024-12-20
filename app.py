import streamlit as st
import json
import pydeck as pdk
from openai import OpenAI
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger
from shapely import wkt

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
You are a geospatial query interpreter specialized in converting natural language location queries into structured JSON for OpenStreetMap operations. Your goal is to maintain spatial context and relationships between locations while interpreting queries.

Output Schema:
{
    "operation": string,  # Required: operation type
    "locations": {
        "primary": string[],  # Required: main location(s)
        "exclude": string[],  # Optional: locations to exclude
        "context": string[]   # Optional: parent locations for context
    },
    "features": {
        "primary": string,    # Required if features mentioned
        "specific": string,   # Required if specific feature type
        "additional": []      # Optional: related features
    },
    "modifiers": {
        "spatial": [],        # Optional: spatial relationships
        "distance": {
            "value": number,
            "unit": string
        },
        "filters": {
            "tags": {},       # Optional: OSM tags
            "properties": {}   # Optional: feature properties
        }
    }
}

Operations:
- "get_area": Single area retrieval
- "union_areas": Combine multiple areas
- "subtract_areas": Remove areas/features
- "intersection": Find common areas
- "buffer": Create area around point/line
- "filter": Apply feature filters

Spatial Context Rules:
1. Always include full location paths (e.g., "Melbourne, Victoria, Australia")
2. Maintain parent-child relationships in location hierarchy
3. Use "context" field for implicit spatial relationships
4. Preserve administrative boundaries when mentioned

Example Queries and Responses:

1. "Find schools in Melbourne's CBD"
{
    "operation": "filter",
    "locations": {
        "primary": ["Melbourne CBD, Melbourne, Victoria, Australia"],
        "context": ["Melbourne, Victoria, Australia"]
    },
    "features": {
        "primary": "amenity",
        "specific": "school",
        "additional": ["education"]
    }
}

2. "Show me parks within 5km of Sydney Harbor"
{
    "operation": "buffer",
    "locations": {
        "primary": ["Sydney Harbor, Sydney, New South Wales, Australia"],
        "context": ["Sydney, New South Wales, Australia"]
    },
    "features": {
        "primary": "leisure",
        "specific": "park"
    },
    "modifiers": {
        "spatial": ["within"],
        "distance": {
            "value": 5,
            "unit": "km"
        }
    }
}

3. "Give me Victoria and NSW excluding national parks"
{
    "operation": "subtract_areas",
    "locations": {
        "primary": ["Victoria, Australia", "New South Wales, Australia"],
        "exclude": [],
        "context": ["Australia"]
    },
    "features": {
        "primary": "boundary",
        "specific": "national_park",
        "additional": ["protected_area"]
    }
}

Error Handling:
1. If query is ambiguous, request clarification:
{
    "error": "Ambiguous location. Did you mean Melbourne, Australia or Melbourne, Florida?",
    "suggestions": ["Melbourne, Victoria, Australia", "Melbourne, Florida, USA"]
}

2. If query cannot be interpreted:
{
    "error": "Cannot interpret query. Please rephrase with clearer location or feature details."
}

Remember to:
- Always include administrative context for locations
- Preserve spatial relationships between mentioned places
- Use appropriate OSM tags for features
- Include relevant parent locations in context field
- Handle ambiguous cases with clarification requests
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
    # logger.debug("check")
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

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Get Mapbox token
    mapbox_token = st.secrets["mapbox"][
        "token"
    ]  # Store your token in streamlit secrets

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
                # Convert GeoDataFrame to format suitable for PyDeck
                if "geometry" in result.columns:
                    # Extract coordinates from geometry
                    geojson_data = result.__geo_interface__

                    # Create PyDeck layer based on geometry type
                    if geojson_data["features"][0]["geometry"]["type"] == "Polygon":
                        layer = pdk.Layer(
                            "PolygonLayer",
                            data=geojson_data["features"],
                            get_polygon="geometry.coordinates",
                            get_fill_color=[0, 0, 255, 75],  # Blue with 75% opacity
                            get_line_color=[0, 0, 255],
                            line_width_min_pixels=1,
                            pickable=True,
                            auto_highlight=True,
                        )
                    elif geojson_data["features"][0]["geometry"]["type"] == "Point":
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=geojson_data["features"],
                            get_position="geometry.coordinates",
                            get_fill_color=[0, 0, 255, 75],
                            get_radius=1000,
                            pickable=True,
                        )
                    else:
                        layer = pdk.Layer(
                            "GeoJsonLayer",
                            data=geojson_data,
                            get_fill_color=[0, 0, 255, 75],
                            get_line_color=[0, 0, 255],
                            pickable=True,
                        )

                    # Calculate bounds for viewport
                    bounds = result.total_bounds
                    center_lat = (bounds[1] + bounds[3]) / 2
                    center_lon = (bounds[0] + bounds[2]) / 2

                    # TODO: Better way to do this
                    # Calculate zoom level based on bounds
                    zoom_level = 11  # Default zoom level
                    if bounds[2] - bounds[0] > 0 and bounds[3] - bounds[1] > 0:
                        zoom_level = min(
                            20, max(1, 20 - (bounds[2] - bounds[0]) * 7)
                        )  # Adjust zoom based on width

                    logger.debug(zoom_level)

                    # Create view state
                    view_state = pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=zoom_level,
                        pitch=0,
                    )

                    # Create deck
                    deck = pdk.Deck(
                        map_style="mapbox://styles/mapbox/light-v9",
                        initial_view_state=view_state,
                        layers=[layer],
                        api_keys={"mapbox": mapbox_token},
                    )

                    # Display the map
                    st.pydeck_chart(deck)
            else:
                st.warning("No results found for your query.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if __name__ == "__main__":
    main()
