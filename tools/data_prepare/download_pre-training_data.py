import os
import math
import requests
import json

def latlon_to_tile(lat, lon, zoom):
    """Converts latitude and longitude to tile coordinates at a given zoom level."""
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x_tile, y_tile

def download_google_maps_tile(api_key, session_token, lat, lon, zoom=18, output_folder="tiles", filename="tile"):
    """Downloads a satellite tile from Google Maps API for given coordinates."""
    x_tile, y_tile = latlon_to_tile(lat, lon, zoom)
    base_url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x_tile}/{y_tile}?session={session_token}&key={api_key}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, f"{filename}.png")
        with open(filepath, "wb") as file:
            file.write(response.content)
        print(f"Tile saved: {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download tile for {lat}, {lon}. Error: {e}")

def extract_coordinates_from_json(json_file):
    """Extracts latitude and longitude from the JSON file."""
    with open(json_file, "r") as file:
        data = json.load(file)
        entries = []
        for item in data["data"]:
            name_parts = item["name"].split("_")
            lon1, lat1 = float(name_parts[1]), float(name_parts[2])
            lon2, lat2 = float(name_parts[3]), float(name_parts[4])
            midpoint_lat = ((lat1 + lat2) / 2)
            midpoint_lon = ((lon1 + lon2) / 2)
            entries.append({
                "name": item["name"],
                "lat": midpoint_lat,
                "lon": midpoint_lon,
                "info": item["info"],
                "attrs": item["attrs"],
                "cap": item["cap"]
            })
    return entries

def save_selected_entries_to_json(entries, output_file):
    """Saves the selected entries to a new JSON file."""
    data = {"data": entries}
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Selected entries saved to {output_file}")

if __name__ == "__main__":
    API_KEY = "" # Replace with your API key
    SESSION_TOKEN = "" # Replace with your session token
    JSON_FILE = "" # Replace with the path to your JSON file
    OUTPUT_FOLDER = "" # Folder to save downloaded tiles
    ZOOM = 18 # Adjust zoom level as needed

    # Extract all entries from the JSON file
    all_entries = extract_coordinates_from_json(JSON_FILE)

    # Download tiles for the selected entries
    for entry in all_entries:
        download_google_maps_tile(API_KEY, SESSION_TOKEN, entry["lat"], entry["lon"], ZOOM, OUTPUT_FOLDER,
                                  entry["name"])