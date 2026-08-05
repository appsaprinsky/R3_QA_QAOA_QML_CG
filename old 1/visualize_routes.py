import json
import os
import requests
import polyline
import folium
import pandas as pd

def load_amazon_dataset(data_dir):
    """Loads route metadata and actual driver sequences."""
    build_inputs = os.path.join(data_dir, "model_build_inputs")
    route_data_path = os.path.join(build_inputs, "route_data.json")
    actual_seq_path = os.path.join(build_inputs, "actual_sequences.json")

    print("Loading JSON dataset files...")
    with open(route_data_path, 'r') as f:
        routes = json.load(f)
    with open(actual_seq_path, 'r') as f:
        sequences = json.load(f)

    return routes, sequences

def get_osrm_street_geometry(coords_list):
    """
    Queries the public OSRM Routing Engine to retrieve exact road geometries
    between consecutive coordinates (lon, lat).
    """
    if len(coords_list) < 2:
        return []

    # OSRM expects coordinates formatted as: {lng},{lat};{lng},{lat}...
    # Chunk long routes to keep URL lengths manageable for public endpoints
    chunk_size = 25
    full_road_path = []

    for i in range(0, len(coords_list) - 1, chunk_size - 1):
        chunk = coords_list[i : i + chunk_size]
        loc_str = ";".join([f"{lng:.6f},{lat:.6f}" for lat, lng in chunk])
        url = f"http://router.project-osrm.org/route/v1/driving/{loc_str}?overview=full&geometries=polyline"

        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "routes" in data and len(data["routes"]) > 0:
                    encoded_geom = data["routes"][0]["geometry"]
                    # Decode Polyline into [(lat, lng), ...] format for Folium
                    decoded_coords = polyline.decode(encoded_geom)
                    full_road_path.extend(decoded_coords)
            else:
                print(f"Warning: OSRM request returned status {res.status_code}. Falling back to straight lines.")
                full_road_path.extend(chunk)
        except Exception as e:
            print(f"OSRM request failed ({e}). Falling back to straight lines.")
            full_road_path.extend(chunk)

    return full_road_path

def build_route_dataframe(route_info, actual_seq):
    """Extracts stop coordinates (lat, lng) ordered by driver sequence."""
    stops = route_info.get('stops', {})
    actual_seq_dict = actual_seq.get('actual', {})

    records = []
    for stop_id, stop_info in stops.items():
        seq_num = actual_seq_dict.get(stop_id, float('inf'))
        lat = stop_info.get('lat')
        lng = stop_info.get('lng')
        zone_id = stop_info.get('zone_id', stop_id)
        stop_type = stop_info.get('type', 'dropoff')

        if lat is not None and lng is not None:
            records.append({
                'stop_id': stop_id,
                'seq_num': seq_num,
                'lat': float(lat),
                'lng': float(lng),
                'zone_id': zone_id,
                'type': stop_type
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(by='seq_num').reset_index(drop=True)
    return df

def generate_route_map(data_dir, max_routes_to_plot=2, output_html="amazon_street_routes_map.html"):
    routes, sequences = load_amazon_dataset(data_dir)
    route_ids = list(routes.keys())

    valid_route_dfs = []
    map_center = None

    for r_id in route_ids:
        r_info = routes[r_id]
        r_seq = sequences.get(r_id, {})
        df = build_route_dataframe(r_info, r_seq)

        if not df.empty:
            if map_center is None:
                map_center = [df.iloc[0]['lat'], df.iloc[0]['lng']]
            valid_route_dfs.append((r_id, df, r_info))
            if len(valid_route_dfs) >= max_routes_to_plot:
                break

    if not valid_route_dfs:
        print("Error: Could not extract latitude/longitude coordinates.")
        return

    print(f"Plotting top {len(valid_route_dfs)} routes along actual street networks via OSRM...")

    m = folium.Map(location=map_center, zoom_start=13, tiles='OpenStreetMap')
    colors = ['blue', 'red', 'purple', 'green', 'orange', 'darkred']

    for idx, (r_id, df, r_info) in enumerate(valid_route_dfs):
        color = colors[idx % len(colors)]
        station_code = r_info.get('station_code', 'Unknown')
        route_group = folium.FeatureGroup(name=f"Route: {r_id[:8]}... ({station_code})")

        # 1. Fetch street navigation geometry from OSRM
        stop_coords = df[['lat', 'lng']].values.tolist()
        print(f"Fetching road geometry for Route {r_id[:8]}... ({len(stop_coords)} stops)")
        street_path = get_osrm_street_geometry(stop_coords)

        # 2. Draw actual street driving polyline
        folium.PolyLine(
            street_path,
            weight=4,
            color=color,
            opacity=0.85,
            tooltip=f"Route {r_id[:8]} ({station_code})"
        ).add_to(route_group)

        # 3. Add Stop Markers along the road
        for i, row in df.iterrows():
            is_start = (row['type'] == 'station' or i == 0)
            marker_color = 'black' if is_start else color
            seq_str = "Depot/Start" if is_start else f"Stop #{int(row['seq_num'])}"

            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=7 if is_start else 4,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.9,
                popup=f"<b>Route:</b> {r_id[:12]}...<br><b>Sequence:</b> {seq_str}<br><b>Zone:</b> {row['zone_id']}"
            ).add_to(route_group)

        route_group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"\nStreet-level map generated! Open '{output_html}' in your browser.")

if __name__ == "__main__":
    DATA_PATH = "./almrrc2021-data-training"
    if os.path.exists(DATA_PATH):
        # Recommend keeping max_routes_to_plot small (1-3) to keep OSRM API calls snappy
        generate_route_map(DATA_PATH, max_routes_to_plot=7)
    else:
        print(f"Dataset directory '{DATA_PATH}' not found.")