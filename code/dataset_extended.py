

import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance en kilomètres entre deux points (lat, lon)
    en utilisant la formule de Haversine.
    """
    R = 6371.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Différences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return round(distance, 2)

def generate_massive_flight_dataset(target_size=100000):
    """
    Génère un dataset massif de vols internationaux avec distance de vol.
    """
    print("=" * 70)
    print(" GÉNÉRATION DATASET MASSIF")
    print("=" * 70)


    print("\n  Chargement des données sources...")

    routes = pd.read_csv(
        'https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat',
        header=None,
        names=['airline', 'airline_id', 'source', 'source_id', 
               'dest', 'dest_id', 'codeshare', 'stops', 'equipment']
    )

    airports = pd.read_csv(
        'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat',
        header=None,
        names=['id', 'name', 'city', 'country', 'iata', 'icao',
               'lat', 'lon', 'alt', 'timezone', 'dst', 'tz', 'type', 'source']
    )

    airlines = pd.read_csv(
        'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat',
        header=None,
        names=['id', 'name', 'alias', 'iata', 'icao', 'callsign', 'country', 'active']
    )

    print(f"   ✓ {len(routes):,} routes")
    print(f"   ✓ {len(airports):,} aéroports")
    print(f"   ✓ {len(airlines):,} compagnies")


    print("\n  Préparation des mappings...")

    airports_valid = airports[
        (airports['iata'] != '\\N') & 
        airports['iata'].notna() &
        (airports['type'] == 'airport')
    ].copy()

    airport_info = {}
    for _, row in airports_valid.iterrows():
        airport_info[row['iata']] = {
            'city': row['city'],
            'country': row['country'],
            'lat': row['lat'],
            'lon': row['lon'],
            'timezone': row['timezone']
        }

    airline_info = {}
    for _, row in airlines.iterrows():
        if pd.notna(row['iata']) and row['iata'] != '\\N':
            airline_info[row['iata']] = {
                'name': row['name'] if pd.notna(row['name']) else row['iata'],
                'country': row['country'] if pd.notna(row['country']) else 'Unknown'
            }


    print("\n Filtrage des routes internationales...")

    def enrich_route(row):
        src = row['source']
        dst = row['dest']
        al = row['airline']

        if src not in airport_info or dst not in airport_info:
            return None
        if al not in airline_info:
            return None

        src_info = airport_info[src]
        dst_info = airport_info[dst]

        if src_info['country'] == dst_info['country']:
            return None

        distance_km = haversine_distance(
            src_info['lat'], src_info['lon'],
            dst_info['lat'], dst_info['lon']
        )

        return {
            'airline_code': al,
            'airline_name': airline_info[al]['name'],
            'airline_country': airline_info[al]['country'],
            'origin_airport': src,
            'origin_city': src_info['city'],
            'origin_country': src_info['country'],
            'origin_lat': src_info['lat'],
            'origin_lon': src_info['lon'],
            'destination_airport': dst,
            'destination_city': dst_info['city'],
            'destination_country': dst_info['country'],
            'dest_lat': dst_info['lat'],
            'dest_lon': dst_info['lon'],
            'distance_km': distance_km, 
            'num_stops': 0 if row['stops'] == 0 else int(row['stops']),
            'aircraft': row['equipment'] if pd.notna(row['equipment']) else 'Unknown'
        }

    print("   Enrichissement des routes...")
    enriched = []
    for _, row in routes.iterrows():
        result = enrich_route(row)
        if result:
            enriched.append(result)

    routes_df = pd.DataFrame(enriched)
    print(f"   ✓ {len(routes_df):,} routes internationales valides")

    print(f"\n    DISTANCES :")
    print(f"   Distance moyenne: {routes_df['distance_km'].mean():.0f} km")
    print(f"   Distance min: {routes_df['distance_km'].min():.0f} km")
    print(f"   Distance max: {routes_df['distance_km'].max():.0f} km")


    print(f"\n Génération de {target_size:,} vols uniques...")

    n_routes = len(routes_df)
    multiplier = (target_size // n_routes) + 1

    all_flights = []
    base_date = datetime.now()

    cabin_classes = ['Economy'] * 70 + ['Premium Economy'] * 15 + ['Business'] * 10 + ['First'] * 5

    for idx in range(multiplier):
        if idx % 10 == 0:
            print(f"   Génération batch {idx+1}/{multiplier}...")

        batch = routes_df.copy()

        batch['flight_date'] = [
            (base_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
            for _ in range(len(batch))
        ]

        hours = list(range(6, 24)) + list(range(0, 6))
        weights = [3]*6 + [5]*6 + [4]*6 + [2]*6
        batch['departure_hour'] = random.choices(hours, weights=weights, k=len(batch))
        batch['departure_minute'] = [random.choice([0, 15, 30, 45]) for _ in range(len(batch))]

        def calc_duration(distance_km):
            if distance_km <= 0:
                return "Unknown", 0

            if distance_km < 500: 
                speed = random.uniform(700, 800)
                taxi_time = random.uniform(0.5, 0.8)
            elif distance_km < 2000:  
                speed = random.uniform(850, 950)
                taxi_time = random.uniform(0.8, 1.2)
            else:  
                speed = random.uniform(900, 1000)
                taxi_time = random.uniform(1.0, 1.5)

            duration_hours = (distance_km / speed) + taxi_time

            hours = int(duration_hours)
            minutes = int((duration_hours - hours) * 60)
            return f"{hours}h {minutes:02d}m", round(duration_hours, 2)

        durations = batch['distance_km'].apply(calc_duration)
        batch['flight_duration'] = [x[0] for x in durations]
        batch['duration_hours'] = [x[1] for x in durations]

        def calc_price(distance, row):
            cabin = random.choice(cabin_classes)

            if distance < 500:
                base_rate = random.uniform(0.20, 0.40) 
            elif distance < 2000:
                base_rate = random.uniform(0.10, 0.20)
            else:
                base_rate = random.uniform(0.05, 0.12)  

            base_price = distance * base_rate if distance > 0 else random.uniform(200, 500)

            class_mult = {
                'Economy': 1.0,
                'Premium Economy': 1.6,
                'Business': 3.5,
                'First': 8.0
            }[cabin]

            region_mult = 1.0
            expensive_countries = ['Switzerland', 'United Kingdom', 'Japan', 'Singapore', 'Australia', 'France']
            cheap_countries = ['India', 'Thailand', 'Egypt', 'Turkey', 'Mexico', 'Brazil', 'Morocco']

            if row['origin_country'] in expensive_countries or row['destination_country'] in expensive_countries:
                region_mult = random.uniform(1.2, 1.5)
            elif row['origin_country'] in cheap_countries or row['destination_country'] in cheap_countries:
                region_mult = random.uniform(0.6, 0.9)

            date_obj = datetime.strptime(row['flight_date'], '%Y-%m-%d')
            month = date_obj.month
            if month in [6, 7, 8, 12]: 
                season_mult = random.uniform(1.2, 1.8)
            elif month in [1, 2, 3, 11]:  
                season_mult = random.uniform(0.7, 0.9)
            else:
                season_mult = random.uniform(0.9, 1.2)

            final_price = base_price * class_mult * region_mult * season_mult
            return round(max(final_price, 50), 2), cabin

        prices_cabins = batch.apply(lambda row: calc_price(row['distance_km'], row), axis=1)
        batch['price_usd'] = [x[0] for x in prices_cabins]
        batch['cabin_class'] = [x[1] for x in prices_cabins]

        batch['fare_type'] = random.choices(
            ['Non-Refundable', 'Semi-Flexible', 'Flexible', 'Full-Fare'],
            weights=[50, 30, 15, 5],
            k=len(batch)
        )

        batch['seats_available'] = random.choices(
            list(range(0, 20)) + list(range(20, 100)),
            weights=[5]*20 + [1]*80,
            k=len(batch)
        )

        batch['flight_number'] = [
            f"{row['airline_code']}{random.randint(10, 9999)}"
            for _, row in batch.iterrows()
        ]

        all_flights.append(batch)

        if len(pd.concat(all_flights)) >= target_size:
            break

    final_df = pd.concat(all_flights, ignore_index=True)

    if len(final_df) > target_size:
        final_df = final_df.sample(n=target_size, random_state=42)



    print("\n  Nettoyage final...")

    columns_order = [
        'flight_number',
        'airline_name',
        'airline_code',
        'airline_country',
        'origin_airport',
        'origin_city',
        'origin_country',
        'destination_airport',
        'destination_city',
        'destination_country',
        'distance_km',         
        'flight_date',
        'departure_hour',
        'departure_minute',
        'flight_duration',
        'duration_hours',      
        'num_stops',
        'cabin_class',
        'fare_type',
        'price_usd',
        'seats_available',
        'aircraft'
    ]

    final_df = final_df[columns_order].copy()

    final_df.columns = [
        'flight_number',
        'airline',
        'airline_code',
        'airline_country',
        'origin_airport',
        'origin_city',
        'origin_country',
        'dest_airport',
        'dest_city',
        'dest_country',
        'distance_km',          
        'flight_date',
        'departure_hour',
        'departure_minute',
        'duration',
        'duration_hours',
        'stops',
        'cabin_class',
        'fare_type',
        'price_usd',
        'seats_available',
        'aircraft_type'
    ]

    final_df['departure_time'] = final_df.apply(
        lambda x: f"{x['departure_hour']:02d}:{x['departure_minute']:02d}", axis=1
    )

    def categorize_distance(km):
        if km < 500:
            return 'Court-courrier (< 500 km)'
        elif km < 2000:
            return 'Moyen-courrier (500-2000 km)'
        elif km < 6000:
            return 'Long-courrier (2000-6000 km)'
        else:
            return 'Ultra long-courrier (> 6000 km)'

    final_df['flight_category'] = final_df['distance_km'].apply(categorize_distance)



    print("\n" + "=" * 70)
    print(" DATASET CRÉÉ AVEC SUCCÈS")
    print("=" * 70)

    print(f"\n DIMENSIONS : {len(final_df):,} vols × {len(final_df.columns)} colonnes")

    print(f"\n STATISTIQUES DE DISTANCE :")
    print(f"   • Distance moyenne: {final_df['distance_km'].mean():.0f} km")
    print(f"   • Distance médiane: {final_df['distance_km'].median():.0f} km")
    print(f"   • Distance min: {final_df['distance_km'].min():.0f} km")
    print(f"   • Distance max: {final_df['distance_km'].max():.0f} km")

    print(f"\n CATÉGORIES DE VOL :")
    print(final_df['flight_category'].value_counts().to_string())

    print(f"\n COUVERTURE GÉOGRAPHIQUE :")
    print(f"   • Pays d'origine : {final_df['origin_country'].nunique()}")
    print(f"   • Pays destination : {final_df['dest_country'].nunique()}")
    print(f"   • Villes d'origine : {final_df['origin_city'].nunique()}")
    print(f"   • Aéroports : {final_df['origin_airport'].nunique()}")

    print(f"\n COMPAGNIES :")
    print(f"   • Nombre de compagnies : {final_df['airline'].nunique()}")
    print(f"   • Top 5 :")
    for airline, count in final_df['airline'].value_counts().head(5).items():
        print(f"     - {airline}: {count:,} vols")

    print(f"\n PRIX (USD) :")
    price_stats = final_df['price_usd'].describe()
    print(f"   • Min: ${price_stats['min']:.2f}")
    print(f"   • Max: ${price_stats['max']:.2f}")
    print(f"   • Moyenne: ${price_stats['mean']:.2f}")
    print(f"   • Médiane: ${price_stats['50%']:.2f}")

    correlation = final_df['distance_km'].corr(final_df['price_usd'])
    print(f"\n CORRÉLATION DISTANCE-PRIX : {correlation:.3f}")

    print(f"\n RÉPARTITION PAR CLASSE :")
    print(final_df['cabin_class'].value_counts().to_string())

    print(f"\n RÉPARTITION PAR MOIS :")
    final_df['month'] = pd.to_datetime(final_df['flight_date']).dt.month
    print(final_df['month'].value_counts().sort_index().to_string())

    print(f"\n TOP 10 ROUTES LES PLUS LONGUES (en km) :")
    longest_routes = final_df.nlargest(10, 'distance_km')[['origin_city', 'origin_country', 
                                                            'dest_city', 'dest_country',
                                                            'distance_km', 'duration']]
    for _, row in longest_routes.iterrows():
        print(f"   {row['origin_city']} ({row['origin_country']}) → {row['dest_city']} ({row['dest_country']}): {row['distance_km']:,.0f} km ({row['duration']})")

    print(f"\n🔝 TOP 10 ROUTES LES PLUS FRÉQUENTES :")
    route_counts = final_df.groupby(['origin_city', 'dest_city']).size().sort_values(ascending=False).head(10)
    for (orig, dest), count in route_counts.items():
        avg_dist = final_df[(final_df['origin_city'] == orig) & (final_df['dest_city'] == dest)]['distance_km'].mean()
        print(f"   {orig} → {dest}: {count:,} vols (avg {avg_dist:.0f} km)")

    print(f"\n TOP 10 VOLS LES PLUS CHERS :")
    expensive = final_df.nlargest(10, 'price_usd')[['origin_city', 'origin_country', 
                                                     'dest_city', 'dest_country',
                                                     'distance_km', 'airline', 'cabin_class', 'price_usd']]
    print(expensive.to_string())

    print(f"\n TOP 10 VOLS LES MOINS CHERS :")
    cheap = final_df.nsmallest(10, 'price_usd')[['origin_city', 'origin_country', 
                                                  'dest_city', 'dest_country',
                                                  'distance_km', 'airline', 'cabin_class', 'price_usd']]
    print(cheap.to_string())

    filename = f'flights_dataset_{len(final_df)}.csv'
    final_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n💾 FICHIER PRINCIPAL : {filename}")

    excel_filename = 'flights_sample_10000.xlsx'
    final_df.head(10000).to_excel(excel_filename, index=False, engine='openpyxl')
    print(f"💾 FICHIER EXCEL (échantillon) : {excel_filename}")

    json_filename = 'flights_sample_5000.json'
    final_df.head(5000).to_json(json_filename, orient='records', indent=2)
    print(f"💾 FICHIER JSON (échantillon) : {json_filename}")

    stats_distance = 'stats_by_distance.csv'
    distance_stats = final_df.groupby('flight_category').agg({
        'price_usd': ['mean', 'min', 'max'],
        'duration_hours': 'mean',
        'flight_number': 'count'
    }).round(2)
    distance_stats.to_csv(stats_distance)
    print(f"💾 STATISTIQUES PAR DISTANCE : {stats_distance}")

    return final_df


if __name__ == "__main__":
    df = generate_massive_flight_dataset(target_size=100000)

    print("\n" + "=" * 70)
    print("🎉 TERMINÉ !")
    print("=" * 70)
    print("\nVous avez maintenant plusieurs fichiers :")
    print("  • CSV complet avec distances (100k+ vols)")
    print("  • Excel (10k vols, facile à ouvrir)")
    print("  • JSON (5k vols, pour développeurs)")

    print("  • Stats par catégorie de distance")

