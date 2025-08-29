from fastapi import FastAPI, HTTPException, Response, Request, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import base64
import numpy as np
import pandas as pd
import requests
from itertools import combinations
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time
import weakref
import csv
import folium
import polyline
import itertools
import numpy as np
from folium.plugins import MarkerCluster
import sys
import os
import hashlib
import math

app = FastAPI(title="VRP API", description="API pour la résolution du problème de tournées de véhicules (VRP) avec OR-Tools et OSRM")

# Upload dataset endpoint
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Reçoit un fichier dataset (xlsx ou csv), le stocke sous uploads/latest.xlsx ou uploads/latest.csv
    et renvoie un court résumé.
    """
    try:
        os.makedirs('uploads', exist_ok=True)
        filename = file.filename or ''
        lower = filename.lower()
        if lower.endswith('.xlsx'):
            target = os.path.join('uploads', 'latest.xlsx')
            # Supprimer l'ancien CSV si présent pour éviter l'ambiguïté
            other = os.path.join('uploads', 'latest.csv')
            if os.path.exists(other):
                try:
                    os.remove(other)
                except Exception:
                    pass
        elif lower.endswith('.csv'):
            target = os.path.join('uploads', 'latest.csv')
            # Supprimer l'ancien XLSX si présent pour éviter l'ambiguïté
            other = os.path.join('uploads', 'latest.xlsx')
            if os.path.exists(other):
                try:
                    os.remove(other)
                except Exception:
                    pass
        else:
            raise HTTPException(status_code=400, detail="Format non supporté: utilisez .xlsx ou .csv")

        # Écrire le contenu sur disque
        content = await file.read()
        with open(target, 'wb') as f:
            f.write(content)

        # Validation rapide des colonnes attendues
        try:
            if target.endswith('.xlsx'):
                df = pd.read_excel(target)
            else:
                df = pd.read_csv(target)
            cols = set(df.columns.astype(str))
            required = { 'PARTNER_CODE', 'LATITUDE', 'LONGITUDE', 'WEIGHT' }
            if not required.issubset(cols):
                missing = sorted(list(required - cols))
                raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {', '.join(missing)}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Fichier invalide: {e}")

        return { "success": True, "message": "Dataset chargé", "rows": int(len(df)) }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Modèles Pydantic pour l'API
class MatrixRequest(BaseModel):
    depot_lat: float = 33.604427
    depot_lng: float = -7.558631

class VRPRequest(BaseModel):
    num_vehicles: int = 36
    vehicle_capacity: float = 4000.0    # en kg
    service_time: int = 5               # minutes par client
    time_limit: int = 300               # secondes
    vehicle_time_limit: int = 480       # limite en minutes par véhicule
    interval_seconds: int = 60          # intervalle d'affichage des solutions intermédiaires

class MatrixResponse(BaseModel):
    success: bool
    message: str
    execution_time: float
    num_points: int
    up_to_date: bool = False

class VRPResponse(BaseModel):
    success: bool
    message: str
    vehicles_used: int
    total_distance: float
    total_load: float
    routes: List[Dict[str, Any]]
    execution_time: float

# Variables globales pour stocker les données
clients_df = None
distance_matrix = None
time_matrix = None

@app.get("/")
async def root():
    return {"message": "VRP API - API pour la résolution du problème de tournées de véhicules"}

@app.post("/generate-matrices", response_model=MatrixResponse)
async def generate_matrices(request: MatrixRequest, response: Response):
    """
    Génère les matrices de distances et de temps avec OSRM et les stocke dans les cookies
    """
    global clients_df, distance_matrix, time_matrix
    
    start_time = time.time()
    
    try:
        # Étape 1 : Chargement des données des clients
        print("## Étape 1 : Chargement des données des clients")
        
        # Sélection du dataset: utiliser le dernier fichier uploadé s'il existe, sinon fallback
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        uploaded_xlsx = os.path.join(upload_dir, 'latest.xlsx')
        uploaded_csv = os.path.join(upload_dir, 'latest.csv')

        if os.path.exists(uploaded_xlsx):
            clients_df = pd.read_excel(uploaded_xlsx)
        elif os.path.exists(uploaded_csv):
            clients_df = pd.read_csv(uploaded_csv)
        else:
            # Fallback legacy dataset
            clients_df = pd.read_excel("clients/casa-nord.xlsx")

        # Renommer les colonnes pour correspondre au format attendu
        clients_df = clients_df.rename(columns={
            'PARTNER_CODE': 'id',
            'LATITUDE': 'lat', 
            'LONGITUDE': 'lng',
            'WEIGHT': 'weight'
        })

        # Ajouter le dépôt (à Casablanca) au début
        depot = pd.DataFrame([{
            'id': 'DEPOT', 
            'lat': request.depot_lat, 
            'lng': request.depot_lng, 
            'weight': 0
        }])

        clients_df = pd.concat([depot, clients_df], ignore_index=True)

        # Supprimer les points avec des coordonnées == 0
        clients_df = clients_df[~((clients_df['lat'] == 0) & (clients_df['lng'] == 0))]
        # Réinitialiser les index
        clients_df = clients_df.reset_index(drop=True)

        print(f"Nombre total de points (dépôt + clients): {len(clients_df)}")
        print(f"Poids total à livrer: {clients_df['weight'][1:].sum():.2f} kg -- {clients_df['weight'][1:].sum() / 1000:.2f} tonnes")

        # Étape 2 : Calcul de la matrice de distances et de temps avec OSRM
        print("## Étape 2 : Calcul de la matrice de distances et de temps avec OSRM")

        # Tentative d'optimisation incrémentale si des matrices existent déjà
        try:
            prev_clients_path = 'matrices/clients_processed.csv'
            dist_path = 'matrices/distance_matrix.csv'
            time_path = 'matrices/time_matrix.csv'
            incremental_done = False

            if os.path.exists(prev_clients_path) and os.path.exists(dist_path) and os.path.exists(time_path):
                prev_df = pd.read_csv(prev_clients_path)

                # Vérifier que les anciens points sont inclus et que leurs coordonnées n'ont pas changé
                prev_map = {str(row['id']): (float(row['lat']), float(row['lng'])) for _, row in prev_df.iterrows()}
                curr_map = {str(row['id']): (float(row['lat']), float(row['lng'])) for _, row in clients_df.iterrows()}

                prev_ids = list(prev_df['id'].astype(str))
                curr_ids = list(clients_df['id'].astype(str))

                # Les anciens doivent exister dans le nouveau, aux mêmes coordonnées (avec tolérance)
                def same_coords(a, b, tol=1e-6):
                    try:
                        return abs(a[0]-b[0]) <= tol and abs(a[1]-b[1]) <= tol
                    except Exception:
                        return False
                removable_or_changed = [
                    pid for pid in prev_ids
                    if (pid not in curr_map) or (not same_coords(curr_map[pid], prev_map[pid]))
                ]
                if len(removable_or_changed) == 0:
                    # Déterminer les nouveaux points (ajouts uniquement)
                    new_ids = [cid for cid in curr_ids if cid not in prev_map]

                    if len(new_ids) == 0:
                        # Rien à faire: matrices identiques, renvoyer succès rapide
                        exec_time = time.time() - start_time
                        # Écrire/mettre à jour le résumé
                        try:
                            summary_payload = {
                                "type": "matrices",
                                "success": True,
                                "message": "Matrices déjà à jour (aucun nouveau point)",
                                "execution_time": round(exec_time, 2),
                                "num_points": len(clients_df),
                                "generated_at": time.time(),
                            }
                            with open('matrices/summary.json', 'w') as f:
                                json.dump(summary_payload, f)
                        except Exception:
                            pass
                        return MatrixResponse(
                            success=True,
                            message="Matrices inchangées: réutilisation",
                            execution_time=round(exec_time, 2),
                            num_points=len(clients_df),
                            up_to_date=True,
                        )

                    # Réordonner: anciens d'abord (dans l'ordre précédent), puis nouveaux (dans l'ordre actuel)
                    old_rows = clients_df.set_index(clients_df['id'].astype(str)).loc[prev_ids].reset_index(drop=True)
                    new_rows = clients_df[clients_df['id'].astype(str).isin(new_ids)].reset_index(drop=True)
                    clients_df = pd.concat([old_rows, new_rows], ignore_index=True)

                    # Charger matrices existantes
                    distance_df = pd.read_csv(dist_path)
                    time_df = pd.read_csv(time_path)
                    distance_matrix = distance_df.values.tolist()
                    time_matrix = time_df.values.tolist()

                    M = len(prev_ids)
                    N = len(clients_df)
                    K = N - M

                    # Étendre les matrices à N x N
                    for row in distance_matrix:
                        row.extend([0] * K)
                    for row in time_matrix:
                        row.extend([0] * K)
                    distance_matrix.extend([[0] * N for _ in range(K)])
                    time_matrix.extend([[0] * N for _ in range(K)])

                    # Préparer les points (lat,lng) dans l'ordre courant
                    coords = [(float(row['lat']), float(row['lng'])) for _, row in clients_df.iterrows()]

                    def osrm_table_rect(src_idx, dst_idx, max_src_chunk=25, max_dst_chunk=50, timeout=20, retries=2):
                        """Récupère la sous-matrice distances (km) et temps (min) pour src_idx x dst_idx.
                        Pour éviter les URLs trop longues, on découpe les sources et les destinations.
                        Retourne deux matrices de taille len(src_idx) x len(dst_idx).
                        """
                        S = len(src_idx)
                        D = len(dst_idx)
                        # Préparer buffers de sortie
                        out_dist = [[0 for _ in range(D)] for _ in range(S)]
                        out_time = [[0 for _ in range(D)] for _ in range(S)]

                        # Calcul du nombre total de tuiles pour logs
                        total_src_tiles = max(1, math.ceil(S / max_src_chunk))
                        total_dst_tiles = max(1, math.ceil(D / max_dst_chunk))
                        tiles_total = total_src_tiles * total_dst_tiles
                        tile_idx = 0
                        print(f"[Incrémental] Début calcul d'un bloc: {S} sources x {D} destinations en {tiles_total} tuiles")

                        # Boucler sur tuiles de sources et destinations
                        for si in range(0, S, max_src_chunk):
                            src_slice_idx = src_idx[si:si + max_src_chunk]
                            for dj in range(0, D, max_dst_chunk):
                                dst_slice_idx = dst_idx[dj:dj + max_dst_chunk]
                                tile_idx += 1

                                # Points locaux: sources suivies des destinations
                                local_points = [coords[s] for s in src_slice_idx] + [coords[d] for d in dst_slice_idx]
                                local_sources = list(range(0, len(src_slice_idx)))
                                local_dests = list(range(len(src_slice_idx), len(src_slice_idx) + len(dst_slice_idx)))

                                coord_str = ";".join([f"{lng},{lat}" for (lat, lng) in local_points])
                                src_param = ";".join(map(str, local_sources))
                                dst_param = ";".join(map(str, local_dests))
                                url = (
                                    f"http://localhost:5001/table/v1/driving/{coord_str}"
                                    f"?annotations=distance,duration&sources={src_param}&destinations={dst_param}"
                                )

                                percent = int((tile_idx / tiles_total) * 100)
                                print(f"[Incrémental] Traitement du lot — tuile {tile_idx}/{tiles_total} ({percent}%) ...")

                                attempt = 0
                                success = False
                                last_err = None
                                while attempt <= retries and not success:
                                    try:
                                        r = requests.get(url, timeout=timeout)
                                        if r.status_code == 200:
                                            data = r.json()
                                            dists = [[round((d or 0) / 1000, 2) for d in row] for row in data['distances']]
                                            durs = [[round((t or 0) / 60, 2) for t in row] for row in data['durations']]
                                            success = True
                                        else:
                                            last_err = f"HTTP {r.status_code}"
                                            attempt += 1
                                            time.sleep(0.5)
                                    except Exception as e:
                                        last_err = str(e)
                                        attempt += 1
                                        time.sleep(0.5)

                                if not success:
                                    # En cas d'échec, laisser 0 sur cette tuile et log
                                    print(f"[Incrémental] [ECHEC] tuile {tile_idx}/{tiles_total} : {last_err}")
                                    continue
                                else:
                                    print(f"[Incrémental] [OK] tuile {tile_idx}/{tiles_total}")

                                # Écrire dans les buffers de sortie
                                for li, global_si in enumerate(range(si, si + len(src_slice_idx))):
                                    for lj, global_dj in enumerate(range(dj, dj + len(dst_slice_idx))):
                                        out_dist[global_si][global_dj] = dists[li][lj]
                                        out_time[global_si][global_dj] = durs[li][lj]

                        return out_dist, out_time

                    print(f"## Incrémental: M(anciens)={M}, K(nouveaux)={K}, N(total)={N}")

                    # 1) anciens -> nouveaux (bloc M x K)
                    if K > 0:
                        print("[Incrémental] Bloc 1/3: anciens -> nouveaux (M x K)")
                        src_idx = list(range(M))
                        dst_idx = list(range(M, N))
                        dist_block, time_block = osrm_table_rect(src_idx, dst_idx)
                        for i in range(M):
                            for j, val in enumerate(dist_block[i]):
                                distance_matrix[i][M + j] = val
                        for i in range(M):
                            for j, val in enumerate(time_block[i]):
                                time_matrix[i][M + j] = val

                        print("[Incrémental] Bloc 2/3: nouveaux -> anciens (K x M)")
                        # 2) nouveaux -> anciens (bloc K x M)
                        src_idx = list(range(M, N))
                        dst_idx = list(range(0, M))
                        dist_block, time_block = osrm_table_rect(src_idx, dst_idx)
                        for i in range(K):
                            for j, val in enumerate(dist_block[i]):
                                distance_matrix[M + i][j] = val
                        for i in range(K):
                            for j, val in enumerate(time_block[i]):
                                time_matrix[M + i][j] = val

                        print("[Incrémental] Bloc 3/3: nouveaux -> nouveaux (K x K)")
                        # 3) nouveaux -> nouveaux (bloc K x K)
                        src_idx = list(range(M, N))
                        dst_idx = list(range(M, N))
                        dist_block, time_block = osrm_table_rect(src_idx, dst_idx)
                        for i in range(K):
                            for j, val in enumerate(dist_block[i]):
                                distance_matrix[M + i][M + j] = val
                        for i in range(K):
                            for j, val in enumerate(time_block[i]):
                                time_matrix[M + i][M + j] = val

                        print("[Incrémental] Remplissage des blocs terminé. Sauvegarde des matrices...")

                    # Sauvegarder matrices et clients réordonnés
                    pd.DataFrame(distance_matrix).to_csv(dist_path, index=False)
                    pd.DataFrame(time_matrix).to_csv(time_path, index=False)
                    clients_df.to_csv(prev_clients_path, index=False)

                    # Écrire un résumé persistant
                    try:
                        summary_payload = {
                            "type": "matrices",
                            "success": True,
                            "message": f"Mise à jour incrémentale: +{K} nouveaux points",
                            "execution_time": round(time.time() - start_time, 2),
                            "num_points": len(clients_df),
                            "generated_at": time.time(),
                        }
                        with open('matrices/summary.json', 'w') as f:
                            json.dump(summary_payload, f)
                    except Exception:
                        pass

                    incremental_done = True

            if incremental_done:
                return MatrixResponse(
                    success=True,
                    message="Matrices mises à jour incrémentalement",
                    execution_time=round(time.time() - start_time, 2),
                    num_points=len(clients_df),
                    up_to_date=False,
                )
        except Exception as _:
            # En cas de souci, on retombe sur la génération complète
            pass
        
        # Nombre total de points (dépôt + clients)
        N = len(clients_df)

        # Initialiser la matrice de distances avec des zéros (en km)
        distance_matrix = [[0 for _ in range(N)] for _ in range(N)]

        # Initialiser la matrice de temps (en minutes)
        time_matrix = [[0 for _ in range(N)] for _ in range(N)]

        # Définir la taille des lots (réduite à 50 pour éviter les limites d'OSRM)
        batch_size = 50

        # Créer des lots d'indices de clients (1 à N-1, car 0 est le dépôt)
        client_indices = list(range(1, N))
        batches = [client_indices[i:i + batch_size] for i in range(0, len(client_indices), batch_size)]

        # Fonction pour valider les coordonnées
        def validate_coords(lng, lat):
            # Validation de base : vérifier si les coordonnées sont dans des limites raisonnables
            return -180 <= lng <= 180 and -90 <= lat <= 90

        # Fonction pour faire la requête OSRM et retourner les sous-matrices distance (km) et temps (min)
        def get_sub_matrices(points):
            # Extraire et valider les coordonnées
            coords_list = []
            for p in points:
                lng, lat = clients_df['lng'].iloc[p], clients_df['lat'].iloc[p]
                if not validate_coords(lng, lat):
                    print(f"Coordonnées invalides pour le point {p} : ({lat}, {lng})")
                    return None, None
                coords_list.append(f"{lng},{lat}")

            coords = ";".join(coords_list)
            url = f"http://localhost:5001/table/v1/driving/{coords}?annotations=distance,duration"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    sub_distances = [[round(d / 1000, 2) if d is not None else 0 for d in row] for row in data['distances']]  # km
                    sub_durations = [[round(t / 60, 2) if t is not None else 0 for t in row] for row in data['durations']]  # min
                    return sub_distances, sub_durations
                else:
                    print(f"Erreur OSRM : {response.status_code} pour les points {points}")
                    print(f"Réponse : {response.text}")
                    return None, None
            except requests.exceptions.RequestException as e:
                print(f"Échec de la requête pour les points {points} : {e}")
                return None, None

        # Étape 1 : Requêtes pour le dépôt + chaque lot
        for batch_idx, batch in enumerate(batches):
            print(f"Traitement du lot {batch_idx + 1}/{len(batches)}")
            points = [0] + batch  # Inclure le dépôt (index 0) + le lot
            sub_distances, sub_durations = get_sub_matrices(points)
            if sub_distances is None:
                print(f"Lot {batch_idx + 1} ignoré à cause d'une erreur")
                continue
            
            # r est l'index d'une ligne (le point de départ)
            # c est l'index d'une colonne (le point d'arrivée)
            for r in range(len(points)):
                for c in range(len(points)):
                    if r == 0 and c == 0:
                        distance_matrix[0][0] = sub_distances[r][c]  # Dépôt vers dépôt
                        time_matrix[0][0] = sub_durations[r][c]  # Temps dépôt vers dépôt
                    elif r == 0:
                        distance_matrix[0][points[c]] = sub_distances[r][c]  # Dépôt vers client
                        time_matrix[0][points[c]] = sub_durations[r][c]  # Temps dépôt vers client
                    elif c == 0:
                        distance_matrix[points[r]][0] = sub_distances[r][c]  # Client vers dépôt
                        time_matrix[points[r]][0] = sub_durations[r][c]  # Temps client vers dépôt
                    else:
                        distance_matrix[points[r]][points[c]] = sub_distances[r][c]  # Entre clients du même lot
                        time_matrix[points[r]][points[c]] = sub_durations[r][c]  # Entre clients du même lot

        # Étape 2 : Requêtes pour chaque paire de lots
        # Utiliser des combinaisons pour éviter les doublons 
        # (On utilise combinations pour ne pas répéter (1,2) et (2,1), car la distance est la même)
        for b1_idx, b2_idx in combinations(range(len(batches)), 2):
            b1, b2 = batches[b1_idx], batches[b2_idx]
            print(f"Traitement de la paire de lots ({b1_idx + 1}, {b2_idx + 1})")
            points = b1 + b2
            sub_distances, sub_durations = get_sub_matrices(points)
            if sub_distances is None:
                print(f"Paire de lots ({b1_idx + 1}, {b2_idx + 1}) ignorée à cause d'une erreur")
                continue
            
            len_b1 = len(b1)
            for i in range(len_b1):
                for j in range(len(b2)):
                    distance_matrix[b1[i]][b2[j]] = sub_distances[i][len_b1 + j]  # b1 vers b2
                    distance_matrix[b2[j]][b1[i]] = sub_distances[len_b1 + j][i]  # b2 vers b1 (supposé symétrique)

                    time_matrix[b1[i]][b2[j]] = sub_durations[i][len_b1 + j]  # Temps b1 vers b2
                    time_matrix[b2[j]][b1[i]] = sub_durations[len_b1 + j][i]  # Temps b2 vers b1 (supposé symétrique)

        # Vérifier si la matrice de distances est complètement remplie
        non_zero_count = sum(1 for row in distance_matrix for d in row if d != 0)
        # Verifier si la matrice de temps est complètement remplie
        non_zero_time_count = sum(1 for row in time_matrix for t in row if t != 0)
        print(f"Matrice de distances remplie avec {non_zero_count} entrées non nulles")
        print(f"Matrice de temps remplie avec {non_zero_time_count} entrées non nulles")
        print(f"Exemple (dépôt -> premier client) : {distance_matrix[0][1]} km, {time_matrix[0][1]} min")

        # Créer le dossier matrices s'il n'existe pas
        os.makedirs('matrices', exist_ok=True)
        
        # Sauvegarder les matrices dans des fichiers CSV
        distance_df = pd.DataFrame(distance_matrix)
        time_df = pd.DataFrame(time_matrix)
        
        distance_df.to_csv('matrices/distance_matrix.csv', index=False)
        time_df.to_csv('matrices/time_matrix.csv', index=False)
        
        # Sauvegarder les données des clients traitées (avec dépôt ajouté et nettoyage)
        clients_df.to_csv('matrices/clients_processed.csv', index=False)
        # Écrire un résumé persistant pour permettre au frontend de restaurer l'état
        try:
            summary_payload = {
                "type": "matrices",
                "success": True,
                "message": "Matrices générées et stockées",
                "execution_time": round(time.time() - start_time, 2),
                "num_points": len(clients_df),
                "generated_at": time.time(),
            }
            with open('matrices/summary.json', 'w') as f:
                json.dump(summary_payload, f)
        except Exception as _:
            pass
        
        execution_time = time.time() - start_time
        
        return MatrixResponse(
            success=True,
            message="Matrices générées et stockées dans les fichiers CSV avec succès",
            execution_time=round(execution_time, 2),
            num_points=len(clients_df),
            up_to_date=False,
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"Erreur lors de la génération des matrices: {e}")
        
        return MatrixResponse(
            success=False,
            message=f"Erreur lors de la génération des matrices: {str(e)}",
            execution_time=round(execution_time, 2),
            num_points=0,
            up_to_date=False,
        )

@app.post("/solve-vrp", response_model=VRPResponse)
async def solve_vrp(request: VRPRequest, http_request: Request):
    """
    Résout le problème VRP avec OR-Tools en utilisant les matrices stockées dans les cookies
    """
    global clients_df, distance_matrix, time_matrix
    
    start_time = time.time()
    
    try:
        # Vérifier si les fichiers de matrices existent
        if not os.path.exists('matrices/distance_matrix.csv') or not os.path.exists('matrices/time_matrix.csv'):
            return VRPResponse(
                success=False,
                message="Matrices non trouvées dans les fichiers. Veuillez d'abord générer les matrices avec /generate-matrices",
                vehicles_used=0,
                total_distance=0,
                total_load=0,
                routes=[],
                execution_time=round(time.time() - start_time, 2)
            )
        
        # Charger les matrices depuis les fichiers CSV
        try:
            distance_df = pd.read_csv('matrices/distance_matrix.csv')
            time_df = pd.read_csv('matrices/time_matrix.csv')
            clients_df = pd.read_csv('matrices/clients_processed.csv')
            
            # Convertir en listes de listes
            distance_matrix = distance_df.values.tolist()
            time_matrix = time_df.values.tolist()
            
        except Exception as e:
            return VRPResponse(
                success=False,
                message=f"Erreur lors du chargement des matrices: {str(e)}",
                vehicles_used=0,
                total_distance=0,
                total_load=0,
                routes=[],
                execution_time=round(time.time() - start_time, 2)
            )
        
        # Extraire les demandes en kg -> grammes (int)
        # Si une ligne est vide, on met 0
        demands = (clients_df['weight'].fillna(0) * 1000).astype(int).tolist()

        # Définir les paramètres du VRP
        num_vehicles = request.num_vehicles
        depot_index = 0
        vehicle_capacities = [int(request.vehicle_capacity * 1000)] * num_vehicles  # Convertir en grammes

        # Dictionnaire de données
        data = {
            'distance_matrix': distance_matrix,   # km
            'time_matrix': time_matrix,           # minutes
            'demands': demands,                   # grammes
            'vehicle_capacities': vehicle_capacities,
            'num_vehicles': num_vehicles,
            'depot': depot_index,
        }

        # Créer l'index manager
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )

        # Créer le modèle de routage
        routing = pywrapcp.RoutingModel(manager)

        # Callback interactif à chaque solution
        class InteractiveSolutionCallback:
            """
            Appelé par OR-Tools à chaque solution; affiche distance totale et nb de véhicules,
            puis propose d'arrêter ou continuer. Filtré toutes les 'interval_seconds'.
            """
            def __init__(self, manager, routing_model, interval_seconds=60):
                self._mgr_ref = weakref.ref(manager)
                self._routing_ref = weakref.ref(routing_model)
                self._interval = interval_seconds
                self._last = 0.0

            def __call__(self):
                now = time.time()
                if now - self._last < self._interval:
                    return
                self._last = now

                routing_model = self._routing_ref()
                manager = self._mgr_ref()

                # Objectif courant (mètres, car on convertit km -> m dans le coût d'arc)
                try:
                    total_cost_m = int(routing_model.CostVar().Value())
                    # Soustraire les coûts fixes des véhicules pour obtenir seulement la distance
                    vehicles_used = 0
                    for v in range(manager.GetNumberOfVehicles()):
                        start = routing_model.Start(v)
                        if not routing_model.IsEnd(routing_model.NextVar(start).Value()):
                            vehicles_used += 1
                    total_distance_only = total_cost_m - (vehicles_used * fixed_cost)
                except Exception:
                    total_cost_m = None
                    total_distance_only = None

                # Afficher la solution intermédiaire (flush pour environnements non interactifs)
                print("\n--- Solution intermédiaire ---", flush=True)
                if total_distance_only is not None:
                    print(f"Distance totale (objectif) : {total_distance_only / 1000:.2f} km", flush=True)
                else:
                    print("Distance totale (objectif) : N/A", flush=True)
                print(f"Nombre de véhicules utilisés : {vehicles_used}", flush=True)
                print("Tapez 's' puis Entrée pour STOPPER et garder la solution actuelle,", flush=True)
                print("ou appuyez sur Entrée pour CONTINUER l'optimisation.", flush=True)
                
                try:
                    ans = input("Votre choix (s=stop, Entrée=continuer) : ").strip().lower()
                except Exception:
                    ans = ''

                if ans == 's':
                    routing_model.solver().FinishCurrentSearch()
                    print("Arrêt demandé : retour de la meilleure solution actuelle.")
                else:
                    print("Poursuite de la recherche...")

        # Définir le callback de distance
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['distance_matrix'][from_node][to_node] * 1000)  # Convertir km → m

        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

        # Définir le callback de temps
        SERVICE_TIME = request.service_time  # minutes par client

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = data['time_matrix'][from_node][to_node]
            # Ajouter temps de service seulement si ce n'est pas le dépôt
            service = SERVICE_TIME if from_node != data['depot'] else 0
            return int(travel_time + service)

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        routing.AddDimension(
            time_callback_index,
            0,     # pas de temps d'attente autorisé
            request.vehicle_time_limit,   # limite en minutes par véhicule
            True,  # commence à zéro
            'Time'
        )

        # Définir le callback de capacité
        def demand_callback(from_index):
            return data['demands'][manager.IndexToNode(from_index)]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # pas de slack, c'est-à-dire que les véhicules ne peuvent pas dépasser leur capacité
            data['vehicle_capacities'],
            True,
            'Capacity'
        )

        # Ajouter un coût fixe par véhicule pour encourager l'utilisation de moins de véhicules 
        # L'unité est la même que celle de l'arc-cost (ici mètres), donc 1_000_000 ≈ 1000 km
        fixed_cost = 1_000_000  # ≈ 1000 km

        for v in range(data['num_vehicles']):
            routing.SetFixedCostOfVehicle(fixed_cost, v)

        # Callback interactif branché
        solution_callback = InteractiveSolutionCallback(manager, routing, interval_seconds=request.interval_seconds)
        routing.AddAtSolutionCallback(solution_callback)

        # Paramètres de recherche
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.log_search = True
        search_parameters.time_limit.seconds = request.time_limit

        # Résoudre
        solution = routing.SolveWithParameters(search_parameters)

        # Affichage des résultats
        if solution:
            print("Solution trouvée :")
            total_distance = 0
            total_load = 0
            vehicles_used = 0
            routes_data = []
            
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id) # Index de départ pour le véhicule qui est le dépôt
                route = []
                route_distance = 0
                route_load = 0
                time_dimension = routing.GetDimensionOrDie('Time')

                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    client_id = clients_df.iloc[node_index]['id']
                    route.append(str(client_id))
                    route_load += data['demands'][node_index]
                    previous_index = index # Index précédent pour calculer la distance ensuite
                    index = solution.Value(routing.NextVar(index)) # Index suivant
                    # route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                    from_node = manager.IndexToNode(previous_index)
                    to_node = manager.IndexToNode(index)
                    route_distance += int(data['distance_matrix'][from_node][to_node] * 1000)
                if len(route) > 1:
                    vehicles_used += 1  # Compte le véhicule utilisé
                    route.append("RETOUR_DEPOT")
                    route_time = solution.Min(time_dimension.CumulVar(index))  # minutes totales
                    
                    route_info = {
                        "vehicle_id": vehicle_id + 1,
                        "route": route,
                        "distance_km": round(route_distance / 1000, 2),
                        "load_kg": round(route_load / 1000, 2),
                        "time_min": route_time
                    }
                    routes_data.append(route_info)
                    
                    print(f"Camion {vehicle_id + 1}: {' → '.join(route)} "
                        f"(Distance: {round(route_distance / 1000, 2)} km, "
                        f"Charge: {round(route_load / 1000, 2)} kg, "
                        f"Temps: {route_time} min)")
                    total_distance += route_distance
                    total_load += route_load
            
            print(f"\nNombre de véhicules/camions utilisés: {vehicles_used}")
            print(f"Distance totale: {round(total_distance / 1000, 2)} km")
            print(f"Charge totale: {round(total_load / 1000, 2)} kg")
            
            # Sauvegarde des routes pour visualisation avec Folium
            print("Sauvegarde des routes pour visualisation avec Folium")
            
            # Créer le dossier static s'il n'existe pas
            os.makedirs('static', exist_ok=True)
            
            # Liste pour stocker les segments de chaque route
            route_lines = []

            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                
                # Vérifier d'abord si ce véhicule a une route réelle
                temp_index = index
                route_nodes = []
                while not routing.IsEnd(temp_index):
                    node_index = manager.IndexToNode(temp_index)
                    route_nodes.append(node_index)
                    temp_index = solution.Value(routing.NextVar(temp_index))
                
                # Si le véhicule n'a qu'un seul noeud (le dépôt), on l'ignore
                if len(route_nodes) <= 1:
                    continue
                    
                # Si on arrive ici, le véhicule a une route réelle, on traite ses segments
                index = routing.Start(vehicle_id)
                previous_index = index
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    next_is_end = routing.IsEnd(index)
                    
                    if not next_is_end:
                        next_node_index = manager.IndexToNode(index)

                        # Coordonnées de la source (point actuel)
                        src_lat = clients_df.loc[node_index, 'lat']
                        src_lng = clients_df.loc[node_index, 'lng']
                        
                        # Coordonnées de la cible (prochain point)
                        dst_lat = clients_df.loc[next_node_index, 'lat']
                        dst_lng = clients_df.loc[next_node_index, 'lng']
                        # IDs source et cible
                        src_id = clients_df.loc[node_index, 'id']
                        dst_id = clients_df.loc[next_node_index, 'id']
                        # Distance (km) et durée (min) pour cet arc
                        arc_distance_km = round(float(data['distance_matrix'][node_index][next_node_index]), 2)
                        arc_duration_min = round(float(data['time_matrix'][node_index][next_node_index]), 2)
                        
                        # Ajouter le segment à la liste
                        route_lines.append({
                            'vehicle_id': vehicle_id + 1,
                            'source_id': src_id,
                            'source_lat': src_lat,
                            'source_lng': src_lng,
                            'target_id': dst_id,
                            'target_lat': dst_lat,
                            'target_lng': dst_lng,
                            'distance_km': arc_distance_km,
                            'duration_min': arc_duration_min,
                        })
                    else:
                        # Dernier segment : retour au dépôt
                        src_lat = clients_df.loc[node_index, 'lat']
                        src_lng = clients_df.loc[node_index, 'lng']
                        depot_lat = clients_df.loc[data['depot'], 'lat']
                        depot_lng = clients_df.loc[data['depot'], 'lng']
                        # IDs pour retour dépôt
                        src_id = clients_df.loc[node_index, 'id']
                        dst_id = clients_df.loc[data['depot'], 'id']
                        # Distance (km) et durée (min) pour retour dépôt
                        arc_distance_km = round(float(data['distance_matrix'][node_index][data['depot']]), 2)
                        arc_duration_min = round(float(data['time_matrix'][node_index][data['depot']]), 2)
                        
                        route_lines.append({
                            'vehicle_id': vehicle_id + 1,
                            'source_id': src_id,
                            'source_lat': src_lat,
                            'source_lng': src_lng,
                            'target_id': dst_id,
                            'target_lat': depot_lat,
                            'target_lng': depot_lng,
                            'distance_km': arc_distance_km,
                            'duration_min': arc_duration_min,
                        })

            # Sauvegarder dans routes.csv pour visualisation
            with open('static/routes.csv', mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['vehicle_id', 'source_id', 'source_lat', 'source_lng', 'target_id', 'target_lat', 'target_lng', 'distance_km', 'duration_min'])
                writer.writeheader()
                writer.writerows(route_lines)

            print(f"Fichier routes.csv généré avec succès pour {len(set(line['vehicle_id'] for line in route_lines))} véhicules utilisés.")
            
            # Générer la carte Folium
            print("Génération de la carte Folium...")
            
            # Charger le fichier routes.csv
            def load_routes():
                try:
                    df = pd.read_csv('static/routes.csv')
                    required_cols = {'vehicle_id', 'source_lat', 'source_lng', 'target_lat', 'target_lng'}
                    if not required_cols.issubset(df.columns):
                        raise ValueError(f"Colonnes manquantes : {required_cols - set(df.columns)}")
                    return df
                except Exception as e:
                    print(f"Erreur chargement fichier: {e}")
                    return None

            routes_df = load_routes()
            if routes_df is not None:
                print("Données chargées depuis routes.csv")

                # Nettoyage
                routes_df_clean = routes_df.dropna(subset=['source_lat', 'source_lng', 'target_lat', 'target_lng'])

                # Fonction OSRM pour tracer les routes réelles
                def get_route_coords(start, end, max_retries=3):
                    for _ in range(max_retries):
                        try:
                            url = f"http://localhost:5001/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=polyline"
                            response = requests.get(url, timeout=10).json()
                            if response.get('code') == 'Ok':
                                return polyline.decode(response['routes'][0]['geometry'])
                        except Exception:
                            continue
                    return None

                # Créer le dossier carte s'il n'existe pas
                os.makedirs('carte', exist_ok=True)

                # Initialisation de la carte Folium
                first_row = routes_df_clean.iloc[0]
                start_lat, start_lng = first_row['source_lat'], first_row['source_lng']

                m = folium.Map(location=[start_lat, start_lng], zoom_start=12, tiles='cartodbpositron')

                # Marqueur du dépôt
                folium.Marker(
                    location=[start_lat, start_lng],
                    popup="🏭 Dépôt",
                    icon=folium.Icon(color='black', icon='home', prefix='fa')
                ).add_to(m)

                # Tracer les routes pour chaque véhicule
                vehicle_ids = routes_df_clean['vehicle_id'].unique()

                icon_colors = [
                    'red', 'blue', 'green', 'orange',
                    'purple', 'darkred', 'darkblue', 'darkgreen',
                    'pink', 'lightblue', 'lightgreen', 'beige',
                    'cadetblue', 'gray', 'black', 'lightgray',
                ]
                colors = itertools.cycle(icon_colors)

                cluster = MarkerCluster(name="Camions").add_to(m)

                for vid in vehicle_ids:
                    try:
                        color = next(colors)
                        fg = folium.FeatureGroup(name=f"Camion {vid}", show=True)

                        vehicle_df = routes_df_clean[routes_df_clean['vehicle_id'] == vid]

                        for _, row in vehicle_df.iterrows():
                            src = (row['source_lat'], row['source_lng'])
                            dst = (row['target_lat'], row['target_lng'])

                            if np.isnan(src[0]) or np.isnan(src[1]) or np.isnan(dst[0]) or np.isnan(dst[1]):
                                continue

                            segment = get_route_coords(src, dst)
                            if segment:
                                folium.PolyLine(segment, color=color, weight=4, opacity=0.8,
                                                popup=f"Camion {vid}").add_to(fg)

                            # Marqueurs source et destination
                            folium.CircleMarker(location=src, radius=4, color=color, fill=True, fill_opacity=0.7).add_to(fg)
                            folium.CircleMarker(location=dst, radius=4, color=color, fill=True, fill_opacity=0.7).add_to(fg)

                        # Ajouter l'icône camion au départ
                        first_point = vehicle_df.iloc[0]
                        folium.Marker(
                            [first_point['source_lat'], first_point['source_lng']],
                            icon=folium.Icon(color=color, icon='truck', prefix='fa'),
                            popup=f"<b>Camion {vid}</b>"
                        ).add_to(cluster)

                        m.add_child(fg)
                        print(f"Camion {vid} tracé")

                    except Exception as e:
                        print(f"Erreur pour Camion {vid} : {e}")
                        continue

                # Affichage de la carte
                folium.LayerControl(collapsed=False).add_to(m)
                m.save('carte/routes_folium.html')
                print("Carte Folium enregistrée sous 'carte/routes_folium.html'")
            
            execution_time = time.time() - start_time
            solved_at = time.time()

            # Écrire un résumé persistant de la solution
            try:
                os.makedirs('static', exist_ok=True)
                solution_summary = {
                    "type": "solution",
                    "success": True,
                    "message": "Solution VRP trouvée",
                    "vehicles_used": vehicles_used,
                    "total_distance": round(total_distance / 1000, 2),
                    "total_load": round(total_load / 1000, 2),
                    "routes_count": len(routes_data),
                    "execution_time": round(execution_time, 2),
                    "generated_at": solved_at,
                }
                with open('static/solution_summary.json', 'w') as f:
                    json.dump(solution_summary, f)

                # Sauvegarder la solution complète pour restauration côté frontend
                full_solution = {
                    "success": True,
                    "message": "Solution VRP trouvée avec succès",
                    "num_vehicles": num_vehicles,
                    "vehicles_used": vehicles_used,
                    "total_distance": round(total_distance / 1000, 2),
                    "total_load": round(total_load / 1000, 2),
                    "routes": routes_data,
                    "execution_time": round(execution_time, 2),
                    "solved_at": solved_at,
                }
                with open('static/last_solution.json', 'w') as f:
                    json.dump(full_solution, f)
            except Exception as _:
                pass
            
            return VRPResponse(
                success=True,
                message="Solution VRP trouvée avec succès",
                vehicles_used=vehicles_used,
                total_distance=round(total_distance / 1000, 2),
                total_load=round(total_load / 1000, 2),
                routes=routes_data,
                execution_time=round(execution_time, 2)
            )
        else:
            print("Aucune solution trouvée.")
            print("Statut du solveur:", routing.status())
            
            execution_time = time.time() - start_time
            
            return VRPResponse(
                success=False,
                message="Aucune solution trouvée",
                vehicles_used=0,
                total_distance=0,
                total_load=0,
                routes=[],
                execution_time=round(execution_time, 2)
            )
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"Erreur lors de la résolution VRP: {e}")
        
        return VRPResponse(
            success=False,
            message=f"Erreur lors de la résolution: {str(e)}",
            vehicles_used=0,
            total_distance=0,
            total_load=0,
            routes=[],
            execution_time=round(execution_time, 2)
        )

@app.get("/carte")
async def get_carte():
    """
    Retourne la carte HTML générée
    """
    try:
        return FileResponse('carte/routes_folium.html', media_type='text/html')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Carte non trouvée. Veuillez d'abord résoudre le VRP.")

@app.get("/routes")
async def get_routes_csv():
    """
    Retourne le fichier CSV des routes
    """
    try:
        return FileResponse('static/routes.csv', media_type='text/csv')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fichier routes.csv non trouvé. Veuillez d'abord résoudre le VRP.")

@app.get("/status")
async def get_status():
    """
    Retourne le statut de l'API et des services
    """
    # Vérifier si OSRM est accessible
    osrm_status = "unknown"
    try:
        response = requests.get("http://localhost:5001/route/v1/driving/-7.558631,33.604427;-7.558631,33.604427", timeout=5)
        if response.status_code == 200:
            osrm_status = "running"
        else:
            osrm_status = "error"
    except:
        osrm_status = "not_available"
    
    return {
        "api_status": "running",
        "osrm_status": osrm_status,
        "message": "VRP API est opérationnelle"
    }

@app.get("/summary")
async def get_summary():
    """
    Retourne un résumé des dernières matrices et solution si disponibles,
    afin que le frontend puisse restaurer l'état après rafraîchissement.
    """
    data = {
        "matrices": {
            "exists": False,
        },
        "solution": {
            "exists": False,
        },
        "files": {
            "routes_csv": os.path.exists('static/routes.csv'),
            "carte_html": os.path.exists('carte/routes_folium.html'),
        }
    }

    try:
        if os.path.exists('matrices/summary.json'):
            with open('matrices/summary.json', 'r') as f:
                m = json.load(f)
                m['generated_at'] = m.get('generated_at')
                data['matrices'] = { **m, "exists": True }
        else:
            # fallback basique sur la présence des CSV
            if os.path.exists('matrices/distance_matrix.csv') and os.path.exists('matrices/time_matrix.csv'):
                data['matrices'] = { "exists": True }
    except Exception:
        pass

    try:
        if os.path.exists('static/solution_summary.json'):
            with open('static/solution_summary.json', 'r') as f:
                s = json.load(f)
                s['generated_at'] = s.get('generated_at')
                data['solution'] = { **s, "exists": True }
        else:
            if os.path.exists('static/routes.csv'):
                data['solution'] = { "exists": True }
    except Exception:
        pass

    return data

@app.get("/last-solution")
async def get_last_solution():
    """Retourne la dernière solution complète si disponible"""
    path = 'static/last_solution.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                raise HTTPException(status_code=500, detail="Solution sauvegardée corrompue")
    raise HTTPException(status_code=404, detail="Aucune solution sauvegardée")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)