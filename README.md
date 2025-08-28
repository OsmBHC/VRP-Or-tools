# VRP Or-Tools — Frontend (Vite/React) + Backend (FastAPI)

Application complète pour générer des matrices de distances/temps via OSRM et résoudre un problème de tournées de véhicules (VRP) avec Google OR-Tools. Frontend moderne avec Vite/React + Tailwind (shadcn/ui).


## Sommaire
- [Architecture du projet](#architecture-du-projet)
- [Prérequis](#prérequis)
- [Installation rapide](#installation-rapide)
- [Lancer le Backend (FastAPI)](#lancer-le-backend-fastapi)
- [Lancer le Frontend (Vite/React)](#lancer-le-frontend-vitereact)
- [Commandes utiles](#commandes-utiles)
- [API Backend](#api-backend)
- [Flux d’utilisation recommandé](#flux-dutilisation-recommandé)


## Architecture du projet
```
Projet/
├─ Backend/
│  ├─ code/
│  │  └─ main.py            # Application FastAPI (endpoints VRP)
│  └─ requirements.txt      # Dépendances Python
├─ Frontend/
│  ├─ src/                  # Code React + UI
│  │  └─ components/       # Composants React
│  │  └─ pages/            # Pages React
│  │  └─ hooks/            # Hooks React
│  │  └─ lib/              # Api Config
│  │  └─ App.tsx           # Application React
│  │  └─ index.css         # Styles globaux
│  ├─ vite.config.ts        # Configuration Vite
│  ├─ package.json          # Scripts et dépendances
└─ README.md
```


## Prérequis
- Node.js (recommandé: LTS) et un gestionnaire de paquets (npm, pnpm, yarn ou bun)
- Python 3 (avec venv)
- OSRM backend accessible en local sur `http://localhost:5001` (profil driving) pour la génération des matrices

Exemple de démarrage rapide d’OSRM avec Docker (exige un extrait de carte `.osm.pbf`):
```bash
# 1) Télécharger un extrait de carte (ex: maroc-latest.osm.pbf) depuis Geofabrik
# 2) Préparer et lancer OSRM (profil car/driving)
docker run -t -i -v "$PWD/osrm_data:/data" osrm/osrm-backend:latest osrm-extract -p /opt/car.lua /data/morocco-latest.osm.pbf

# 3) Partitionner le graphe
docker run -t -i -v "$PWD/osrm_data:/data" osrm/osrm-backend:latest osrm-partition /data/morocco-latest.osrm

# 4) Optimiser le graphe (MLD)
docker run -t -i -v "$PWD/osrm_data:/data" osrm/osrm-backend:latest osrm-customize /data/morocco-latest.osrm

# Lancer le serveur OSRM sur le port 5001
docker run -t -i -p 5001:5000 -v "$PWD/osrm_data:/data" osrm/osrm-backend:latest osrm-routed --algorithm mld /data/morocco-latest.osrm
```


## Installation rapide
```bash
# Cloner le repo (si ce n'est pas déjà fait)
# git clone <url>
# cd Projet

# 1) Backend: créer un venv et installer les deps
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r Backend/requirements.txt

# 2) Frontend: installer les deps
cd Frontend
npm install                 # ou: pnpm i | yarn | bun install
```


## Lancer le Backend (FastAPI)
Depuis `Backend/code/` (avec le venv activé):
```bash
python main.py
# ou
python3 main.py
```
- L’API sera disponible par défaut sur: `http://localhost:8000`
- Le backend requiert OSRM sur `http://localhost:5001` pour les endpoints de matrices.


## Lancer le Frontend (Vite/React)
Depuis `Frontend/`:
```bash
npm run dev   # ou: pnpm dev | yarn dev | bun dev
```
- Par défaut, Vite lance l’app sur `http://localhost:8080`.
- Assurez-vous que l’API FastAPI tourne (port 8000) et qu’OSRM est accessible (port 5001).


## Commandes utiles
- Frontend (`Frontend/package.json`):
  - `npm run dev` — lancer le serveur de dev Vite
  - `npm run build` — build production
  - `npm run build:dev` — build en mode développement
  - `npm run preview` — prévisualiser le build
  - `npm run lint` — lint du code

- Backend:
  - Installer les deps: `pip install -r Backend/requirements.txt`
  - Lancer l’API: `python main.py`


## API Backend
Fichier principal: `Backend/code/main.py`

- `GET /`
  - Ping simple: `{ "message": "..." }`

- `POST /upload-dataset` (multipart/form-data)
  - Paramètre: `file` (xlsx ou csv)
  - Stocke le fichier sous `uploads/latest.xlsx` ou `uploads/latest.csv`
  - Vérifie la présence des colonnes: `PARTNER_CODE`, `LATITUDE`, `LONGITUDE`, `WEIGHT`

- `POST /generate-matrices` (JSON)
  - Corps (par défaut):
    ```json
    {
      "depot_lat": 33.604427,
      "depot_lng": -7.558631
    }
    ```
  - Produit: `matrices/distance_matrix.csv`, `matrices/time_matrix.csv`, `matrices/clients_processed.csv` et `matrices/summary.json`
  - S’appuie sur OSRM (`http://localhost:5001/table/v1/driving/...`)

- `POST /solve-vrp` (JSON)
  - Corps (exemple par défaut):
    ```json
    {
      "num_vehicles": 36,
      "vehicle_capacity": 4000.0,
      "service_time": 5,
      "time_limit": 300,
      "vehicle_time_limit": 480,
      "interval_seconds": 60
    }
    ```
  - Utilise OR-Tools pour optimiser les tournées à partir des matrices CSV générées


## Flux d’utilisation recommandé
1. Lancer OSRM (port 5001)
2. Lancer le Backend (port 8000)
3. Lancer le Frontend (port 8080)
4. Depuis l’UI:
   - Importer un dataset via l’interface (ou appeler `POST /upload-dataset`)
   - Générer les matrices (`POST /generate-matrices`)
   - Lancer la résolution (`POST /solve-vrp`)
5. Visualiser les résultats dans le composant `ResultsViewer`
