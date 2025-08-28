import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Truck, MapPin, Download, Clock, Zap, Package, BarChart3, Map } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ResultsViewerProps {
  results: any;
}

export const ResultsViewer: React.FC<ResultsViewerProps> = ({ results }) => {
  const { toast } = useToast();
  if (!results) {
    return (
      <div className="text-center py-16">
        <div className="mb-4">
          <BarChart3 className="h-16 w-16 mx-auto text-muted-foreground opacity-50" />
        </div>
        <h3 className="text-xl font-semibold mb-2">Aucun résultat disponible</h3>
        <p className="text-muted-foreground mb-6">
          Générez d'abord les matrices puis résolvez le problème VRP pour voir les résultats ici.
        </p>
        <div className="flex gap-3 justify-center">
          <Badge variant="outline">Étape 1: Générer les matrices</Badge>
          <Badge variant="outline">Étape 2: Résoudre le VRP</Badge>
        </div>
      </div>
    );
  }

  // Harmonious unique palette (reused consistently)
  const palette = [
    '#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6',
    '#eab308', '#ec4899', '#22c55e', '#06b6d4', '#84cc16', '#f97316',
  ];
  const getVehicleColor = (vehicleId: number) => palette[(vehicleId - 1) % palette.length];

  const routes = results.routes ?? [];
  const totalTime = typeof results.total_time !== 'undefined'
    ? results.total_time
    : routes.reduce((sum: number, r: any) => sum + (Number(r.time_min) || 0), 0);
  const vehicleCapacityKg = Number(results?.vehicle_capacity) || 4000; // fallback to default
  const maxTime = Math.max(1, ...routes.map((r: any) => Number(r.time_min) || 0));
  const computeTime = typeof results.execution_time !== 'undefined' ? results.execution_time : (results.computation_time ?? 0);
  const vehiclesUsed = (typeof results?.vehicles_used !== 'undefined')
    ? Number(results.vehicles_used)
    : (Array.isArray(routes) ? routes.filter((r: any) => (r?.route?.length || 0) > 1).length : Number(results?.num_vehicles) || 0);

  const renderRoutePreview = (stops: string[]) => {
    const full = stops.join(' → ');
    const maxStops = 6;
    const shown = stops.slice(0, maxStops);
    const hidden = Math.max(0, stops.length - shown.length);
    return (
      <div className="flex items-center gap-1 max-w-[520px] overflow-hidden" title={full}>
        <span className="text-xs truncate">
          {shown.join(' → ')}{hidden > 0 ? ` … (+${hidden})` : ''}
        </span>
      </div>
    );
  };

  const handleDownloadCSV = async () => {
    try {
      const res = await fetch('/api/routes', { method: 'GET', cache: 'no-store' });
      if (!res.ok) throw new Error('Fichier indisponible');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'routes.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      toast({
        title: 'Téléchargement impossible',
        description: e?.message || 'Aucun CSV trouvé. Lancez une résolution VRP pour générer routes.csv',
        variant: 'destructive',
      });
    }
  };

  const handleViewMap = async () => {
    try {
      const res = await fetch('/api/carte', { method: 'GET', cache: 'no-store' });
      if (!res.ok) throw new Error('Carte indisponible');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
      // Optionnellement révoquer après un certain délai
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e: any) {
      toast({
        title: 'Ouverture de la carte impossible',
        description: e?.message || 'Aucune carte disponible. Lancez une résolution VRP pour générer la carte',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2 mb-2">
            <BarChart3 className="h-6 w-6 text-success" />
            Résultats de l'optimisation
          </h2>
          <p className="text-muted-foreground">
            Solution optimale trouvée avec {vehiclesUsed} véhicules utilisés
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="flex items-center gap-2" onClick={handleDownloadCSV}>
            <Download className="h-4 w-4" />
            Télécharger les routes (CSV)
          </Button>
          <Button variant="accent" className="flex items-center gap-2" onClick={handleViewMap}>
            <Map className="h-4 w-4" />
            Voir la carte
          </Button>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="p-6 bg-gradient-card shadow-elegant">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Distance totale</p>
              <p className="text-2xl font-bold text-primary">{results.total_distance} km</p>
            </div>
            <MapPin className="h-8 w-8 text-primary opacity-20" />
          </div>
        </Card>

        <Card className="p-6 bg-gradient-card shadow-elegant">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Temps total</p>
              <p className="text-2xl font-bold text-accent">{totalTime} min</p>
            </div>
            <Clock className="h-8 w-8 text-accent opacity-20" />
          </div>
        </Card>

        <Card className="p-6 bg-gradient-card shadow-elegant">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Véhicules utilisés</p>
              <p className="text-2xl font-bold text-success">{vehiclesUsed}</p>
            </div>
            <Truck className="h-8 w-8 text-success opacity-20" />
          </div>
        </Card>

        <Card className="p-6 bg-gradient-card shadow-elegant">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Temps de calcul</p>
              <p className="text-2xl font-bold text-warning">{computeTime}s</p>
            </div>
            <Zap className="h-8 w-8 text-warning opacity-20" />
          </div>
        </Card>
      </div>

      {/* Routes Table */}
      <Card className="shadow-elegant">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Truck className="h-5 w-5 text-primary" />
            Détail des routes
          </h3>
        </div>
        
        <div className="p-6">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Véhicule</TableHead>
                <TableHead>Route</TableHead>
                <TableHead className="text-right">Distance</TableHead>
                <TableHead className="text-right">Charge</TableHead>
                <TableHead className="text-right">Temps</TableHead>
                <TableHead className="text-right">Utilisation</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {results.routes?.map((route: any, index: number) => (
                <TableRow key={index}>
                  <TableCell>
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getVehicleColor(route.vehicle_id) }}></div>
                      <span className="font-medium">Véhicule {route.vehicle_id}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    {renderRoutePreview(route.route || [])}
                  </TableCell>
                  <TableCell className="text-right font-medium">
                    {Number(route.distance_km ?? 0)} km
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center gap-2 justify-end">
                      <Package className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">{Number(route.load_kg ?? 0)} kg</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-medium">
                    {Number(route.time_min ?? 0)} min
                  </TableCell>
                  <TableCell className="text-right">
                    <Badge 
                      variant={(Number(route.load_kg ?? 0) / vehicleCapacityKg) > 0.8 ? "warning" : (Number(route.load_kg ?? 0) / vehicleCapacityKg) > 0.6 ? "default" : "success"}
                      className="text-xs"
                    >
                      {Math.round(((Number(route.load_kg ?? 0)) / vehicleCapacityKg) * 100)}%
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="p-6 bg-gradient-card shadow-elegant">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            Répartition des charges
          </h3>
          <div className="space-y-3">
            {results.routes?.map((route: any, index: number) => (
              <div key={index} className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getVehicleColor(route.vehicle_id) }}></div>
                <span className="text-sm flex-1">Véhicule {route.vehicle_id}</span>
                <div className="flex-1 bg-muted rounded-full h-2">
                  <div 
                    className="h-2 rounded-full"
                    style={{ width: `${Math.min(100, ((Number(route.load_kg ?? 0)) / vehicleCapacityKg) * 100)}%`, backgroundColor: getVehicleColor(route.vehicle_id) }}
                  ></div>
                </div>
                <span className="text-sm font-medium text-right w-16">{Number(route.load_kg ?? 0)}kg</span>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-6 bg-gradient-card shadow-elegant">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Clock className="h-5 w-5 text-accent" />
            Temps par véhicule
          </h3>
          <div className="space-y-3">
            {results.routes?.map((route: any, index: number) => (
              <div key={index} className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getVehicleColor(route.vehicle_id) }}></div>
                <span className="text-sm flex-1">Véhicule {route.vehicle_id}</span>
                <div className="flex-1 bg-muted rounded-full h-2">
                  <div 
                    className="h-2 rounded-full"
                    style={{ width: `${Math.min(100, ((Number(route.time_min ?? 0)) / maxTime) * 100)}%`, backgroundColor: getVehicleColor(route.vehicle_id) }}
                  ></div>
                </div>
                <span className="text-sm font-medium text-right w-16">{Number(route.time_min ?? 0)}min</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Separator />

      <div className="text-center text-sm text-muted-foreground">
        Solution générée le {new Date(results.solved_at).toLocaleString('fr-FR')}
      </div>
    </div>
  );
};