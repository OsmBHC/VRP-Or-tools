import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MapPin, Truck, Settings, Play, Download, Activity, Clock } from 'lucide-react';
import { MatrixGenerator } from './MatrixGenerator';
import { VRPSolver } from './VRPSolver';
import { ResultsViewer } from './ResultsViewer';
import { getStatus, getSummary, getLastSolution } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface APIStatus {
  status: 'online' | 'offline' | 'loading';
  message: string;
}

export const VRPDashboard = () => {
  const [apiStatus, setApiStatus] = useState<APIStatus>({
    status: 'loading',
    message: 'Vérification du statut...'
  });
  const [lastResults, setLastResults] = useState<any>(null);
  const [matricesInfo, setMatricesInfo] = useState<{ num_points?: number; generated_at?: number; exists?: boolean } | null>(null);
  const [isGeneratingMatrix, setIsGeneratingMatrix] = useState(false);
  const [isSolvingVRP, setIsSolvingVRP] = useState(false);
  const { toast } = useToast();

  // Vérification du statut de l'API
  const checkAPIStatus = async () => {
    try {
      setApiStatus({ status: 'loading', message: 'Vérification...' });
      const res = await getStatus();
      const osrm = res?.osrm_status || 'unknown';
      setApiStatus({
        status: 'online',
        message: `API opérationnelle • OSRM: ${osrm}`,
      });
    } catch (error) {
      setApiStatus({ 
        status: 'offline', 
        message: 'Erreur de connexion à l\'API' 
      });
    }
  };

  const formatDateTime = (ts?: number) => {
    if (!ts) return '-';
    try { return new Date(ts * 1000).toLocaleString(); } catch { return '-'; }
  };

  const loadSummary = async () => {
    try {
      const summary = await getSummary();
      if (summary?.solution?.exists) {
        // Charger la solution complète (avec routes) pour peupler la page Résultats
        try {
          const full = await getLastSolution();
          setLastResults(full);
        } catch {
          // fallback aux champs du résumé si la solution complète est indisponible
          const s = summary.solution;
          setLastResults({
            num_vehicles: s.vehicles_used,
            total_distance: s.total_distance,
            computation_time: s.execution_time,
            generated_at: s.generated_at,
            routes: [],
          });
        }
      }
      if (summary?.matrices?.exists) {
        const m = summary.matrices;
        setMatricesInfo({ num_points: m.num_points, generated_at: m.generated_at, exists: true });
      }
    } catch (e) {
      // silencieux
    }
  };

  React.useEffect(() => {
    checkAPIStatus();
    loadSummary();
  }, []);

  const getStatusBadge = () => {
    switch (apiStatus.status) {
      case 'online':
        return <Badge variant="success" className="bg-success text-success-foreground">En ligne</Badge>;
      case 'offline':
        return <Badge variant="destructive">Hors ligne</Badge>;
      case 'loading':
        return <Badge variant="secondary">Vérification...</Badge>;
    }
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
        description: e?.message || 'Générez d\'abord une solution VRP pour créer routes.csv',
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
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e: any) {
      toast({
        title: 'Ouverture de la carte impossible',
        description: e?.message || 'Résolvez d\'abord le VRP pour générer la carte',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-hero rounded-xl p-8 text-white shadow-primary">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <img src="logo.png" alt="" width={100} height={100}/>
              <h1 className="text-4xl font-bold">Livraison Express.</h1>
            </div>
            <p className="text-xl opacity-90 mt-2">Résolution de problèmes de tournées de véhicules</p>
          </div>
          <div className="text-right space-y-2">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              <span>Statut API:</span>
              {getStatusBadge()}
            </div>
            <Button 
              variant="secondary" 
              size="sm" 
              onClick={checkAPIStatus}
              className="bg-white/20 hover:bg-white/30 border-white/20"
            >
              Actualiser
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-12 gap-6">
        {/* Side Panel - Quick Stats */}
        <div className="col-span-3 space-y-4">
          <Card className="p-6 bg-gradient-card shadow-elegant">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <MapPin className="h-5 w-5 text-primary" />
              Matrices
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Points traités:</span>
                <Badge variant="outline">{matricesInfo?.num_points ?? 0}</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Dernière génération:</span>
                <span className="text-sm text-muted-foreground">{formatDateTime(matricesInfo?.generated_at)}</span>
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-gradient-card shadow-elegant">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <Truck className="h-5 w-5 text-accent" />
              Dernière solution
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Véhicules:</span>
                <Badge variant="outline">{
                  (typeof lastResults?.vehicles_used !== 'undefined')
                    ? Number(lastResults.vehicles_used)
                    : (Array.isArray(lastResults?.routes)
                        ? lastResults.routes.filter((r: any) => (r?.route?.length || 0) > 1).length
                        : Number(lastResults?.num_vehicles) || 0)
                }</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Distance totale:</span>
                <span className="text-sm font-medium">{lastResults?.total_distance || 0} km</span>
              </div>
              {(() => {
                const routes = lastResults?.routes ?? [];
                const totalTime = (typeof lastResults?.total_time !== 'undefined')
                  ? lastResults.total_time
                  : routes.reduce((sum: number, r: any) => sum + (Number(r?.time_min) || 0), 0);
                return (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Temps total:</span>
                    <span className="text-sm font-medium">{totalTime} min</span>
                  </div>
                );
              })()}
              <div className="flex justify-between">
                <span className="text-muted-foreground">Temps calcul:</span>
                <span className="text-sm text-muted-foreground">{(lastResults?.execution_time ?? lastResults?.computation_time ?? 0)}s</span>
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-gradient-card shadow-elegant">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <Download className="h-5 w-5 text-success" />
              Exports
            </h3>
            <div className="space-y-2">
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full justify-start"
                disabled={!lastResults}
                onClick={handleDownloadCSV}
              >
                Télécharger les routes (CSV)
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full justify-start"
                disabled={!lastResults}
                onClick={handleViewMap}
              >
                Voir la carte
              </Button>
            </div>
          </Card>
        </div>

        {/* Main Content Area */}
        <div className="col-span-9">
          <Card className="min-h-[600px] shadow-elegant">
            <Tabs defaultValue="matrices" className="h-full">
              <div className="border-b p-6 pb-4">
                <TabsList className="grid grid-cols-3 w-full max-w-md">
                  <TabsTrigger value="matrices" className="flex items-center gap-2">
                    <MapPin className="h-4 w-4" />
                    Matrices
                  </TabsTrigger>
                  <TabsTrigger value="solver" className="flex items-center gap-2">
                    <Settings className="h-4 w-4" />
                    Résolution
                  </TabsTrigger>
                  <TabsTrigger value="results" className="flex items-center gap-2">
                    <Truck className="h-4 w-4" />
                    Résultats
                  </TabsTrigger>
                </TabsList>
              </div>

              <div className="p-6">
                <TabsContent value="matrices" className="mt-0">
                  <MatrixGenerator 
                    onGenerationStart={() => setIsGeneratingMatrix(true)}
                    onGenerationComplete={() => setIsGeneratingMatrix(false)}
                    onMatricesGenerated={({ num_points, generated_at }) => {
                      setMatricesInfo({ num_points, generated_at, exists: true });
                    }}
                    isGenerating={isGeneratingMatrix}
                  />
                </TabsContent>

                <TabsContent value="solver" className="mt-0">
                  <VRPSolver 
                    onSolveStart={() => setIsSolvingVRP(true)}
                    onSolveComplete={(results) => {
                      setIsSolvingVRP(false);
                      setLastResults(results);
                    }}
                    isSolving={isSolvingVRP}
                  />
                </TabsContent>

                <TabsContent value="results" className="mt-0">
                  <ResultsViewer results={lastResults} />
                </TabsContent>
              </div>
            </Tabs>
          </Card>
        </div>
      </div>
    </div>
  );
};