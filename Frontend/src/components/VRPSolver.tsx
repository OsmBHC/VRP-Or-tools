import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Settings, Play, Truck, Clock, Zap, Package } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { solveVRP as apiSolveVRP } from '@/lib/api';

interface VRPSolverProps {
  onSolveStart: () => void;
  onSolveComplete: (results: any) => void;
  isSolving: boolean;
}

export const VRPSolver: React.FC<VRPSolverProps> = ({
  onSolveStart,
  onSolveComplete,
  isSolving
}) => {
  const [numVehicles, setNumVehicles] = useState([3]);
  const [vehicleCapacity, setVehicleCapacity] = useState('1000');
  const [serviceTime, setServiceTime] = useState('15');
  const [timeLimit, setTimeLimit] = useState('60');
  const [intervalSeconds, setIntervalSeconds] = useState('60');
  const [vehicleTimeLimit, setVehicleTimeLimit] = useState('480');
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  const solveVRP = async () => {
    onSolveStart();
    setProgress(10);

    try {
      const response = await apiSolveVRP({
        num_vehicles: numVehicles[0],
        vehicle_capacity: parseInt(vehicleCapacity, 10),
        service_time: parseInt(serviceTime, 10),
        time_limit: parseInt(timeLimit, 10),
        vehicle_time_limit: parseInt(vehicleTimeLimit, 10),
        interval_seconds: parseInt(intervalSeconds, 10),
      });

      setProgress(100);
      // Propager la capacité choisie par l'utilisateur pour l'affichage dans ResultsViewer
      const augmented = { ...response, vehicle_capacity: parseInt(vehicleCapacity, 10) };
      onSolveComplete(augmented);

      toast({
        title: augmented.success ? 'Problème VRP résolu avec succès' : 'Échec de la résolution du VRP',
        description: augmented.success
          ? `Véhicules utilisés: ${augmented.vehicles_used} | Distance totale: ${augmented.total_distance} km`
          : augmented.message,
        variant: augmented.success ? 'default' : 'destructive',
      });

    } catch (error: any) {
      toast({
        title: "Erreur",
        description: error?.message || "Échec de la résolution du VRP",
        variant: "destructive"
      });
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2 mb-2">
          <Settings className="h-6 w-6 text-accent" />
          Résolution du problème VRP
        </h2>
        <p className="text-muted-foreground">
          Configurez les paramètres d'optimisation pour résoudre votre problème de tournées.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Configuration Parameters */}
        <Card className="p-6 bg-gradient-card shadow-elegant">
          <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <Truck className="h-5 w-5 text-primary" />
            Paramètres des véhicules
          </h3>
          
          <div className="space-y-6">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <Label>Nombre de véhicules</Label>
                <Badge variant="outline">{numVehicles[0]}</Badge>
              </div>
              <Slider
                value={numVehicles}
                onValueChange={setNumVehicles}
                max={200}
                min={1}
                step={1}
                disabled={isSolving}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="capacity">Capacité par véhicule (kg)</Label>
              <Input
                id="capacity"
                type="number"
                value={vehicleCapacity}
                onChange={(e) => setVehicleCapacity(e.target.value)}
                disabled={isSolving}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="service-time">Temps de service par client (min)</Label>
              <Input
                id="service-time"
                type="number"
                value={serviceTime}
                onChange={(e) => setServiceTime(e.target.value)}
                disabled={isSolving}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="vehicle-time-limit">Limite de temps par véhicule (min)</Label>
              <Input
                id="vehicle-time-limit"
                type="number"
                value={vehicleTimeLimit}
                onChange={(e) => setVehicleTimeLimit(e.target.value)}
                disabled={isSolving}
                className="w-full"
              />
            </div>
          </div>
        </Card>

        {/* Current Configuration Summary */}
        <Card className="p-4 bg-gradient-card shadow-elegant h-full">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Package className="h-5 w-5 text-success" />
            Configuration actuelle
          </h3>
          <div className="grid grid-cols-2 gap-3 text-base h-full">
            <div className="text-center p-3 bg-background/50 rounded-lg flex flex-col justify-center">
              <div className="font-semibold text-2xl text-primary">{numVehicles[0]}</div>
              <div className="text-muted-foreground text-base">Véhicules</div>
            </div>
            <div className="text-center p-3 bg-background/50 rounded-lg flex flex-col justify-center">
              <div className="font-semibold text-2xl text-accent">{vehicleCapacity} kg</div>
              <div className="text-muted-foreground text-base">Capacité</div>
            </div>
            <div className="text-center p-3 bg-background/50 rounded-lg flex flex-col justify-center">
              <div className="font-semibold text-2xl text-success">{serviceTime} min</div>
              <div className="text-muted-foreground text-base">Service</div>
            </div>
            <div className="text-center p-3 bg-background/50 rounded-lg flex flex-col justify-center">
              <div className="font-semibold text-2xl text-info">{vehicleTimeLimit} min</div>
              <div className="text-muted-foreground text-base">Temps véhicule</div>
            </div>
          </div>
        </Card>
      </div>

      {/* Optimization and Solve Section */}
      <Card className="p-6 bg-gradient-card shadow-elegant">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap className="h-5 w-5 text-accent" />
          Paramètres d'optimisation
        </h3>
        
        <div className="space-y-4">
          <div className="flex gap-4">
            <div className="space-y-2 flex-1">
              <Label htmlFor="time-limit">Limite de temps d'optimisation (s)</Label>
              <Input
                id="time-limit"
                type="number"
                value={timeLimit}
                onChange={(e) => setTimeLimit(e.target.value)}
                disabled={isSolving}
                className="w-full"
              />
            </div>
            <div className="space-y-2 flex-1">
              <Label htmlFor="interval-seconds">Intervalle d'affichage (s)</Label>
              <Input
                id="interval-seconds"
                type="number"
                value={intervalSeconds}
                onChange={(e) => setIntervalSeconds(e.target.value)}
                disabled={isSolving}
                className="w-full"
              />
            </div>
          </div>

          <Separator />

          <Button 
            onClick={solveVRP}
            disabled={isSolving}
            className="w-1/4 py-6 bg-gradient-accent hover:opacity-90 shadow-accent"
            size="lg"
          >
            {isSolving ? (
              <>
                <Clock className="mr-2 h-4 w-4 animate-spin" />
                Résolution en cours...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Résoudre le VRP
              </>
            )}
          </Button>

          {isSolving && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progression de la résolution</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};