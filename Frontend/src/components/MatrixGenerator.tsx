import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { MapPin, Play, CheckCircle2, Clock, Hash, Upload, FileText, Users } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { generateMatrices as apiGenerateMatrices, MatrixResponse, uploadDataset as apiUploadDataset } from '@/lib/api';

interface MatrixGeneratorProps {
  onGenerationStart: () => void;
  onGenerationComplete: () => void;
  // New: notify parent so sidebar updates instantly
  onMatricesGenerated?: (info: { num_points: number; generated_at: number }) => void;
  isGenerating: boolean;
}

export const MatrixGenerator: React.FC<MatrixGeneratorProps> = ({
  onGenerationStart,
  onGenerationComplete,
  onMatricesGenerated,
  isGenerating
}) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [depotLat, setDepotLat] = useState('');
  const [depotLng, setDepotLng] = useState('');
  const [progress, setProgress] = useState(0);
  const [lastGeneration, setLastGeneration] = useState<any>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const { toast } = useToast();

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const validateAndSetFile = (file: File) => {
    if (file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || 
        file.name.endsWith('.xlsx') || 
        file.type === 'text/csv' || 
        file.name.endsWith('.csv')) {
      setUploadedFile(file);
      toast({
        title: "Fichier chargé",
        description: `Dataset "${file.name}" chargé avec succès`,
        variant: "default"
      });
    } else {
      toast({
        title: "Format non supporté",
        description: "Veuillez choisir un fichier Excel (.xlsx) ou CSV (.csv)",
        variant: "destructive"
      });
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const generateMatrices = async () => {
    if (!uploadedFile) {
      toast({
        title: "Erreur",
        description: "Veuillez d'abord charger votre dataset clients",
        variant: "destructive"
      });
      return;
    }

    if (!depotLat || !depotLng) {
      toast({
        title: "Erreur",
        description: "Veuillez saisir les coordonnées du dépôt",
        variant: "destructive"
      });
      return;
    }

    onGenerationStart();
    setProgress(10);

    try {
      const depot_lat = parseFloat(depotLat);
      const depot_lng = parseFloat(depotLng);
      // 1) Upload dataset avant la génération
      setProgress(20);
      const up = await apiUploadDataset(uploadedFile);
      if (up?.success) {
        toast({ title: 'Dataset chargé', description: `${uploadedFile.name} (${up.rows ?? '?'} lignes)`, variant: 'default' });
      }
      setProgress(35);
      const res: MatrixResponse = await apiGenerateMatrices(depot_lat, depot_lng);
      setProgress(100);

      const results = {
        status: res.success ? 'success' : 'error',
        execution_time: res.execution_time,
        num_points: res.num_points,
        generated_at: new Date().toISOString()
      };

      setLastGeneration(results);
      if (res.success) {
        // notify parent for immediate sidebar refresh
        onMatricesGenerated?.({
          num_points: res.num_points,
          generated_at: Math.floor(Date.now() / 1000),
        });
      }
      if (res.success && res.up_to_date) {
        toast({
          title: "Matrices déjà à jour",
          description: `${res.num_points} points — aucune mise à jour nécessaire (en ${res.execution_time}s)`,
          variant: "default",
        });
      } else {
        toast({
          title: res.success ? "Matrices générées avec succès" : "Échec de la génération",
          description: res.success
            ? `${res.num_points} points traités en ${res.execution_time}s`
            : res.message,
          variant: res.success ? "default" : "destructive",
        });
      }

    } catch (error: any) {
      toast({
        title: "Erreur",
        description: error?.message || "Échec de la génération des matrices",
        variant: "destructive"
      });
    } finally {
      onGenerationComplete();
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2 mb-2">
          <MapPin className="h-6 w-6 text-primary" />
          Génération des matrices
        </h2>
        <p className="text-muted-foreground">
          Générez les matrices de distances et temps avec OSRM pour optimiser vos tournées.
        </p>
      </div>

      {/* Étape 1: Upload du dataset clients */}
      <Card className="p-6 bg-gradient-card shadow-elegant">
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-primary-foreground text-sm font-medium">
              1
            </div>
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Users className="h-5 w-5" />
              Dataset clients
            </h3>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div 
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  isDragOver 
                    ? 'border-primary bg-primary/5' 
                    : 'border-border hover:border-primary/50'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  id="dataset-upload"
                  accept=".xlsx,.csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={isGenerating}
                />
                <div className="flex flex-col items-center space-y-3">
                  <div className="p-3 rounded-full bg-primary/10">
                    <Upload className="h-8 w-8 text-primary" />
                  </div>
                  <div>
                    <p className="text-lg font-medium">Chargez votre dataset clients</p>
                    <p className="text-sm text-muted-foreground">
                      {isDragOver ? 'Déposez votre fichier ici' : 'Glissez-déposez ou cliquez pour choisir un fichier'}
                    </p>
                  </div>
                  <Button 
                    variant="outline" 
                    className="mt-2" 
                    disabled={isGenerating}
                    onClick={() => document.getElementById('dataset-upload')?.click()}
                  >
                    Choisir un fichier
                  </Button>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {uploadedFile ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 p-4 bg-success/10 border border-success/20 rounded-lg">
                    <FileText className="h-5 w-5 text-success" />
                    <div className="flex-1">
                      <p className="font-medium">{uploadedFile.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(uploadedFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <Badge variant="success">Chargé</Badge>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    <p className="font-medium mb-2">Format attendu du fichier:</p>
                    <ul className="space-y-1 text-xs">
                      <li>• Colonnes: PARTNER_CODE, LATITUDE, LONGITUDE, WEIGHT</li>
                      <li>• Format: Excel (.xlsx) ou CSV (.csv)</li>
                      <li>• Première ligne: en-têtes</li>
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-muted/50 border border-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-2">
                    <strong>Format attendu du fichier:</strong>
                  </p>
                  <div className="text-xs font-mono bg-background p-2 rounded border">
                    PARTNER_CODE,LATITUDE,LONGITUDE,WEIGHT<br/>
                    C36713,33.586656,-7.581679,18.82<br/>
                    C33753,33.591718,-7.556250,3.16<br/>
                    ...
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>

      {/* Étape 2: Configuration du dépôt */}
      <Card className="p-6 bg-gradient-card shadow-elegant">
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <div className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
              uploadedFile ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            }`}>
              2
            </div>
            <h3 className={`text-lg font-semibold flex items-center gap-2 ${
              !uploadedFile ? 'text-muted-foreground' : ''
            }`}>
              <MapPin className="h-5 w-5" />
              Configuration du dépôt
            </h3>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="depot-lat">Latitude</Label>
                <Input
                  id="depot-lat"
                  type="number"
                  step="0.000001"
                  placeholder="46.227638"
                  value={depotLat}
                  onChange={(e) => setDepotLat(e.target.value)}
                  disabled={isGenerating || !uploadedFile}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="depot-lng">Longitude</Label>
                <Input
                  id="depot-lng"
                  type="number"
                  step="0.000001"
                  placeholder="2.213749"
                  value={depotLng}
                  onChange={(e) => setDepotLng(e.target.value)}
                  disabled={isGenerating || !uploadedFile}
                />
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              <p className="font-medium mb-2">Point de départ et d'arrivée:</p>
              <ul className="space-y-1 text-xs">
                <li>• Coordonnées GPS du dépôt principal</li>
                <li>• Tous les véhicules partent et reviennent ici</li>
                <li>• Format décimal (ex: 46.227638)</li>
              </ul>
            </div>
          </div>
        </div>
      </Card>

      {/* Étape 3: Génération */}
      <Card className="p-6 bg-gradient-card shadow-elegant">
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <div className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
              uploadedFile && depotLat && depotLng ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            }`}>
              3
            </div>
            <h3 className={`text-lg font-semibold ${
              !uploadedFile || !depotLat || !depotLng ? 'text-muted-foreground' : ''
            }`}>
              Génération des matrices
            </h3>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4 my-auto">
              <Button 
                onClick={generateMatrices}
                disabled={isGenerating || !uploadedFile || !depotLat || !depotLng}
                className="w-full bg-gradient-primary hover:opacity-90 shadow-primary"
                size="lg"
              >
                {isGenerating ? (
                  <>
                    <Clock className="mr-2 h-4 w-4 animate-spin" />
                    Génération en cours...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4 center" />
                    Générer les matrices
                  </>
                )}
              </Button>

              {isGenerating && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progression</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}
            </div>

            <div className="space-y-4">
              {lastGeneration ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-success" />
                    <span className="font-medium">Matrices générées</span>
                    <Badge variant="success" className="ml-auto">
                      Succès
                    </Badge>
                  </div>
                  
                  <Separator />
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Points:</span>
                      <div className="flex items-center gap-1">
                        <Hash className="h-3 w-3" />
                        <span className="font-medium">{lastGeneration.num_points}</span>
                      </div>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Temps:</span>
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        <span className="font-medium">{lastGeneration.execution_time}s</span>
                      </div>
                    </div>
                  </div>

                  <div className="text-xs text-muted-foreground">
                    {new Date(lastGeneration.generated_at).toLocaleString('fr-FR')}
                  </div>
                </div>
              ) : (
                <div className="text-center py-6 text-muted-foreground">
                  <MapPin className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Prêt pour la génération</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};