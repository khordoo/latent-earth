export interface PredictionPoint {
  coordinate_id: string;
  latitude: number;
  longitude: number;
  cardinal: string;
  probability_land: number | null;
  is_land: boolean;
  uncertainty: number;
  entropy: number;
  sequence_index: number;
  actual_land?: boolean;
  error_category?: 'correct_land' | 'correct_water' | 'false_land' | 'false_water';
  is_coastal?: boolean;
  distance_to_coastline?: number;
  distance_to_coastline_km?: number;
  provider: string;
  model: string;
  raw_response?: any;
}


export interface DetailedMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  brier_score?: number | null;
  log_loss?: number | null;
}

export interface CoastalMetrics {
  coastal_distance_threshold_deg?: number;
  coastal_distance_threshold_km?: number;
  coastal_cell_count: number;
  coastal_cell_fraction: number;
  accuracy?: number | null;
  brier_score?: number | null;
  area_weighted_accuracy?: number | null;
  area_weighted_brier_score?: number | null;
}

export interface AgreementComparison {
  coarse_resolution_deg: number;
  fine_resolution_deg: number;
  subdivision_factor: number;
  children_per_parent: number;
  total_parents: number;
  total_child_evaluations: number;
  child_agreement_rate: number;
  majority_agreement_rate: number;
  mean_probability_delta: number;
  probability_correlation: number;
  coarse_file?: string;
  fine_file?: string;
}

export interface EvaluationMetrics extends DetailedMetrics {
  area_weighted_accuracy?: number;
  area_weighted_brier_score?: number | null;
  coastal_accuracy?: number | null;
  coastal_brier_score?: number | null;
  coastal_area_weighted_accuracy?: number | null;
  grid_metrics?: DetailedMetrics;
  area_weighted_metrics?: DetailedMetrics;
  coastal_metrics?: CoastalMetrics;
}

export interface RegionInfo {
  name?: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface RunData {
  run_id: string;
  provider: string;
  model: string;
  prompt_version: string;
  step_degrees: number;
  convention: string;
  total_coordinates: number;
  sampled_count: number;
  created_at: string;
  region?: RegionInfo | null;
  metrics?: EvaluationMetrics | null;
  grid_metrics?: DetailedMetrics | null;
  area_weighted_metrics?: DetailedMetrics | null;
  coastal_metrics?: CoastalMetrics | null;
  cross_resolution_agreement?: AgreementComparison[] | null;
  points: PredictionPoint[];
}

export interface RunSummary {
  run_id: string;
  filename: string;
  provider: string;
  model: string;
  prompt_version: string;
  region_name?: string;
  step_degrees?: number;
  point_count: number;
  has_evaluation: boolean;
  metrics?: EvaluationMetrics | null;
  modified_time: number;
  size_bytes: number;
}

export type ProjectionType = 'orthographic' | 'equirectangular' | 'mercator';
export type RenderMode = 'probability' | 'binary' | 'uncertainty' | 'ground_truth' | 'error_map';
