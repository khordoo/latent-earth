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
  provider: string;
  model: string;
  raw_response?: any;
}


export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  brier_score?: number | null;
  log_loss?: number | null;
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
  metrics?: EvaluationMetrics | null;
  points: PredictionPoint[];
}

export interface RunSummary {
  run_id: string;
  filename: string;
  provider: string;
  model: string;
  prompt_version: string;
  point_count: number;
  has_evaluation: boolean;
  metrics?: EvaluationMetrics | null;
  modified_time: number;
  size_bytes: number;
}

export type ProjectionType = 'orthographic' | 'equirectangular' | 'mercator';
export type RenderMode = 'probability' | 'binary' | 'uncertainty' | 'ground_truth' | 'error_map';
