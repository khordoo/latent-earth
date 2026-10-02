import React from 'react';
import { PredictionPoint } from '../types';

interface TooltipProps {
  point: PredictionPoint | null;
  mousePos: { x: number; y: number } | null;
}

function formatCoordinate(lat: number, lon: number): string {
  const latDir = lat >= 0 ? 'N' : 'S';
  const lonDir = lon >= 0 ? 'E' : 'W';
  const absLat = Math.abs(lat);
  const absLon = Math.abs(lon);
  const latStr = Number.isInteger(absLat) ? absLat.toString() : absLat.toFixed(1);
  const lonStr = Number.isInteger(absLon) ? absLon.toString() : absLon.toFixed(1);
  return `${latStr}° ${latDir}, ${lonStr}° ${lonDir}`;
}

function getRawAnswerString(point: PredictionPoint): string {
  const raw = point.raw_response;
  if (!raw) {
    if (point.probability_land !== null && point.probability_land !== undefined) {
      return point.probability_land.toFixed(2);
    }
    return 'N/A';
  }
  // TypeSafe Jev structure: raw.answers.is_land.noul
  if (raw.answers?.is_land?.noul !== undefined) {
    return String(raw.answers.is_land.noul);
  }
  if (typeof raw === 'number' || typeof raw === 'string') {
    return String(raw);
  }
  if (raw.decision !== undefined) {
    return String(raw.decision);
  }
  if (raw.text !== undefined) {
    return String(raw.text);
  }
  if (raw.choices?.[0]?.message?.content) {
    return String(raw.choices[0].message.content).trim();
  }
  if (point.probability_land !== null && point.probability_land !== undefined) {
    return point.probability_land.toFixed(2);
  }
  return JSON.stringify(raw);
}

export const Tooltip: React.FC<TooltipProps> = ({ point, mousePos }) => {
  if (!point || !mousePos) return null;

  const coordStr = formatCoordinate(point.latitude, point.longitude);
  const pLand =
    point.probability_land !== null && point.probability_land !== undefined
      ? point.probability_land.toFixed(2)
      : 'N/A';
  const predictionStr = point.is_land ? 'LAND' : 'WATER';
  const groundTruthStr =
    point.actual_land !== undefined && point.actual_land !== null
      ? point.actual_land
        ? 'LAND'
        : 'WATER'
      : 'N/A';
  const isCorrect =
    point.actual_land !== undefined && point.actual_land !== null
      ? point.is_land === point.actual_land
      : null;
  const uncertaintyStr = point.uncertainty.toFixed(2);
  const rawResponseStr = getRawAnswerString(point);

  return (
    <div
      className="fixed pointer-events-none z-50 p-3.5 rounded-xl border border-cyan-500/40 shadow-2xl flex flex-col gap-2.5 w-64 bg-[#090F1E]/95 backdrop-blur-md text-xs font-mono"
      style={{
        left: Math.min(window.innerWidth - 280, mousePos.x + 16),
        top: Math.min(window.innerHeight - 240, mousePos.y + 16),
      }}
    >
      {/* Coordinate Header */}
      <div className="flex items-center justify-between border-b border-slate-700/80 pb-2">
        <span className="font-bold text-cyan-300 text-sm tracking-wide">
          {coordStr}
        </span>
        <span
          className={`text-[10px] font-bold px-2 py-0.5 rounded font-sans uppercase tracking-wider ${
            point.is_land
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
          }`}
        >
          {predictionStr}
        </span>
      </div>

      {/* Structured Key-Value Inspection Table */}
      <div className="flex flex-col gap-1.5 text-xs">
        <div className="flex justify-between items-center py-0.5 border-b border-slate-800/60">
          <span className="text-slate-400">P(land)</span>
          <span className="font-bold text-slate-100">{pLand}</span>
        </div>

        <div className="flex justify-between items-center py-0.5 border-b border-slate-800/60">
          <span className="text-slate-400">Prediction</span>
          <span className={`font-bold ${point.is_land ? 'text-emerald-400' : 'text-blue-400'}`}>
            {predictionStr}
          </span>
        </div>

        <div className="flex justify-between items-center py-0.5 border-b border-slate-800/60">
          <span className="text-slate-400">Ground truth</span>
          <span
            className={`font-bold ${
              point.actual_land === true
                ? 'text-emerald-300'
                : point.actual_land === false
                ? 'text-blue-300'
                : 'text-slate-500'
            }`}
          >
            {groundTruthStr}
          </span>
        </div>

        <div className="flex justify-between items-center py-0.5 border-b border-slate-800/60">
          <span className="text-slate-400">Correct</span>
          {isCorrect === true && (
            <span className="text-emerald-400 font-bold flex items-center gap-1">
              ✓ <span className="text-[10px] text-emerald-400/80 font-sans font-normal">({point.is_land ? 'TP' : 'TN'})</span>
            </span>
          )}
          {isCorrect === false && (
            <span className="text-red-400 font-bold flex items-center gap-1">
              ✗ <span className="text-[10px] text-red-400/80 font-sans font-normal">({point.is_land ? 'False Land' : 'False Water'})</span>
            </span>
          )}
          {isCorrect === null && (
            <span className="text-slate-500">N/A</span>
          )}
        </div>

        <div className="flex justify-between items-center py-0.5 border-b border-slate-800/60">
          <span className="text-slate-400">Uncertainty</span>
          <span className="font-bold text-amber-300">{uncertaintyStr}</span>
        </div>

        {(point.distance_to_coastline_km !== undefined || point.distance_to_coastline !== undefined) && (
          <div className="flex justify-between items-center py-0.5 border-b border-slate-800/60">
            <span className="text-slate-400">Distance to coast</span>
            <span className="font-bold text-slate-200">
              {point.distance_to_coastline_km !== undefined
                ? `${point.distance_to_coastline_km.toFixed(1)} km`
                : `${(point.distance_to_coastline! * 111.0).toFixed(1)} km`}
            </span>
          </div>
        )}

        {rawResponseStr && (
          <div className="flex justify-between items-center pt-0.5">
            <span className="text-slate-400">Raw response</span>
            <span className="text-cyan-300 font-bold">{rawResponseStr}</span>
          </div>
        )}
      </div>
    </div>
  );
};
