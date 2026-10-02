import React from 'react';
import { PredictionPoint } from '../types';
import { Compass, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';

interface TooltipProps {
  point: PredictionPoint | null;
  mousePos: { x: number; y: number } | null;
}

export const Tooltip: React.FC<TooltipProps> = ({ point, mousePos }) => {
  if (!point || !mousePos) return null;

  const pLand = point.probability_land !== null ? (point.probability_land * 100).toFixed(1) : 'N/A';

  return (
    <div
      className="fixed pointer-events-none z-50 glass-panel p-3 rounded-xl border border-cyan-500/40 shadow-2xl flex flex-col gap-1.5 w-60 text-xs"
      style={{
        left: Math.min(window.innerWidth - 260, mousePos.x + 15),
        top: Math.min(window.innerHeight - 200, mousePos.y + 15),
      }}
    >
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-1.5">
        <span className="font-mono text-cyan-300 font-bold flex items-center gap-1">
          <Compass className="w-3.5 h-3.5 text-cyan-400" />
          {point.cardinal}
        </span>
        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
          point.is_land ? 'bg-emerald-500/20 text-emerald-300' : 'bg-blue-500/20 text-blue-300'
        }`}>
          {point.is_land ? 'LAND' : 'WATER'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-1.5 text-[11px] pt-1">
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400">P(Land)</span>
          <span className="font-mono font-bold text-slate-100">{pLand}%</span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400">Uncertainty</span>
          <span className="font-mono font-bold text-slate-100">{(point.uncertainty * 100).toFixed(1)}%</span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400">Latitude</span>
          <span className="font-mono text-slate-300">{point.latitude.toFixed(3)}°</span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400">Longitude</span>
          <span className="font-mono text-slate-300">{point.longitude.toFixed(3)}°</span>
        </div>
      </div>

      {point.actual_land !== undefined && (
        <div className="border-t border-slate-700/60 pt-1.5 flex items-center justify-between text-[11px]">
          <span className="text-slate-400">Ground Truth:</span>
          <span className="font-semibold text-slate-200">
            {point.actual_land ? 'Actual Land' : 'Actual Water'}
          </span>
        </div>
      )}

      {point.error_category && (
        <div className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider">
          {point.error_category === 'correct_land' && (
            <span className="text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" /> Correct Land (TP)
            </span>
          )}
          {point.error_category === 'correct_water' && (
            <span className="text-blue-400 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" /> Correct Water (TN)
            </span>
          )}
          {point.error_category === 'false_land' && (
            <span className="text-red-400 flex items-center gap-1">
              <XCircle className="w-3 h-3" /> False Land (FP)
            </span>
          )}
          {point.error_category === 'false_water' && (
            <span className="text-amber-400 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" /> False Water (FN)
            </span>
          )}
        </div>
      )}
    </div>
  );
};
