import React from 'react';
import { Activity, BarChart2, CheckCircle2, AlertTriangle, ShieldCheck } from 'lucide-react';
import { RunData, RenderMode, RunSummary } from '../types';

interface StatsPanelProps {
  runData: RunData | null;
  runsList: RunSummary[];
  selectedRunId: string;
  onSelectRun: (id: string) => void;
  renderMode: RenderMode;
}

export const StatsPanel: React.FC<StatsPanelProps> = ({
  runData,
  runsList,
  selectedRunId,
  onSelectRun,
  renderMode,
}) => {
  if (!runData) return null;

  const metrics = runData.metrics;

  return (
    <div className="absolute top-5 left-5 z-20 w-80 glass-panel p-4 rounded-2xl flex flex-col gap-3.5 shadow-2xl border border-slate-800">
      {/* Title & Run Selector */}
      <div className="flex flex-col gap-1.5 border-b border-slate-800/80 pb-3">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5" />
            Latent Earth
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-semibold uppercase">
            {runData.provider}
          </span>
        </div>

        {/* Run Selector Dropdown */}
        <select
          value={selectedRunId}
          onChange={(e) => onSelectRun(e.target.value)}
          className="bg-[#070B14]/80 text-xs font-semibold text-slate-200 border border-slate-700/80 rounded-lg px-2.5 py-1.5 outline-none hover:border-cyan-500/50 transition-colors"
        >
          {runsList.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.model} ({r.point_count.toLocaleString()} pts) • {r.run_id}
            </option>
          ))}
        </select>

        <div className="flex items-center justify-between text-xs text-slate-400 mt-1">
          <span>Model: <strong className="text-slate-200">{runData.model}</strong></span>
          <span className="font-mono text-cyan-400">{runData.step_degrees}° Grid</span>
        </div>
      </div>

      {/* Metrics Card (if available) */}
      {metrics && (
        <div className="bg-[#070B14]/60 p-3 rounded-xl border border-slate-800/80 flex flex-col gap-2">
          <div className="flex items-center justify-between text-[11px] font-bold text-slate-300 uppercase tracking-wider">
            <span className="flex items-center gap-1">
              <BarChart2 className="w-3.5 h-3.5 text-cyan-400" />
              Evaluation vs Ground Truth
            </span>
            <span className="font-mono text-emerald-400 font-bold">
              {Math.round(metrics.accuracy * 100)}% Acc
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">F1-Score</span>
              <span className="font-mono font-bold text-slate-200">{metrics.f1.toFixed(3)}</span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Brier Score</span>
              <span className="font-mono font-bold text-cyan-300">
                {metrics.brier_score !== null && metrics.brier_score !== undefined
                  ? metrics.brier_score.toFixed(3)
                  : 'N/A'}
              </span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Precision</span>
              <span className="font-mono font-bold text-slate-200">{metrics.precision.toFixed(3)}</span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Recall</span>
              <span className="font-mono font-bold text-slate-200">{metrics.recall.toFixed(3)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Dynamic Legend based on active mode */}
      <div className="flex flex-col gap-1.5 pt-1">
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
          Color Scale: {renderMode.toUpperCase().replace('_', ' ')}
        </span>

        {renderMode === 'probability' && (
          <div className="flex flex-col gap-1">
            <div
              className="h-3 w-full rounded-md shadow-inner"
              style={{
                background: 'linear-gradient(to right, #081225, #1E3A8A, #0284C7, #06B6D4, #F59E0B, #16A34A, #F1F5F9)',
              }}
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>0.0 (Ocean)</span>
              <span>0.5 (Coast)</span>
              <span>1.0 (Land)</span>
            </div>
          </div>
        )}

        {renderMode === 'binary' && (
          <div className="flex items-center gap-4 text-xs font-medium">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-[#0B132B] border border-slate-700" />
              <span className="text-slate-400">Water (P &lt; 0.5)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-[#F8FAFC] border border-slate-400" />
              <span className="text-slate-200">Land (P ≥ 0.5)</span>
            </div>
          </div>
        )}

        {renderMode === 'uncertainty' && (
          <div className="flex flex-col gap-1">
            <div
              className="h-3 w-full rounded-md shadow-inner"
              style={{
                background: 'linear-gradient(to right, #090D16, #2E1065, #701A75, #C026D3, #EA580C, #FDE047)',
              }}
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>0.0 (Certain)</span>
              <span>0.5</span>
              <span>1.0 (Max Ambiguity)</span>
            </div>
          </div>
        )}

        {renderMode === 'ground_truth' && (
          <div className="flex items-center gap-4 text-xs font-medium">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-[#0F2942]" />
              <span className="text-slate-400">True Water</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-[#10B981]" />
              <span className="text-emerald-300">True Land</span>
            </div>
          </div>
        )}

        {renderMode === 'error_map' && (
          <div className="grid grid-cols-2 gap-1.5 text-[11px] font-medium">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#0F2942]" />
              <span className="text-slate-400">Correct Water</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#10B981]" />
              <span className="text-emerald-300">Correct Land</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#EF4444]" />
              <span className="text-red-400">False Land</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#F59E0B]" />
              <span className="text-amber-400">False Water</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
