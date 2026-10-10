import React from 'react';
import { Play, Pause, RotateCw, Globe, Map, Eye, Layers, FastForward, Crosshair } from 'lucide-react';
import { ProjectionType, RenderMode } from '../types';

interface ControlsProps {
  projectionType: ProjectionType;
  onSelectProjection: (p: ProjectionType) => void;
  renderMode: RenderMode;
  onSelectRenderMode: (m: RenderMode) => void;
  autoRotate: boolean;
  onToggleAutoRotate: () => void;
  showCoastlines: boolean;
  onToggleCoastlines: () => void;
  onRecenterView: () => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  visibleCount: number;
  totalPoints: number;
  onScrub: (val: number) => void;
  playbackSpeed: number;
  onChangeSpeed: (spd: number) => void;
}

export const Controls: React.FC<ControlsProps> = ({
  projectionType,
  onSelectProjection,
  renderMode,
  onSelectRenderMode,
  autoRotate,
  onToggleAutoRotate,
  showCoastlines,
  onToggleCoastlines,
  onRecenterView,
  isPlaying,
  onTogglePlay,
  visibleCount,
  totalPoints,
  onScrub,
  playbackSpeed,
  onChangeSpeed,
}) => {
  const modes: { id: RenderMode; label: string; desc: string }[] = [
    { id: 'probability', label: 'P(Land)', desc: 'Continuous Decision Probability' },
    { id: 'binary', label: 'Binary', desc: 'Crisp Land vs Water (P ≥ 0.5)' },
    { id: 'uncertainty', label: 'Uncertainty', desc: 'Coastline Ambiguity (P ≈ 0.5)' },
    { id: 'ground_truth', label: 'Ground Truth', desc: 'Natural Earth 110m reference' },
    { id: 'error_map', label: 'Error Overlay', desc: 'TP, TN, False Land, False Water' },
  ];

  const speeds = [1, 5, 20, 50, 100, 200];

  const percent = totalPoints > 0 ? (visibleCount / totalPoints) * 100 : 0;
  const percentRound = Math.round(percent);

  return (
    <>
      {/* Top Floating Controls Bar - Responsively anchored on right on narrow screens, centered on 2xl */}
      <div className="absolute top-4 right-4 sm:top-5 sm:right-5 2xl:left-1/2 2xl:-translate-x-1/2 2xl:right-auto z-20 flex flex-wrap items-center gap-1.5 sm:gap-2 p-1 sm:p-1.5 rounded-2xl glass-panel shadow-2xl max-w-[calc(100vw-2rem)]">
        {/* Render Modes */}
        <div className="flex items-center bg-[#070B14]/60 p-1 rounded-xl border border-slate-800/80">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => onSelectRenderMode(m.id)}
              className={`px-2.5 sm:px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                renderMode === m.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
              title={m.desc}
            >
              {m.label}
            </button>
          ))}
        </div>

        <div className="h-5 w-[1px] bg-slate-800 mx-0.5 sm:mx-1 hidden md:block" />

        {/* Projections */}
        <div className="flex items-center bg-[#070B14]/60 p-1 rounded-xl border border-slate-800/80">
          <button
            onClick={() => onSelectProjection('orthographic')}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              projectionType === 'orthographic'
                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Globe className="w-3.5 h-3.5" />
            Globe
          </button>
          <button
            onClick={() => onSelectProjection('equirectangular')}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              projectionType === 'equirectangular'
                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Map className="w-3.5 h-3.5" />
            2D Map
          </button>
        </div>

        <div className="h-5 w-[1px] bg-slate-800 mx-0.5 sm:mx-1 hidden md:block" />

        {/* Quick Toggles */}
        <div className="flex items-center gap-1">
          {projectionType === 'orthographic' && (
            <button
              onClick={onToggleAutoRotate}
              className={`p-1.5 rounded-lg text-xs font-medium border transition-all ${
                autoRotate
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
                  : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
              title="Toggle Auto Rotation"
            >
              <RotateCw className={`w-4 h-4 ${autoRotate ? 'animate-spin' : ''}`} style={{ animationDuration: '6s' }} />
            </button>
          )}

          <button
            onClick={onToggleCoastlines}
            className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-all ${
              showCoastlines
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
                : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
            title="Toggle Natural Earth Coastline Overlay"
          >
            <Layers className="w-3.5 h-3.5" />
            <span className="hidden lg:inline">Coastlines</span>
          </button>

          <button
            onClick={onRecenterView}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium border bg-slate-900/60 text-slate-400 border-slate-800 hover:text-cyan-300 hover:border-cyan-500/40 hover:bg-slate-800/80 transition-all"
            title="Center View on Dataset (Double-click canvas also recenters)"
          >
            <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden sm:inline">Center</span>
          </button>
        </div>
      </div>

      {/* Bottom Playback & Timeline Bar */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 w-11/12 max-w-3xl glass-panel p-3.5 sm:p-4 rounded-2xl flex flex-col gap-2.5 shadow-2xl">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center gap-3">
            <button
              onClick={onTogglePlay}
              className="flex items-center justify-center w-8 h-8 rounded-full bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition-all shadow-lg shadow-cyan-500/30 shrink-0"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <div className="flex flex-col">
              <span className="text-[10px] sm:text-[11px] font-semibold text-slate-200 uppercase tracking-wider">
                Experiment Sampling Playback
              </span>
              <span className="text-xs font-mono text-cyan-400 font-bold">
                {visibleCount.toLocaleString()} / {totalPoints.toLocaleString()} queries ({percentRound}% reconstructed)
              </span>
            </div>
          </div>

          {/* Speed Selector */}
          <div className="flex items-center gap-1 bg-slate-900/80 px-2 py-1 rounded-lg border border-slate-800">
            <span className="text-[10px] uppercase font-bold text-slate-400 mr-1 hidden sm:flex items-center gap-1">
              <FastForward className="w-3 h-3 text-cyan-400" />
              Speed:
            </span>
            {speeds.map((s) => (
              <button
                key={s}
                onClick={() => onChangeSpeed(s)}
                className={`px-1.5 py-0.5 text-[11px] font-mono rounded ${
                  playbackSpeed === s
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Timeline Range Input acting visually as reconstruction progress indicator */}
        <input
          type="range"
          min={1}
          max={totalPoints || 1}
          value={visibleCount}
          onChange={(e) => onScrub(Number(e.target.value))}
          style={{
            background: `linear-gradient(to right, #06B6D4 0%, #06B6D4 ${percent}%, #1E293B ${percent}%, #1E293B 100%)`,
          }}
          className="w-full cursor-pointer h-2.5 rounded-lg appearance-none transition-all"
        />
      </div>
    </>
  );

};
