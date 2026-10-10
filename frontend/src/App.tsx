import React, { useState, useEffect } from 'react';
import { GlobeCanvas } from './components/GlobeCanvas';
import { Controls } from './components/Controls';
import { StatsPanel } from './components/StatsPanel';
import { Tooltip } from './components/Tooltip';
import { Billboard } from './components/Billboard';
import { RunData, RunSummary, ProjectionType, RenderMode, PredictionPoint } from './types';
import { Loader2, Globe, Trophy, Sparkles } from 'lucide-react';
import { REGIONAL_JAPAN_KOREA_2112 } from './data/billboardData';

export const App: React.FC = () => {
  // Navigation: 'globe' or 'billboard'
  const [activeTab, setActiveTab] = useState<'globe' | 'billboard'>('globe');

  const [runsList, setRunsList] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [runData, setRunData] = useState<RunData | null>(null);
  const [coastlineGeoJson, setCoastlineGeoJson] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Visual & Projection Settings
  const [projectionType, setProjectionType] = useState<ProjectionType>('orthographic');
  const [renderMode, setRenderMode] = useState<RenderMode>('probability');
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [showCoastlines, setShowCoastlines] = useState<boolean>(false);

  // Playback / Timeline Animation
  const [visibleCount, setVisibleCount] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(100);

  // Hover Inspection
  const [hoveredPoint, setHoveredPoint] = useState<PredictionPoint | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Fallback runs list for Vercel / static deployments
  const fallbackRuns: RunSummary[] = REGIONAL_JAPAN_KOREA_2112.models.map((m) => ({
    run_id: m.runId,
    filename: `${m.runId}.jsonl`,
    provider: m.provider,
    model: m.name,
    prompt_version: 'land-water-v1',
    region_name: 'japan_korea',
    step_degrees: 0.5,
    point_count: 2112,
    has_evaluation: true,
    metrics: {
      accuracy: m.gridAccuracy / 100,
      precision: m.precision / 100,
      recall: m.recall / 100,
      f1: m.f1,
      brier_score: m.brierScore,
      area_weighted_accuracy: m.areaWeightedAccuracy / 100,
      coastal_accuracy: m.coastalAccuracy / 100,
    },
    modified_time: Date.now(),
    size_bytes: 1000000,
  }));

  // Load runs list (tries local API first, falls back to static bundled runs for Vercel)
  useEffect(() => {
    fetch('/api/runs')
      .then((res) => {
        if (!res.ok) throw new Error('API unavailable');
        return res.json();
      })
      .then((data: RunSummary[]) => {
        if (data && data.length > 0) {
          setRunsList(data);
          setSelectedRunId(data[0].run_id);
        } else {
          setRunsList(fallbackRuns);
          setSelectedRunId(fallbackRuns[0].run_id);
        }
      })
      .catch(() => {
        // Vercel serverless / static fallback
        setRunsList(fallbackRuns);
        if (fallbackRuns.length > 0) {
          setSelectedRunId(fallbackRuns[0].run_id);
        } else {
          setLoading(false);
        }
      });

    // Fetch ground-truth geojson (tries local API first, falls back to public static asset)
    fetch('/api/ground-truth')
      .then((res) => {
        if (!res.ok) throw new Error('API unavailable');
        return res.json();
      })
      .then((data) => setCoastlineGeoJson(data))
      .catch(() => {
        fetch('/data/ne_110m_land.geojson')
          .then((res) => res.json())
          .then((data) => setCoastlineGeoJson(data))
          .catch((err) => console.warn('Could not load coastline GeoJSON:', err));
      });
  }, []);

  // Load selected run details (tries local API first, falls back to static JSON in /data/runs/)
  useEffect(() => {
    if (!selectedRunId) return;
    setLoading(true);

    fetch(`/api/runs/${selectedRunId}`)
      .then((res) => {
        if (!res.ok) throw new Error('Local API unavailable');
        return res.json();
      })
      .then((data: RunData) => {
        setRunData(data);
        setVisibleCount(data.points.length);
        setLoading(false);
      })
      .catch(() => {
        // Fallback for Vercel static hosting
        fetch(`/data/runs/${selectedRunId}.json`)
          .then((res) => {
            if (!res.ok) throw new Error('Static file not found');
            return res.json();
          })
          .then((data: RunData) => {
            setRunData(data);
            setVisibleCount(data.points.length);
            setLoading(false);
          })
          .catch((err) => {
            console.error(`Failed to load run ${selectedRunId}:`, err);
            setLoading(false);
          });
      });
  }, [selectedRunId]);

  // Playback animation tick
  useEffect(() => {
    if (!isPlaying || !runData) return;

    const interval = setInterval(() => {
      setVisibleCount((prev) => {
        if (prev >= runData.points.length) {
          setIsPlaying(false);
          return runData.points.length;
        }
        return Math.min(runData.points.length, prev + playbackSpeed);
      });
    }, 50);

    return () => clearInterval(interval);
  }, [isPlaying, runData, playbackSpeed]);

  const handleTogglePlay = () => {
    if (!runData) return;
    if (visibleCount >= runData.points.length) {
      setVisibleCount(1);
      setIsPlaying(true);
    } else {
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="relative w-screen h-screen bg-[#070B14] overflow-hidden select-none flex flex-col">
      {/* Top Main Navigation Tabs Bar */}
      <header className="z-30 w-full flex items-center justify-between px-4 sm:px-6 py-2.5 bg-[#070B14]/90 backdrop-blur-md border-b border-slate-800/80 shadow-md">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            <span className="font-extrabold text-sm sm:text-base tracking-tight bg-gradient-to-r from-slate-100 via-cyan-200 to-sky-400 bg-clip-text text-transparent">
              Latent Earth
            </span>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-semibold uppercase hidden md:inline">
            Geospatial Decision Platform
          </span>
        </div>

        {/* Navigation Switcher Pills */}
        <nav className="flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-800 shadow-inner">
          <button
            onClick={() => setActiveTab('globe')}
            className={`px-3 sm:px-4 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeTab === 'globe'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 shadow-md font-extrabold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <Globe className="w-3.5 h-3.5" />
            <span>3D Globe & Map</span>
          </button>

          <button
            onClick={() => setActiveTab('billboard')}
            className={`px-3 sm:px-4 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 relative ${
              activeTab === 'billboard'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 shadow-md font-extrabold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <Trophy className="w-3.5 h-3.5 text-amber-300" />
            <span>Billboard</span>
            <span className="flex h-1.5 w-1.5 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
            </span>
          </button>
        </nav>

        {/* Right Status Indicator */}
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span className="hidden lg:inline text-[11px] text-slate-400">
            {runsList.length} Runs Evaluated
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold hidden sm:inline">
            Vercel Ready
          </span>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="relative flex-1 w-full h-[calc(100vh-48px)] overflow-hidden">
        {activeTab === 'billboard' ? (
          <Billboard
            onSelectModelForGlobe={(runId) => {
              setSelectedRunId(runId);
              setActiveTab('globe');
            }}
            onSwitchToGlobe={() => setActiveTab('globe')}
          />
        ) : (
          <div className="relative w-full h-full overflow-hidden">
            {/* Loading Overlay */}
            {loading && !runData && (
              <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-[#070B14]/80 backdrop-blur-md">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin mb-3" />
                <span className="text-sm font-semibold tracking-wider text-slate-300">
                  Loading Geographic Model Reconstructions...
                </span>
              </div>
            )}
            {loading && runData && (
              <div className="absolute top-4 right-4 z-40 flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-cyan-500/50 text-cyan-300 text-xs font-semibold shadow-lg backdrop-blur-sm">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Switching resolution...</span>
              </div>
            )}

            {/* Main Interactive Canvas */}
            {runData && (
              <GlobeCanvas
                points={runData.points}
                stepDegrees={runData.step_degrees}
                projectionType={projectionType}
                renderMode={renderMode}
                visibleCount={visibleCount}
                autoRotate={autoRotate}
                showCoastlineOverlay={showCoastlines}
                coastlineGeoJson={coastlineGeoJson}
                onHoverPoint={(point, pos) => {
                  setHoveredPoint(point);
                  setMousePos(pos);
                }}
              />
            )}

            {/* Top Controls & Bottom Playback Bar */}
            {runData && (
              <Controls
                projectionType={projectionType}
                onSelectProjection={setProjectionType}
                renderMode={renderMode}
                onSelectRenderMode={setRenderMode}
                autoRotate={autoRotate}
                onToggleAutoRotate={() => setAutoRotate(!autoRotate)}
                showCoastlines={showCoastlines}
                onToggleCoastlines={() => setShowCoastlines(!showCoastlines)}
                isPlaying={isPlaying}
                onTogglePlay={handleTogglePlay}
                visibleCount={visibleCount}
                totalPoints={runData.points.length}
                onScrub={(val) => {
                  setIsPlaying(false);
                  setVisibleCount(val);
                }}
                playbackSpeed={playbackSpeed}
                onChangeSpeed={setPlaybackSpeed}
              />
            )}

            {/* Stats and Run Switcher Card */}
            <StatsPanel
              runData={runData}
              runsList={runsList}
              selectedRunId={selectedRunId}
              onSelectRun={setSelectedRunId}
              renderMode={renderMode}
            />

            {/* Hover Inspection Tooltip */}
            <Tooltip point={hoveredPoint} mousePos={mousePos} />
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
