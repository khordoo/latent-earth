import React, { useState, useEffect, useRef } from 'react';
import { GlobeCanvas } from './components/GlobeCanvas';
import { Controls } from './components/Controls';
import { StatsPanel } from './components/StatsPanel';
import { Tooltip } from './components/Tooltip';
import { RunData, RunSummary, ProjectionType, RenderMode, PredictionPoint } from './types';
import { Loader2 } from 'lucide-react';

export const App: React.FC = () => {
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
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(20);

  // Hover Inspection
  const [hoveredPoint, setHoveredPoint] = useState<PredictionPoint | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Load runs list
  useEffect(() => {
    fetch('/api/runs')
      .then((res) => res.json())
      .then((data: RunSummary[]) => {
        setRunsList(data);
        if (data.length > 0) {
          setSelectedRunId(data[0].run_id);
        } else {
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error('Failed to fetch runs:', err);
        setLoading(false);
      });

    // Fetch ground-truth geojson
    fetch('/api/ground-truth')
      .then((res) => res.json())
      .then((data) => setCoastlineGeoJson(data))
      .catch((err) => console.warn('Could not load coastline GeoJSON:', err));
  }, []);

  // Load selected run details
  useEffect(() => {
    if (!selectedRunId) return;
    setLoading(true);
    fetch(`/api/runs/${selectedRunId}`)
      .then((res) => res.json())
      .then((data: RunData) => {
        setRunData(data);
        setVisibleCount(data.points.length);
        setLoading(false);
      })
      .catch((err) => {
        console.error(`Failed to load run ${selectedRunId}:`, err);
        setLoading(false);
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
    <div className="relative w-screen h-screen bg-[#070B14] overflow-hidden select-none">
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
  );
};

export default App;
