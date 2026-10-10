import React, { useState, useMemo } from 'react';
import {
  Trophy,
  Zap,
  TrendingUp,
  AlertTriangle,
  Clock,
  Compass,
  DollarSign,
  Search,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  BarChart3,
  Globe2,
  Sparkles,
  ArrowUpDown,
  Filter,
} from 'lucide-react';
import {
  BillboardModelEntry,
  ALL_BENCHMARK_DATASETS,
  BenchmarkDatasetInfo,
} from '../data/billboardData';

interface BillboardProps {
  onSelectModelForGlobe?: (runId: string) => void;
  onSwitchToGlobe?: () => void;
}

type SortField =
  | 'paretoRank'
  | 'balancedAccuracy'
  | 'gridAccuracy'
  | 'p90LatencySeconds'
  | 'brierScore'
  | 'coastalAccuracy'
  | 'costPer1kQueriesUsd';

export const Billboard: React.FC<BillboardProps> = ({
  onSelectModelForGlobe,
  onSwitchToGlobe,
}) => {
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('japan_korea_2112');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterType, setFilterType] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('paretoRank');
  const [sortAscending, setSortAscending] = useState<boolean>(true);
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);
  const [hoveredScatterModel, setHoveredScatterModel] = useState<BillboardModelEntry | null>(null);

  const currentDataset: BenchmarkDatasetInfo = useMemo(() => {
    return (
      ALL_BENCHMARK_DATASETS.find((d) => d.id === selectedDatasetId) ||
      ALL_BENCHMARK_DATASETS[0]
    );
  }, [selectedDatasetId]);

  // Filter and sort models
  const filteredModels = useMemo(() => {
    return currentDataset.models
      .filter((m) => {
        const matchesSearch =
          m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          m.provider.toLowerCase().includes(searchQuery.toLowerCase()) ||
          m.architecture.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesType =
          filterType === 'all' ||
          (filterType === 'decision' && m.modelType === 'decision') ||
          (filterType === 'reasoning' && m.modelType === 'reasoning') ||
          (filterType === 'specialized' && m.modelType === 'specialized') ||
          (filterType === 'local_gguf' && m.modelType === 'local_gguf');
        return matchesSearch && matchesType;
      })
      .sort((a, b) => {
        let valA = a[sortField] ?? 0;
        let valB = b[sortField] ?? 0;

        // Invert comparison for metrics where lower is better (latency, brier, cost, rank)
        const lowerIsBetter =
          sortField === 'paretoRank' ||
          sortField === 'p90LatencySeconds' ||
          sortField === 'brierScore' ||
          sortField === 'costPer1kQueriesUsd';

        if (lowerIsBetter) {
          return sortAscending ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
        } else {
          return sortAscending ? (valB as number) - (valA as number) : (valA as number) - (valB as number);
        }
      });
  }, [currentDataset, searchQuery, filterType, sortField, sortAscending]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAscending(!sortAscending);
    } else {
      setSortField(field);
      setSortAscending(true);
    }
  };

  const handleInspectOnGlobe = (runId: string) => {
    if (onSelectModelForGlobe) {
      onSelectModelForGlobe(runId);
    }
    if (onSwitchToGlobe) {
      onSwitchToGlobe();
    }
  };

  return (
    <div className="w-full h-full overflow-y-auto bg-[#070B14] text-slate-100 p-4 sm:p-6 lg:p-8 font-sans select-text">
      {/* Header Banner */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-2.5 mb-2">
              <span className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                <Trophy className="w-6 h-6" />
              </span>
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-slate-100 via-cyan-200 to-sky-400 bg-clip-text text-transparent">
                Model Decision Billboard
              </h1>
              <span className="text-[11px] font-mono uppercase px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-bold">
                Live Benchmarks
              </span>
            </div>
            <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
              Empirical geospatial intelligence across <strong>10 model architectures</strong> (144M to Frontier scale).
              Evaluates non-autoregressive decision models, multimodal geo-foundations, and autoregressive LLMs on the
              continuous coordinate classification manifold.
            </p>
          </div>

          {/* Dataset Switcher Pill */}
          <div className="flex items-center bg-slate-900/90 p-1.5 rounded-2xl border border-slate-800 shadow-xl self-start md:self-auto">
            {ALL_BENCHMARK_DATASETS.map((dataset) => (
              <button
                key={dataset.id}
                onClick={() => setSelectedDatasetId(dataset.id)}
                className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
                  selectedDatasetId === dataset.id
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 shadow-md shadow-cyan-500/20 font-extrabold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Compass className="w-3.5 h-3.5" />
                <span>{dataset.name.split('(')[0].trim()}</span>
                <span className="text-[10px] font-mono opacity-80">
                  {dataset.pointCount.toLocaleString()} pts
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* 4 Hero KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
          <div className="bg-slate-900/60 p-4 rounded-2xl border border-emerald-500/30 shadow-lg backdrop-blur-sm relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none" />
            <div className="flex items-center justify-between text-xs font-bold text-emerald-400 mb-2">
              <span className="flex items-center gap-1.5">
                <Trophy className="w-4 h-4 text-emerald-400" />
                #1 Pareto Decision Winner
              </span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                0.35s Median
              </span>
            </div>
            <div className="text-xl font-extrabold text-slate-100">Perplexity Pplx-Decider</div>
            <div className="text-xs text-slate-400 mt-1">
              <strong>83.57% Acc</strong> • 79.09% Balanced • <strong>$0.02 / 1M</strong> tokens
            </div>
            <div className="text-[11px] text-emerald-300/80 mt-2 font-medium">
              Single-pass System-1 without reasoning latency penalty
            </div>
          </div>

          <div className="bg-slate-900/60 p-4 rounded-2xl border border-sky-500/30 shadow-lg backdrop-blur-sm relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-sky-500/5 rounded-full blur-2xl pointer-events-none" />
            <div className="flex items-center justify-between text-xs font-bold text-sky-400 mb-2">
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-sky-400" />
                Frontier Accuracy Ceiling
              </span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300">
                97.6% Acc
              </span>
            </div>
            <div className="text-xl font-extrabold text-slate-100">GPT-6-Luna</div>
            <div className="text-xs text-slate-400 mt-1">
              <strong>96.67% Bal Acc</strong> • 0.0196 Brier • <strong>95.3% Coastal</strong>
            </div>
            <div className="text-[11px] text-sky-300/80 mt-2 font-medium">
              Near-perfect coastline fidelity, but requires 3.4s latency
            </div>
          </div>

          <div className="bg-slate-900/60 p-4 rounded-2xl border border-amber-500/30 shadow-lg backdrop-blur-sm relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/5 rounded-full blur-2xl pointer-events-none" />
            <div className="flex items-center justify-between text-xs font-bold text-amber-400 mb-2">
              <span className="flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-amber-400" />
                Fastest Edge Inference
              </span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300">
                69 ms
              </span>
            </div>
            <div className="text-xl font-extrabold text-slate-100">Julia-1 (144M)</div>
            <div className="text-xs text-slate-400 mt-1">
              <strong>54.9 req/s</strong> throughput • Zero API cost • Local Metal
            </div>
            <div className="text-[11px] text-amber-300/80 mt-2 font-medium">
              Ultra-light edge, but exhibits integer longitude plaid striping
            </div>
          </div>

          <div className="bg-slate-900/60 p-4 rounded-2xl border border-purple-500/30 shadow-lg backdrop-blur-sm relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full blur-2xl pointer-events-none" />
            <div className="flex items-center justify-between text-xs font-bold text-purple-400 mb-2">
              <span className="flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-purple-400" />
                The Latency Disqualification
              </span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
                12.7s P90
              </span>
            </div>
            <div className="text-xl font-extrabold text-slate-100">Fastino GLiDE</div>
            <div className="text-xs text-slate-400 mt-1">
              88.9% Acc • <strong>12.68s P90 Tail</strong> • 1,026 CoT tokens
            </div>
            <div className="text-[11px] text-purple-300/80 mt-2 font-medium">
              Deranked from #2 because coasts trigger slow autoregressive CoT
            </div>
          </div>
        </div>

        {/* Interactive Pareto Scatter Visualizer */}
        <div className="mt-8 bg-slate-900/50 p-5 sm:p-6 rounded-2xl border border-slate-800 shadow-xl backdrop-blur-sm">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                The Latency vs. Accuracy Pareto Frontier
              </h2>
              <p className="text-xs text-slate-400">
                Models in the upper-left (high accuracy, low latency) dominate. Notice the severe 12+ second latency gap between Fastino GLiDE and Perplexity Pplx-Decider.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400/50" />
                Pareto Optimal
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
                Dominated / Sub-optimal
              </span>
            </div>
          </div>

          {/* Pareto SVG Chart Container */}
          <div className="relative w-full h-64 sm:h-72 bg-[#050811] rounded-xl border border-slate-800/80 p-4 overflow-hidden">
            <svg className="w-full h-full" viewBox="0 0 800 240" preserveAspectRatio="none">
              {/* Grid Lines */}
              <line x1="50" y1="20" x2="780" y2="20" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="50" y1="70" x2="780" y2="70" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="50" y1="120" x2="780" y2="120" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="50" y1="170" x2="780" y2="170" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="50" y1="210" x2="780" y2="210" stroke="#334155" />

              <line x1="50" y1="20" x2="50" y2="210" stroke="#334155" />
              <line x1="200" y1="20" x2="200" y2="210" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="400" y1="20" x2="400" y2="210" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="600" y1="20" x2="600" y2="210" stroke="#1e293b" strokeDasharray="3 3" />

              {/* Axis Labels */}
              <text x="45" y="25" fill="#64748b" fontSize="10" textAnchor="end" fontFamily="monospace">100%</text>
              <text x="45" y="75" fill="#64748b" fontSize="10" textAnchor="end" fontFamily="monospace">80%</text>
              <text x="45" y="125" fill="#64748b" fontSize="10" textAnchor="end" fontFamily="monospace">60%</text>
              <text x="45" y="175" fill="#64748b" fontSize="10" textAnchor="end" fontFamily="monospace">40%</text>

              <text x="50" y="228" fill="#64748b" fontSize="10" textAnchor="middle" fontFamily="monospace">0.1s</text>
              <text x="200" y="228" fill="#64748b" fontSize="10" textAnchor="middle" fontFamily="monospace">0.5s</text>
              <text x="400" y="228" fill="#64748b" fontSize="10" textAnchor="middle" fontFamily="monospace">2.5s</text>
              <text x="600" y="228" fill="#64748b" fontSize="10" textAnchor="middle" fontFamily="monospace">6.0s</text>
              <text x="750" y="228" fill="#64748b" fontSize="10" textAnchor="middle" fontFamily="monospace">13.0s (P90)</text>

              {/* Empirical Pareto Curve Line (connecting Julia -> Pplx -> Luna) */}
              <path
                d="M 60 145 Q 160 80 200 68 T 480 25"
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
                strokeDasharray="4 4"
                opacity="0.6"
              />

              {/* Data Points Mapping */}
              {currentDataset.models.map((m) => {
                // Map log latency from 0.05 to 13s into X [60, 750]
                const latClamped = Math.max(0.05, Math.min(13.0, m.p90LatencySeconds));
                const logMin = Math.log10(0.05);
                const logMax = Math.log10(13.0);
                const normX = (Math.log10(latClamped) - logMin) / (logMax - logMin);
                const cx = 60 + normX * (730 - 60);

                // Map balanced accuracy from 20% to 100% into Y [210, 20]
                const normY = (m.balancedAccuracy - 20) / (100 - 20);
                const cy = 210 - normY * (210 - 25);

                const isPplx = m.id.includes('pplx');
                const isLunaDecision = m.id.includes('decision') && (m.provider.toLowerCase().includes('openai') || m.id.includes('openai'));
                const isLunaChat = m.id.includes('luna') && !isLunaDecision;
                const isClef = m.id.includes('clef');
                const isGlide = m.id.includes('glide');
                const isMsft = m.id.includes('microsoft');

                const isHovered = hoveredScatterModel?.id === m.id;

                const dotColor = isLunaDecision
                  ? '#06b6d4'
                  : isLunaChat
                  ? '#38bdf8'
                  : isPplx
                  ? '#10b981'
                  : isClef
                  ? '#f97316'
                  : isGlide
                  ? '#f43f5e'
                  : isMsft
                  ? '#f59e0b'
                  : m.isParetoOptimal
                  ? '#10b981'
                  : '#64748b';

                // Distinct short display name for each point
                let shortLabel = m.name.split(' ')[0];
                if (isLunaDecision) shortLabel = 'OpenAI Dec';
                else if (isLunaChat) shortLabel = 'Luna (Chat)';
                else if (isClef) shortLabel = 'Clef (27B)';
                else if (isPplx) shortLabel = 'Pplx-Decider';
                else if (isGlide) shortLabel = 'GLiDE';
                else if (isMsft) shortLabel = 'MSFT Dec-1';
                else if (m.id.includes('gemma')) shortLabel = 'Gemma-4';
                else if (m.id.includes('liquid')) shortLabel = 'Liquid D1';
                else if (m.id.includes('jev')) shortLabel = 'JEV';
                else if (m.id.includes('julia')) shortLabel = 'Julia-1';
                else if (m.id.includes('kev')) shortLabel = 'Kev-4B';
                else if (m.id.includes('laya')) shortLabel = 'Laya';

                return (
                  <g
                    key={m.id}
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredScatterModel(m)}
                    onMouseLeave={() => setHoveredScatterModel(null)}
                    onClick={() => {
                      if (onSelectModelForGlobe) onSelectModelForGlobe(m.runId);
                      if (onSwitchToGlobe) onSwitchToGlobe();
                    }}
                  >
                    {/* Generous hit-target circle prevents cursor boundary chatter */}
                    <circle cx={cx} cy={cy} r={16} fill="transparent" />

                    {/* Point Circle (radius changes cleanly without position displacement) */}
                    <circle
                      cx={cx}
                      cy={cy}
                      r={isHovered ? (m.isParetoOptimal ? 9.5 : 7.5) : (m.isParetoOptimal ? 6.5 : 4.5)}
                      fill={dotColor}
                      stroke={isHovered ? '#ffffff' : '#ffffffbb'}
                      strokeWidth={isHovered ? 2.5 : (m.isParetoOptimal ? 2 : 1)}
                    />

                    {/* Distinct Label */}
                    <text
                      x={cx}
                      y={cy - (isHovered ? 13 : 9)}
                      fill={isHovered ? '#38bdf8' : (m.isParetoOptimal ? '#ffffff' : '#94a3b8')}
                      fontSize={isHovered ? '10' : '8.5'}
                      fontWeight={isHovered || m.isParetoOptimal ? 'bold' : 'normal'}
                      textAnchor="middle"
                      className="pointer-events-none select-none drop-shadow"
                    >
                      {shortLabel} ({m.balancedAccuracy.toFixed(0)}%)
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Rich Hover Floating Tooltip Overlay */}
            {hoveredScatterModel && (
              <div className="absolute top-3 right-3 z-20 px-3.5 py-2.5 rounded-xl border border-cyan-500/40 shadow-2xl bg-slate-950/95 text-xs backdrop-blur-md max-w-xs pointer-events-none transition-all">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="font-bold text-slate-100 truncate">{hoveredScatterModel.name}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                      hoveredScatterModel.isParetoOptimal
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    #{hoveredScatterModel.paretoRank}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400 mb-2 truncate">
                  {hoveredScatterModel.architecture}
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[11px]">
                  <div className="text-slate-400">
                    Bal Acc:{' '}
                    <span className="text-emerald-400 font-bold">
                      {hoveredScatterModel.balancedAccuracy.toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-slate-400">
                    P90 Lat:{' '}
                    <span className="text-cyan-300 font-bold">
                      {hoveredScatterModel.p90LatencySeconds < 1
                        ? `${(hoveredScatterModel.p90LatencySeconds * 1000).toFixed(0)}ms`
                        : `${hoveredScatterModel.p90LatencySeconds.toFixed(2)}s`}
                    </span>
                  </div>
                  <div className="text-slate-400">
                    Throughput:{' '}
                    <span className="text-slate-200 font-bold">
                      {hoveredScatterModel.throughputReqPerSec.toFixed(1)} req/s
                    </span>
                  </div>
                  <div className="text-slate-400">
                    Brier:{' '}
                    <span className="text-slate-200 font-bold">
                      {hoveredScatterModel.brierScore.toFixed(4)}
                    </span>
                  </div>
                </div>
                <div className="mt-2 pt-1.5 border-t border-slate-800 text-[10px] text-cyan-400/80 font-medium">
                  Click point to inspect on Globe 🌐
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Filter & Search Bar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mt-8 mb-4">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search models, providers, backbones..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-900/80 border border-slate-800 text-xs text-slate-200 pl-9 pr-4 py-2 rounded-xl outline-none focus:border-cyan-500/60 transition-colors"
            />
          </div>

          <div className="flex items-center gap-2 overflow-x-auto pb-1 sm:pb-0">
            <span className="text-xs text-slate-400 flex items-center gap-1 whitespace-nowrap">
              <Filter className="w-3.5 h-3.5" /> Filter:
            </span>
            {[
              { id: 'all', label: 'All Models' },
              { id: 'decision', label: 'Decision Systems' },
              { id: 'reasoning', label: 'Reasoning LLMs' },
              { id: 'specialized', label: 'Specialized Geo' },
              { id: 'local_gguf', label: 'Local Edge' },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setFilterType(f.id)}
                className={`text-xs px-2.5 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap ${
                  filterType === f.id
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'bg-slate-900/70 text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Main Billboard Table */}
        <div className="bg-slate-900/70 rounded-2xl border border-slate-800/80 overflow-hidden shadow-2xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 font-bold uppercase tracking-wider text-[11px]">
                  <th
                    className="p-3.5 cursor-pointer hover:text-cyan-400 transition-colors"
                    onClick={() => handleSort('paretoRank')}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Pareto Rank</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="p-3.5">Model & Architecture</th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-cyan-400 transition-colors"
                    onClick={() => handleSort('balancedAccuracy')}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Balanced Acc</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-cyan-400 transition-colors"
                    onClick={() => handleSort('gridAccuracy')}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Raw Acc</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-cyan-400 transition-colors"
                    onClick={() => handleSort('brierScore')}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Brier Score</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-cyan-400 transition-colors"
                    onClick={() => handleSort('p90LatencySeconds')}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>P90 Latency</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="p-3.5">Median</th>
                  <th className="p-3.5">Throughput</th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-cyan-400 transition-colors"
                    onClick={() => handleSort('costPer1kQueriesUsd')}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Cost / 1k</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="p-3.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-medium">
                {filteredModels.map((m) => {
                  const isExpanded = expandedModelId === m.id;
                  const rankIcon =
                    m.paretoRank === 1
                      ? '🥇'
                      : m.paretoRank === 2
                      ? '🥈'
                      : m.paretoRank === 3
                      ? '🥉'
                      : `#${m.paretoRank}`;

                  return (
                    <React.Fragment key={m.id}>
                      <tr
                        className={`hover:bg-slate-800/40 transition-colors cursor-pointer ${
                          m.isParetoOptimal ? 'bg-emerald-950/10' : ''
                        }`}
                        onClick={() =>
                          setExpandedModelId(isExpanded ? null : m.id)
                        }
                      >
                        {/* Rank */}
                        <td className="p-3.5 font-bold whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            <span className="text-base">{rankIcon}</span>
                            {m.isParetoOptimal && (
                              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                Optimal
                              </span>
                            )}
                          </div>
                        </td>

                        {/* Model & Architecture */}
                        <td className="p-3.5 max-w-xs">
                          <div className="flex flex-col">
                            <span className="font-bold text-slate-100 flex items-center gap-1.5">
                              {m.name}
                              {m.parameters && (
                                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-300">
                                  {m.parameters}
                                </span>
                              )}
                            </span>
                            <span className="text-[11px] text-slate-400 mt-0.5">
                              {m.architecture}
                            </span>
                            {m.pathologyTag && (
                              <span
                                className={`text-[10px] font-mono px-1.5 py-0.5 rounded self-start mt-1 font-semibold ${
                                  m.pathologyTag.includes('Frontier') || m.pathologyTag.includes('Ceiling')
                                    ? 'bg-sky-500/20 text-sky-300'
                                    : m.pathologyTag.includes('Pareto')
                                    ? 'bg-emerald-500/20 text-emerald-300'
                                    : m.pathologyTag.includes('Tail') || m.pathologyTag.includes('Erased')
                                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                    : 'bg-amber-500/20 text-amber-300'
                                }`}
                              >
                                {m.pathologyTag}
                              </span>
                            )}
                          </div>
                        </td>

                        {/* Balanced Accuracy */}
                        <td className="p-3.5 whitespace-nowrap">
                          <div className="flex flex-col">
                            <span className="font-mono font-bold text-slate-100 text-sm">
                              {m.balancedAccuracy.toFixed(1)}%
                            </span>
                            <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1">
                              <div
                                className={`h-full rounded-full ${
                                  m.balancedAccuracy >= 85
                                    ? 'bg-emerald-400'
                                    : m.balancedAccuracy >= 70
                                    ? 'bg-cyan-400'
                                    : m.balancedAccuracy >= 55
                                    ? 'bg-amber-400'
                                    : 'bg-rose-500'
                                }`}
                                style={{ width: `${Math.min(100, m.balancedAccuracy)}%` }}
                              />
                            </div>
                          </div>
                        </td>

                        {/* Raw Accuracy */}
                        <td className="p-3.5 font-mono text-slate-300 whitespace-nowrap">
                          {m.gridAccuracy.toFixed(1)}%
                        </td>

                        {/* Brier Score */}
                        <td className="p-3.5 whitespace-nowrap">
                          <span
                            className={`font-mono font-semibold px-2 py-0.5 rounded text-[11px] ${
                              m.brierScore < 0.05
                                ? 'bg-emerald-500/20 text-emerald-300'
                                : m.brierScore < 0.15
                                ? 'bg-cyan-500/20 text-cyan-300'
                                : m.brierScore < 0.25
                                ? 'bg-amber-500/20 text-amber-300'
                                : 'bg-rose-500/20 text-rose-300'
                            }`}
                          >
                            {m.brierScore.toFixed(4)}
                          </span>
                        </td>

                        {/* P90 Latency */}
                        <td className="p-3.5 whitespace-nowrap">
                          <span
                            className={`font-mono font-bold ${
                              m.p90LatencySeconds > 5.0
                                ? 'text-rose-400 font-extrabold'
                                : m.p90LatencySeconds > 1.0
                                ? 'text-amber-400'
                                : 'text-emerald-400'
                            }`}
                          >
                            {m.p90LatencySeconds >= 1.0
                              ? `${m.p90LatencySeconds.toFixed(2)}s`
                              : `${(m.p90LatencySeconds * 1000).toFixed(0)}ms`}
                          </span>
                        </td>

                        {/* Median Latency */}
                        <td className="p-3.5 font-mono text-slate-300 whitespace-nowrap">
                          {m.medianLatencySeconds >= 1.0
                            ? `${m.medianLatencySeconds.toFixed(2)}s`
                            : `${(m.medianLatencySeconds * 1000).toFixed(0)}ms`}
                        </td>

                        {/* Throughput */}
                        <td className="p-3.5 font-mono text-cyan-400 whitespace-nowrap">
                          {m.throughputReqPerSec.toFixed(1)} req/s
                        </td>

                        {/* Cost */}
                        <td className="p-3.5 font-mono text-slate-300 whitespace-nowrap">
                          {m.costPer1kQueriesUsd === 0 ? (
                            <span className="text-emerald-400 font-bold">Free (Local)</span>
                          ) : (
                            `$${m.costPer1kQueriesUsd.toFixed(4)}`
                          )}
                        </td>

                        {/* Actions */}
                        <td className="p-3.5 text-right whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-end gap-1.5">
                            <button
                              onClick={() => handleInspectOnGlobe(m.runId)}
                              className="px-2.5 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500 hover:text-slate-950 font-bold text-[11px] transition-all flex items-center gap-1 border border-cyan-500/40"
                              title="Inspect this model on the interactive globe/map visualizer"
                            >
                              <Globe2 className="w-3.5 h-3.5" />
                              <span className="hidden sm:inline">Inspect</span>
                            </button>
                            <button
                              onClick={() =>
                                setExpandedModelId(isExpanded ? null : m.id)
                              }
                              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                            >
                              {isExpanded ? (
                                <ChevronUp className="w-4 h-4" />
                              ) : (
                                <ChevronDown className="w-4 h-4" />
                              )}
                            </button>
                          </div>
                        </td>
                      </tr>

                      {/* Expandable Model Details Drawer */}
                      {isExpanded && (
                        <tr className="bg-slate-950/90 border-b border-slate-800/80">
                          <td colSpan={10} className="p-4 sm:p-6">
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                              {/* Left: Qualitative Pathology & Engineering Rationale */}
                              <div className="lg:col-span-2 space-y-3">
                                <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                                  <AlertTriangle className="w-4 h-4 text-cyan-400" />
                                  Benchmark Diagnostics & Engineering Analysis
                                </h4>
                                <p className="text-xs text-slate-300 leading-relaxed bg-slate-900/80 p-3.5 rounded-xl border border-slate-800">
                                  {m.keyInsight}
                                </p>

                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                                  <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                                    <div className="text-[10px] text-slate-400 uppercase">Coastal Acc (≤100km)</div>
                                    <div className="text-sm font-bold text-slate-100 font-mono mt-0.5">
                                      {m.coastalAccuracy.toFixed(1)}%
                                    </div>
                                  </div>
                                  <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                                    <div className="text-[10px] text-slate-400 uppercase">Precision / Recall</div>
                                    <div className="text-sm font-bold text-slate-100 font-mono mt-0.5">
                                      {m.precision.toFixed(1)}% / {m.recall.toFixed(1)}%
                                    </div>
                                  </div>
                                  <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                                    <div className="text-[10px] text-slate-400 uppercase">F1 Score</div>
                                    <div className="text-sm font-bold text-slate-100 font-mono mt-0.5">
                                      {m.f1.toFixed(3)}
                                    </div>
                                  </div>
                                  {m.japanLandRecoveryPct !== undefined && (
                                    <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                                      <div className="text-[10px] text-slate-400 uppercase">Japan Land Recall</div>
                                      <div className="text-sm font-bold text-cyan-400 font-mono mt-0.5">
                                        {m.japanLandRecoveryPct.toFixed(1)}%
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* Right: Confusion Matrix Visual Grid */}
                              <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
                                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-2">
                                  Confusion Matrix (2,112 Points)
                                </h4>
                                <div className="grid grid-cols-2 gap-2 text-center text-xs">
                                  <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/30">
                                    <div className="text-[10px] text-emerald-400 font-bold uppercase">True Land (TP)</div>
                                    <div className="text-base font-extrabold text-emerald-300 font-mono mt-0.5">
                                      {m.tp.toLocaleString()}
                                    </div>
                                  </div>
                                  <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/30">
                                    <div className="text-[10px] text-rose-400 font-bold uppercase">False Land (FP)</div>
                                    <div className="text-base font-extrabold text-rose-300 font-mono mt-0.5">
                                      {m.fp.toLocaleString()}
                                    </div>
                                  </div>
                                  <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/30">
                                    <div className="text-[10px] text-rose-400 font-bold uppercase">False Water (FN)</div>
                                    <div className="text-base font-extrabold text-rose-300 font-mono mt-0.5">
                                      {m.fn.toLocaleString()}
                                    </div>
                                  </div>
                                  <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/30">
                                    <div className="text-[10px] text-emerald-400 font-bold uppercase">True Water (TN)</div>
                                    <div className="text-base font-extrabold text-emerald-300 font-mono mt-0.5">
                                      {m.tn.toLocaleString()}
                                    </div>
                                  </div>
                                </div>

                                <button
                                  onClick={() => handleInspectOnGlobe(m.runId)}
                                  className="w-full mt-3 py-2 px-3 rounded-lg bg-cyan-500 text-slate-950 hover:bg-cyan-400 font-extrabold text-xs transition-colors flex items-center justify-center gap-2"
                                >
                                  <Globe2 className="w-4 h-4" />
                                  Launch Full Spatial Reconstruction Overlay
                                </button>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer Methodology Notes */}
        <div className="mt-8 pt-6 border-t border-slate-800 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            Latent Earth Continuous Geoclassification Benchmark • Vercel Edge Compatible Build
          </div>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Natural Earth 10m Ground Truth
            </span>
            <span className="flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Area-Weighted Geodesic Distance
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
