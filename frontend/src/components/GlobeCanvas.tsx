import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3-geo';
import { PredictionPoint, ProjectionType, RenderMode } from '../types';

interface GlobeCanvasProps {
  points: PredictionPoint[];
  stepDegrees: number;
  projectionType: ProjectionType;
  renderMode: RenderMode;
  visibleCount: number;
  autoRotate: boolean;
  showCoastlineOverlay: boolean;
  coastlineGeoJson: any | null;
  onHoverPoint: (point: PredictionPoint | null, mousePos: { x: number; y: number } | null) => void;
}

// Color interpolation helpers
function getProbabilityColor(p: number | null): string {
  if (p === null) return '#1E293B';
  if (p < 0.2) return '#060E1E';
  if (p < 0.4) return '#1E3A8A';
  if (p < 0.5) return '#0284C7';
  if (p < 0.55) return '#06B6D4';
  if (p < 0.65) return '#D97706';
  if (p < 0.8) return '#22C55E';
  if (p < 0.92) return '#16A34A';
  return '#14532D'; // Deep forest green for continental heartlands
}

function getUncertaintyColor(u: number): string {
  if (u < 0.2) return '#5A5A65'; // Neutral medium gray (certain, visually recedes)
  if (u < 0.4) return '#735682'; // Gray-purple / muted purple
  if (u < 0.6) return '#9D34B5'; // Muted purple
  if (u < 0.75) return '#D92DA5'; // Magenta
  if (u < 0.9) return '#F97316'; // Orange
  return '#FDE047'; // Bright yellow (maximum ambiguity)
}

function getBinaryColor(isLand: boolean): string {
  return isLand ? '#10B981' : '#0B132B';
}


function getErrorColor(cat?: string): string {
  switch (cat) {
    case 'correct_land': return '#10B981'; // Green
    case 'correct_water': return '#0F2942'; // Dark Navy
    case 'false_land': return '#EF4444';    // Red
    case 'false_water': return '#F59E0B';   // Amber
    default: return '#334155';
  }
}

export const GlobeCanvas: React.FC<GlobeCanvasProps> = ({
  points,
  stepDegrees,
  projectionType,
  renderMode,
  visibleCount,
  autoRotate,
  showCoastlineOverlay,
  coastlineGeoJson,
  onHoverPoint,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rotation, setRotation] = useState<[number, number]>([-10, -20]);
  const [zoomScale, setZoomScale] = useState<number>(320);
  const [hoveredCell, setHoveredCell] = useState<PredictionPoint | null>(null);
  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef<[number, number]>([0, 0]);


  // Handle Dragging
  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    lastMousePosRef.current = [e.clientX, e.clientY];
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDraggingRef.current) {
      const dx = e.clientX - lastMousePosRef.current[0];
      const dy = e.clientY - lastMousePosRef.current[1];
      lastMousePosRef.current = [e.clientX, e.clientY];

      if (projectionType === 'orthographic') {
        const sensitivity = 0.35;
        setRotation(([lambda, phi]) => [
          (lambda + dx * sensitivity) % 360,
          Math.max(-85, Math.min(85, phi - dy * sensitivity)),
        ]);
      } else {
        // Flat map horizontal pan
        setRotation(([lambda, phi]) => [(lambda + dx * 0.5) % 360, phi]);
      }
    }

    // Hover inspection
    // We check in screen space or geo space
    checkHover(mouseX, mouseY, e.clientX, e.clientY);
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomScale((prev) => Math.max(120, Math.min(1800, prev * delta)));
  };

  // Idle rotation animation
  useEffect(() => {
    if (!autoRotate || projectionType !== 'orthographic') return;
    let animId: number;
    const loop = () => {
      if (!isDraggingRef.current) {
        setRotation(([lambda, phi]) => [(lambda + 0.25) % 360, phi]);
      }
      animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [autoRotate, projectionType]);

  // Main Canvas Render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Setup projection
    let projection: d3.GeoProjection;
    if (projectionType === 'orthographic') {
      projection = d3.geoOrthographic()
        .scale(zoomScale)
        .translate([width / 2, height / 2])
        .rotate([rotation[0], rotation[1], 0])
        .clipAngle(90);
    } else if (projectionType === 'equirectangular') {
      projection = d3.geoEquirectangular()
        .scale(zoomScale * 0.8)
        .translate([width / 2, height / 2])
        .rotate([rotation[0], 0, 0]);
    } else {
      projection = d3.geoMercator()
        .scale(zoomScale * 0.55)
        .translate([width / 2, height / 2])
        .rotate([rotation[0], 0, 0]);
    }

    const path = d3.geoPath(projection, ctx);

    // 1. Draw Globe Sphere / Background
    if (projectionType === 'orthographic') {
      // Glow atmosphere
      const grad = ctx.createRadialGradient(
        width / 2, height / 2, zoomScale * 0.95,
        width / 2, height / 2, zoomScale * 1.08
      );
      grad.addColorStop(0, 'rgba(6, 182, 212, 0.2)');
      grad.addColorStop(1, 'rgba(6, 182, 212, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(width / 2, height / 2, zoomScale * 1.08, 0, 2 * Math.PI);
      ctx.fill();

      // Deep space sphere body
      ctx.beginPath();
      path({ type: 'Sphere' } as any);
      ctx.fillStyle = '#060A13';
      ctx.fill();
      ctx.strokeStyle = '#1E293B';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else {
      // Flat map rectangle background
      ctx.fillStyle = '#080D1A';
      ctx.fillRect(0, 0, width, height);
    }

    // 2. Graticule (lat/lon lines)
    const graticule = d3.geoGraticule().step([30, 30]);
    ctx.beginPath();
    path(graticule() as any);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // 3. Grid cell dimensions
    const halfStep = stepDegrees / 2.0;

    // 3a. Render Unqueried Grid Cells (Neutral, very dark empty state distinct from predicted water)
    if (visibleCount < points.length) {
      ctx.beginPath();
      for (let i = visibleCount; i < points.length; i++) {
        const pt = points[i];
        const lat = pt.latitude;
        const lon = pt.longitude;

        const cellPolygon = {
          type: 'Polygon',
          coordinates: [[
            [lon - halfStep, lat + halfStep],
            [lon + halfStep, lat + halfStep],
            [lon + halfStep, lat - halfStep],
            [lon - halfStep, lat - halfStep],
            [lon - halfStep, lat + halfStep],
          ]],
        };
        path(cellPolygon as any);
      }
      ctx.fillStyle = '#080C14'; // Neutral dark charcoal empty state
      ctx.fill();
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.35)'; // Faint slate grid wireframe
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // 3b. Render Queried Model Decision Grid Cells
    const activePoints = points.slice(0, visibleCount);
    ctx.globalAlpha = 1.0;

    for (let i = 0; i < activePoints.length; i++) {
      const pt = activePoints[i];
      const lat = pt.latitude;
      const lon = pt.longitude;


      // Color selection
      let fillColor = '#1E293B';
      if (renderMode === 'probability') {
        fillColor = getProbabilityColor(pt.probability_land);
      } else if (renderMode === 'binary') {
        fillColor = getBinaryColor(pt.is_land);
      } else if (renderMode === 'uncertainty') {
        fillColor = getUncertaintyColor(pt.uncertainty);
      } else if (renderMode === 'ground_truth') {
        fillColor = pt.actual_land ? '#10B981' : '#0F2942';
      } else if (renderMode === 'error_map') {
        fillColor = getErrorColor(pt.error_category);
      }

      // Geo polygon for exact cell coverage
      const cellPolygon = {
        type: 'Polygon',
        coordinates: [[
          [lon - halfStep, lat + halfStep],
          [lon + halfStep, lat + halfStep],
          [lon + halfStep, lat - halfStep],
          [lon - halfStep, lat - halfStep],
          [lon - halfStep, lat + halfStep],
        ]],
      };

      ctx.beginPath();
      path(cellPolygon as any);
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    // 4. Optional Natural Earth Coastline Reference Outline (drawn on top of cells for high contrast)
    if (showCoastlineOverlay && coastlineGeoJson) {
      ctx.beginPath();
      path(coastlineGeoJson);
      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }


    // 5. Highlight hovered cell outline for tactile inspection
    if (hoveredCell) {
      const hLat = hoveredCell.latitude;
      const hLon = hoveredCell.longitude;
      const highlightPoly = {
        type: 'Polygon',
        coordinates: [[
          [hLon - halfStep, hLat + halfStep],
          [hLon + halfStep, hLat + halfStep],
          [hLon + halfStep, hLat - halfStep],
          [hLon - halfStep, hLat - halfStep],
          [hLon - halfStep, hLat + halfStep],
        ]],
      };
      ctx.beginPath();
      path(highlightPoly as any);
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2.0;
      ctx.stroke();
    }

    // 6. Pulsing marker for latest active sampling point
    if (activePoints.length > 0 && visibleCount < points.length) {
      const latest = activePoints[activePoints.length - 1];
      const screenCoord = projection([latest.longitude, latest.latitude]);
      if (screenCoord && screenCoord[0] >= 0 && screenCoord[0] <= width) {
        ctx.beginPath();
        ctx.arc(screenCoord[0], screenCoord[1], 8, 0, 2 * Math.PI);
        ctx.strokeStyle = '#38BDF8';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(screenCoord[0], screenCoord[1], 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();
      }
    }
  }, [
    points,
    stepDegrees,
    projectionType,
    renderMode,
    visibleCount,
    rotation,
    zoomScale,
    showCoastlineOverlay,
    coastlineGeoJson,
    hoveredCell,
  ]);


  // Hover detection
  const checkHover = useCallback((mouseX: number, mouseY: number, clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let projection: d3.GeoProjection;
    const width = canvas.width;
    const height = canvas.height;

    if (projectionType === 'orthographic') {
      projection = d3.geoOrthographic()
        .scale(zoomScale)
        .translate([width / 2, height / 2])
        .rotate([rotation[0], rotation[1], 0])
        .clipAngle(90);
    } else if (projectionType === 'equirectangular') {
      projection = d3.geoEquirectangular()
        .scale(zoomScale * 0.8)
        .translate([width / 2, height / 2])
        .rotate([rotation[0], 0, 0]);
    } else {
      projection = d3.geoMercator()
        .scale(zoomScale * 0.55)
        .translate([width / 2, height / 2])
        .rotate([rotation[0], 0, 0]);
    }

    const inverted = projection.invert?.([mouseX, mouseY]);
    if (!inverted || isNaN(inverted[0]) || isNaN(inverted[1])) {
      onHoverPoint(null, null);
      return;
    }

    const [lon, lat] = inverted;
    const halfStep = stepDegrees / 2.0;

    // Search for closest cell in currently active visible points
    const activePoints = points.slice(0, visibleCount);
    let matched: PredictionPoint | null = null;
    let minDistance = Infinity;

    for (let i = 0; i < activePoints.length; i++) {
      const pt = activePoints[i];
      const dLat = Math.abs(pt.latitude - lat);
      const dLon = Math.abs(pt.longitude - lon);
      if (dLat <= halfStep && (dLon <= halfStep || Math.abs(dLon - 360) <= halfStep)) {
        const dist = dLat * dLat + dLon * dLon;
        if (dist < minDistance) {
          minDistance = dist;
          matched = pt;
        }
      }
    }

    if (matched) {
      setHoveredCell(matched);
      onHoverPoint(matched, { x: clientX, y: clientY });
    } else {
      setHoveredCell(null);
      onHoverPoint(null, null);
    }
  }, [points, visibleCount, stepDegrees, projectionType, zoomScale, rotation, onHoverPoint]);

  // Resize canvas to match display size
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        isDraggingRef.current = false;
        setHoveredCell(null);
        onHoverPoint(null, null);
      }}
      onWheel={handleWheel}
      className="w-full h-full block cursor-grab active:cursor-grabbing"
    />
  );
};

