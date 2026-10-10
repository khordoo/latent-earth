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
  recenterTrigger?: number;
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
  recenterTrigger,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rotation, setRotation] = useState<[number, number]>([-10, -20]);
  const [panOffset, setPanOffset] = useState<[number, number]>([0, 0]);
  const [zoomScale, setZoomScale] = useState<number>(320);
  const [hoveredCell, setHoveredCell] = useState<PredictionPoint | null>(null);
  const isDraggingRef = useRef(false);
  const [isDraggingState, setIsDraggingState] = useState(false);
  const lastMousePosRef = useRef<[number, number]>([0, 0]);

  // Centroid computation for active points
  const centroid = React.useMemo(() => {
    if (!points || points.length === 0) return { lat: 0, lon: 0, isRegional: false };
    let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
    const sampleLimit = Math.min(points.length, 1000);
    for (let i = 0; i < sampleLimit; i++) {
      const p = points[i];
      if (p.latitude < minLat) minLat = p.latitude;
      if (p.latitude > maxLat) maxLat = p.latitude;
      if (p.longitude < minLon) minLon = p.longitude;
      if (p.longitude > maxLon) maxLon = p.longitude;
    }
    const isRegional = (maxLat - minLat < 60) && (maxLon - minLon < 90);
    return {
      lat: (minLat + maxLat) / 2,
      lon: (minLon + maxLon) / 2,
      isRegional,
    };
  }, [points]);

  const recenterView = useCallback(() => {
    if (centroid.isRegional) {
      // Regional dataset focus (e.g. Japan & Korea)
      if (projectionType === 'orthographic') {
        setRotation([-centroid.lon, -centroid.lat]);
        setPanOffset([0, 0]);
        setZoomScale(520);
      } else {
        setRotation([-centroid.lon, 0]);
        const currentScale = (projectionType === 'equirectangular' ? 320 * 0.8 : 320 * 0.55);
        const panY = centroid.lat * (currentScale * Math.PI / 180);
        setPanOffset([0, panY]);
        setZoomScale(450);
      }
    } else {
      // Global Earth full planetary view
      setRotation([-10, -20]);
      setPanOffset([0, 0]);
      setZoomScale(320);
    }
  }, [centroid, projectionType]);

  // Recenter whenever recenterTrigger changes
  useEffect(() => {
    if (recenterTrigger !== undefined && recenterTrigger > 0) {
      recenterView();
    }
  }, [recenterTrigger, recenterView]);

  // Recenter automatically when switching datasets
  const lastPointsLengthRef = useRef(points.length);
  useEffect(() => {
    if (points.length > 0 && points.length !== lastPointsLengthRef.current) {
      lastPointsLengthRef.current = points.length;
      recenterView();
    }
  }, [points.length, recenterView]);

  // Handle Dragging
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0 && e.button !== 1) return;
    isDraggingRef.current = true;
    setIsDraggingState(true);
    lastMousePosRef.current = [e.clientX, e.clientY];
    setHoveredCell(null);
    onHoverPoint(null, null);
  };

  // Window-level mouse listeners for continuous click-and-hold dragging
  useEffect(() => {
    const handleWindowMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const dx = e.clientX - lastMousePosRef.current[0];
      const dy = e.clientY - lastMousePosRef.current[1];
      lastMousePosRef.current = [e.clientX, e.clientY];

      if (projectionType === 'orthographic') {
        if (e.shiftKey || e.buttons === 2) {
          // Shift+Drag or Right-Click Drag: Pan the globe across canvas
          setPanOffset(([px, py]) => [px + dx, py + dy]);
        } else {
          // Left-Drag: Rotate the sphere (sensitivity scales with zoom level)
          const sensitivity = 0.35 * Math.min(1.0, 320 / Math.max(150, zoomScale));
          setRotation(([lambda, phi]) => [
            (lambda + dx * sensitivity) % 360,
            Math.max(-85, Math.min(85, phi - dy * sensitivity)),
          ]);
        }
      } else {
        // Flat 2D maps (Equirectangular & Mercator):
        // Click and hold smoothly drags/pans the entire map in X and Y
        setPanOffset(([px, py]) => [px + dx, py + dy]);
      }

      setHoveredCell(null);
      onHoverPoint(null, null);
    };

    const handleWindowMouseUp = () => {
      if (isDraggingRef.current) {
        isDraggingRef.current = false;
        setIsDraggingState(false);
      }
    };

    window.addEventListener('mousemove', handleWindowMouseMove);
    window.addEventListener('mouseup', handleWindowMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleWindowMouseMove);
      window.removeEventListener('mouseup', handleWindowMouseUp);
    };
  }, [projectionType, zoomScale, onHoverPoint]);

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDraggingRef.current) {
      // Dragging handled by window listener
      return;
    }

    // Hover inspection only when NOT dragging
    checkHover(mouseX, mouseY, e.clientX, e.clientY);
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
    setIsDraggingState(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomScale((prev) => Math.max(120, Math.min(2400, prev * delta)));
  };

  const handleDoubleClick = () => {
    recenterView();
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

    const centerX = width / 2 + panOffset[0];
    const centerY = height / 2 + panOffset[1];

    // Setup projection
    let projection: d3.GeoProjection;
    if (projectionType === 'orthographic') {
      projection = d3.geoOrthographic()
        .scale(zoomScale)
        .translate([centerX, centerY])
        .rotate([rotation[0], rotation[1], 0])
        .clipAngle(90 - 1e-4);
    } else if (projectionType === 'equirectangular') {
      projection = d3.geoEquirectangular()
        .scale(zoomScale * 0.8)
        .translate([centerX, centerY])
        .rotate([rotation[0], 0, 0]);
    } else {
      projection = d3.geoMercator()
        .scale(zoomScale * 0.55)
        .translate([centerX, centerY])
        .rotate([rotation[0], 0, 0]);
    }

    const path = d3.geoPath(projection, ctx);

    // 1. Draw Globe Sphere / Background
    if (projectionType === 'orthographic') {
      // Glow atmosphere
      const grad = ctx.createRadialGradient(
        centerX, centerY, zoomScale * 0.95,
        centerX, centerY, zoomScale * 1.08
      );
      grad.addColorStop(0, 'rgba(6, 182, 212, 0.2)');
      grad.addColorStop(1, 'rgba(6, 182, 212, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(centerX, centerY, zoomScale * 1.08, 0, 2 * Math.PI);
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
    panOffset,
    zoomScale,
    showCoastlineOverlay,
    coastlineGeoJson,
    hoveredCell,
  ]);


  // Hover detection with strict spatial boundary checks
  const checkHover = useCallback((mouseX: number, mouseY: number, clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // 1. Suppress hover completely while dragging/panning
    if (isDraggingRef.current) {
      setHoveredCell(null);
      onHoverPoint(null, null);
      return;
    }

    let projection: d3.GeoProjection;
    let currentScale: number;
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + panOffset[0];
    const centerY = height / 2 + panOffset[1];

    if (projectionType === 'orthographic') {
      // 2. In orthographic mode, check distance to globe center:
      // If outside the globe sphere disk, immediately clear hover!
      const dx = mouseX - centerX;
      const dy = mouseY - centerY;
      if (Math.hypot(dx, dy) > zoomScale) {
        setHoveredCell(null);
        onHoverPoint(null, null);
        return;
      }

      currentScale = zoomScale;
      projection = d3.geoOrthographic()
        .scale(zoomScale)
        .translate([centerX, centerY])
        .rotate([rotation[0], rotation[1], 0])
        .clipAngle(90 - 1e-4);
    } else if (projectionType === 'equirectangular') {
      currentScale = zoomScale * 0.8;
      projection = d3.geoEquirectangular()
        .scale(currentScale)
        .translate([centerX, centerY])
        .rotate([rotation[0], 0, 0]);
    } else {
      currentScale = zoomScale * 0.55;
      projection = d3.geoMercator()
        .scale(currentScale)
        .translate([centerX, centerY])
        .rotate([rotation[0], 0, 0]);
    }

    if (projectionType !== 'orthographic') {
      // 3. In flat 2D maps (equirectangular and mercator), strictly check if mouse
      // is outside the left or right edges of the drawn map rectangle:
      const mapHalfWidth = Math.PI * currentScale;
      const xMin = centerX - mapHalfWidth;
      const xMax = centerX + mapHalfWidth;

      const mapHalfHeight = (projectionType === 'equirectangular' ? Math.PI / 2 : Math.PI) * currentScale;
      const yMin = centerY - mapHalfHeight;
      const yMax = centerY + mapHalfHeight;

      if (mouseX < xMin || mouseX > xMax || mouseY < yMin || mouseY > yMax) {
        setHoveredCell(null);
        onHoverPoint(null, null);
        return;
      }
    }

    const inverted = projection.invert?.([mouseX, mouseY]);
    if (!inverted || isNaN(inverted[0]) || isNaN(inverted[1])) {
      setHoveredCell(null);
      onHoverPoint(null, null);
      return;
    }

    if (projectionType !== 'orthographic') {
      if (inverted[1] < -85 || inverted[1] > 85) {
        setHoveredCell(null);
        onHoverPoint(null, null);
        return;
      }
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
      let dLon = Math.abs(pt.longitude - lon);
      if (dLon > 180) dLon = 360 - dLon;

      if (dLat <= halfStep && dLon <= halfStep) {
        const dist = dLat * dLat + dLon * dLon;
        if (dist < minDistance) {
          minDistance = dist;
          matched = pt;
        }
      }
    }

    if (matched) {
      // 4. Verify candidate point's screen projection is visible and near the cursor
      const screenCoord = projection([matched.longitude, matched.latitude]);
      if (!screenCoord) {
        // Point is clipped / occluded on back side of globe
        setHoveredCell(null);
        onHoverPoint(null, null);
        return;
      }

      // Max allowable screen distance between mouse and cell center
      const cellRadiusPx = Math.max(8, Math.min(80, (stepDegrees * (Math.PI / 180)) * currentScale * 1.25));
      const screenDist = Math.hypot(screenCoord[0] - mouseX, screenCoord[1] - mouseY);
      if (screenDist > cellRadiusPx) {
        setHoveredCell(null);
        onHoverPoint(null, null);
        return;
      }

      setHoveredCell(matched);
      onHoverPoint(matched, { x: clientX, y: clientY });
    } else {
      setHoveredCell(null);
      onHoverPoint(null, null);
    }
  }, [points, visibleCount, stepDegrees, projectionType, zoomScale, rotation, panOffset, onHoverPoint]);

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
        if (!isDraggingRef.current) {
          setHoveredCell(null);
          onHoverPoint(null, null);
        }
      }}
      onWheel={handleWheel}
      onDoubleClick={handleDoubleClick}
      className={`w-full h-full block select-none ${
        isDraggingState
          ? 'cursor-grabbing'
          : 'cursor-grab'
      }`}
    />
  );
};

