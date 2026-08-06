import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { ClassicalGhost } from "../api/types";
import { formatSeconds, ghostAngle, ghostRadius, simSeconds, tauFromWall } from "../lib/classical";
import { GHOST_DISPLAY_WINDINGS } from "../lib/liberties";

/** Wall-clock seconds one collapse loop is stretched over, the disclosed slow-mo window. */
export const GHOST_LOOP_SECONDS = 5;

const GHOST_COLOR = "#8be9fd"; // classical cyan, distinct from the |psi|^2 LUTs
const INNER_RING_COLOR = "#6272a4";
const RING_SEGMENTS = 128;
const SPIRAL_SEGMENTS = 512;
const TAU_MAX = 0.999; // stop just short of r=0 so the line stays finite

/** Circle of the given radius in the scene xz-plane (the physics equatorial plane). */
function xzCircleGeometry(radius: number): THREE.BufferGeometry {
  const pts = new Float32Array(RING_SEGMENTS * 3);
  for (let i = 0; i < RING_SEGMENTS; i++) {
    const a = (2 * Math.PI * i) / RING_SEGMENTS;
    pts[3 * i] = radius * Math.cos(a);
    pts[3 * i + 1] = 0;
    pts[3 * i + 2] = radius * Math.sin(a);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pts, 3));
  return g;
}

/**
 * The collapse spiral, sampled uniformly in swept angle (not in tau, the
 * winding rate diverges as tau -> 1, so uniform-tau sampling starves the inner
 * turns). With u = 1 - sqrt(1 - tau), theta is linear in u.
 */
function spiralGeometry(r0: number, windings: number): THREE.BufferGeometry {
  const uMax = 1 - Math.sqrt(1 - TAU_MAX);
  const pts = new Float32Array(SPIRAL_SEGMENTS * 3);
  for (let i = 0; i < SPIRAL_SEGMENTS; i++) {
    const u = (uMax * i) / (SPIRAL_SEGMENTS - 1);
    const tau = 1 - (1 - u) * (1 - u);
    const r = ghostRadius(tau, r0);
    const a = ghostAngle(tau, windings);
    pts[3 * i] = r * Math.cos(a);
    pts[3 * i + 1] = 0;
    pts[3 * i + 2] = r * Math.sin(a);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pts, 3));
  return g;
}

/**
 * Three.js overlay rendered inside the CloudView <Canvas>: Bohr rings, the
 * collapse spiral, and one animated ghost point. Radii are raw Bohr, exactly
 * the units the point cloud is drawn in; `distance` (the camera distance) only
 * sizes the ghost point, mirroring the nucleus-marker pattern.
 *
 * The animated point is driven through a three.js ref in useFrame, React
 * never re-renders per frame. `tauRef` shares the live loop phase with the
 * HUD clock outside the Canvas, which polls it at ~10 Hz.
 */
export function GhostOverlay({
  ghost,
  distance,
  tauRef,
}: {
  ghost: ClassicalGhost;
  distance: number;
  tauRef?: { current: number };
}) {
  const r0 = ghost.r0_bohr.value;
  // VISUAL LIBERTY (disclosed via CLASSICAL_SLOWMO): the drawn winding count is
  // capped, the honest count (~1e5) would alias into noise at any frame rate.
  const windings = Math.min(ghost.orbit_count.value, GHOST_DISPLAY_WINDINGS);
  const pointRef = useRef<THREE.Mesh>(null);

  const ringGeoms = useMemo(
    () => ghost.orbits.map((o) => xzCircleGeometry(o.radius_bohr.value)),
    [ghost.orbits],
  );
  useEffect(
    () => () => {
      for (const g of ringGeoms) g.dispose();
    },
    [ringGeoms],
  );

  const spiral = useMemo(
    () =>
      new THREE.Line(
        spiralGeometry(r0, windings),
        new THREE.LineBasicMaterial({ color: GHOST_COLOR, transparent: true, opacity: 0.45 }),
      ),
    [r0, windings],
  );
  useEffect(
    () => () => {
      spiral.geometry.dispose();
      (spiral.material as THREE.Material).dispose();
    },
    [spiral],
  );

  useFrame((state) => {
    const tau = tauFromWall(state.clock.elapsedTime, GHOST_LOOP_SECONDS);
    if (tauRef) tauRef.current = tau;
    const p = pointRef.current;
    if (!p) return;
    const r = ghostRadius(tau, r0);
    const a = ghostAngle(tau, windings);
    p.position.set(r * Math.cos(a), 0, r * Math.sin(a));
  });

  if (!ghost.orbits.length) return null;
  const last = ghost.orbits.length - 1;
  return (
    <group>
      {ghost.orbits.map((o, i) => (
        <lineLoop key={o.n} geometry={ringGeoms[i]}>
          <lineBasicMaterial
            color={i === last ? GHOST_COLOR : INNER_RING_COLOR}
            transparent
            opacity={i === last ? 0.9 : 0.35}
          />
        </lineLoop>
      ))}
      <primitive object={spiral} />
      <mesh ref={pointRef}>
        <sphereGeometry args={[distance / 160, 16, 8]} />
        <meshBasicMaterial color={GHOST_COLOR} />
      </mesh>
    </group>
  );
}

/**
 * Live simulated-time clock for the HUD (outside the Canvas). Polls the shared
 * tau ref at 10 Hz with its own interval, a tiny local re-render, never tied
 * to the animation frame rate.
 */
export function GhostClock({
  tauRef,
  collapseSeconds,
}: {
  tauRef: { current: number };
  collapseSeconds: number;
}) {
  const [tau, setTau] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setTau(tauRef.current), 100);
    return () => window.clearInterval(id);
  }, [tauRef]);
  return <span className="ghost-clock">t = {formatSeconds(simSeconds(tau, collapseSeconds))}</span>;
}
