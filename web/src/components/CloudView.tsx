import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import type * as THREE from "three";
import { formatSeconds, slowMotionFactor } from "../lib/classical";
import { buildCloudColors } from "../lib/cloudColors";
import {
  buildSurfaceColors,
  componentsCaption,
  enclosedCaption,
  surfaceExtent,
} from "../lib/isoSurface";
import {
  CLASSICAL_SLOWMO,
  ISOSURFACE_LIBERTY,
  NUCLEUS_MARKER_LIBERTY,
  RENDER_LIBERTIES,
  formatErrorScale,
} from "../lib/liberties";
import { nucleusCaption, nucleusSphere } from "../lib/nucleus";
import { systemKind } from "../lib/systemKind";
import { HF_ORBITAL_CAPTION } from "../lib/hfModel";
import { ISO_FRACTIONS, useAppStore } from "../state/store";
import { AxisTriad, axisArmLength } from "./AxisTriad";
import { Badge } from "./Badge";
import { GhostClock, GhostOverlay } from "./GhostOverlay";
import { IsoSurface } from "./IsoSurface";
import { Legend } from "./Legend";
import { PointCloud } from "./PointCloud";

/** Axis arm length, at a precision the number can support. */
function formatArm(length: number): string {
  return length.toFixed(length >= 100 ? 0 : length >= 10 ? 1 : 2);
}

function CameraRig({ distance }: { distance: number }) {
  const camera = useThree((s) => s.camera as THREE.PerspectiveCamera);
  useEffect(() => {
    camera.position.set(distance * 0.7, distance * 0.45, distance);
    camera.near = distance / 100;
    camera.far = distance * 100;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, distance]);
  return null;
}

function FpsMeter() {
  const setFps = useAppStore((s) => s.setFps);
  const acc = useRef({ frames: 0, t0: 0 });
  useFrame(() => {
    const a = acc.current;
    if (a.t0 === 0) a.t0 = performance.now();
    a.frames += 1;
    const now = performance.now();
    if (now - a.t0 >= 500) {
      setFps(Math.round((a.frames * 1000) / (now - a.t0)));
      a.frames = 0;
      a.t0 = now;
    }
  });
  return null;
}

export function CloudView() {
  const {
    n,
    positions,
    density,
    phase,
    colorMode,
    stateInfo,
    nucleusMode,
    ghost,
    classicalGhost,
    classicalStatus,
    setGhost,
    loadClassical,
    system,
    systems,
    surfaceMode,
    setSurfaceMode,
    isoFraction,
    setIsoFraction,
    iso,
    isoStatus,
    isoProgress,
    loadIso,
    model,
    meta,
  } = useAppStore();
  // The ghost is a Kepler orbit, which exists because the field is exactly
  // 1/r. A screened atom's whole content is that its field is not, so there is
  // no ghost to draw and /api/classical says so with a 422. Three states, not
  // two: offer the toggle when we know the system is hydrogenic, say why not
  // when we know it is screened, and show neither before the systems table
  // arrives, so hydrogen does not flash the screened note on first render.
  const kind = systemKind(systems, system);
  // Deep-link (?ghost=1) sets `ghost` in initial state without going through
  // setGhost, and changing n/system resets the ghost data to idle while the
  // toggle stays on. Either way, fetch when the overlay is on but data is idle.
  // Hiding the toggle is not enough on its own: the deep link reaches `ghost`
  // without ever touching it, and this effect is what it reaches.
  useEffect(() => {
    if (kind === "hydrogenic" && ghost && classicalStatus === "idle") void loadClassical();
  }, [kind, ghost, classicalStatus, loadClassical]);
  // Live loop phase shared between the in-Canvas animation (writes each frame)
  // and the HUD clock (polls at 10 Hz), no per-frame React renders.
  const ghostTauRef = useRef(0);
  const colors = useMemo(
    () => buildCloudColors(colorMode, density, phase),
    [colorMode, density, phase],
  );
  const showSurface = surfaceMode !== "cloud";
  const showCloud = surfaceMode !== "surface";
  // The surface is fetched when it is asked to be shown, exactly like the
  // classical ghost: a deep link can arrive with ?surf=surface set without ever
  // passing through the toggle, and this is what it reaches.
  useEffect(() => {
    if (showSurface && isoStatus === "idle") void loadIso();
  }, [showSurface, isoStatus, loadIso]);
  const surfaceColors = useMemo(
    () => (iso ? buildSurfaceColors(iso.phase) : null),
    [iso],
  );
  const meanRadiusDistance = stateInfo
    ? Math.max(6 * stateInfo.mean_radius.value, 1e-3)
    : 5 * n * n + 3;
  // In surface-only mode the cloud is not there to be framed, and a contour is
  // smaller than the cloud around it, so framing on <r> would leave it small in
  // the middle of an empty canvas.
  const distance =
    surfaceMode === "surface" && iso
      ? Math.max(2.6 * surfaceExtent(iso.vertices), 1e-3)
      : meanRadiusDistance;
  const sysInfo = stateInfo?.system ?? null;
  const nucleus = nucleusSphere(
    nucleusMode,
    sysInfo?.nuclear_radius?.value ?? null,
    distance,
  );
  const caption = nucleusCaption(nucleusMode, sysInfo, nucleus);
  return (
    <div className="canvas-wrap">
      <Canvas camera={{ fov: 50 }}>
        {/* Matches --stage in index.css. The 3-D canvas paints its own opaque
            background, so this is the one place the stage colour is not read
            from the stylesheet and the two have to be kept in step by hand. */}
        <color attach="background" args={["#080c0e"]} />
        <CameraRig distance={distance} />
        <FpsMeter />
        <AxisTriad distance={distance} />
        {showCloud && positions && (
          <PointCloud
            positions={positions}
            pointSize={distance / 350}
            colors={colors}
          />
        )}
        {showSurface && iso && surfaceColors && (
          <>
            {/* A lit material needs light, and the cloud never did. Both are
                presentation and both ride on ISOSURFACE_LIBERTY. */}
            <ambientLight intensity={0.65} />
            <directionalLight position={[1, 1, 1]} intensity={1.1} />
            <IsoSurface
              vertices={iso.vertices}
              triangles={iso.triangles}
              colors={surfaceColors}
            />
          </>
        )}
        {ghost && classicalGhost && (
          <GhostOverlay ghost={classicalGhost} distance={distance} tauRef={ghostTauRef} />
        )}
        {nucleus && (
          <mesh>
            <sphereGeometry args={[nucleus.radius, 32, 16]} />
            <meshBasicMaterial
              color={nucleus.kind === "marker" ? "#ffb86b" : "#ffd9a0"}
            />
          </mesh>
        )}
        <OrbitControls />
      </Canvas>
      {/* What the stage is showing, in its top-left corner. The design also put
          a random seed and a "1px = N pm" scale here; the sampler does not
          report a seed and the camera is a live orbit with no fixed pixel
          scale, so both would have been decoration reading as instrumentation. */}
      {meta && (
        <div className="stage-caption">
          |ψ|² Monte-Carlo · {meta.count.toLocaleString()} draws
        </div>
      )}
      {!positions && surfaceMode === "cloud" && (
        <p className="hint">Choose a state and press Sample</p>
      )}
      <div className="canvas-overlay">
        <Badge provenance={RENDER_LIBERTIES} />
        {nucleus?.kind === "marker" && <Badge provenance={NUCLEUS_MARKER_LIBERTY} />}
        {caption && <span className="nucleus-caption">{caption}</span>}
        {/* The triad's scale. Drawn here rather than in 3-D so it cannot land
            on top of the z tip, and so the arm length is stated in the data's
            own units next to the rest of the disclosures. */}
        <span className="ghost-readout">
          axes ±{formatArm(axisArmLength(distance))} a{"₀"} · z is the
          quantization axis
        </span>
        <Legend mode={colorMode} />
        <div className="surface-controls" data-tour="surface-controls">
          <label>
            Draw
            <select
              value={surfaceMode}
              onChange={(e) =>
                setSurfaceMode(e.target.value as typeof surfaceMode)
              }
            >
              <option value="cloud">point cloud</option>
              <option value="surface">enclosing surface</option>
              <option value="both">both</option>
            </select>
          </label>
          {showSurface && (
            <label>
              Enclosing
              <select
                value={isoFraction}
                onChange={(e) => setIsoFraction(Number(e.target.value))}
              >
                {ISO_FRACTIONS.map((f) => (
                  <option key={f} value={f}>
                    {(f * 100).toFixed(0)}%
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        {showSurface && isoStatus === "sampling" && (
          <span className="ghost-readout">
            solving the level… {Math.round(isoProgress * 100)}%
          </span>
        )}
        {/* Deliberately not .ghost-hud: that box is bordered counterfactual
            pink, and a contour of the real |psi|^2 is not a counterfactual. */}
        {showSurface && iso && (
          <div className="surface-hud">
            <div className="ghost-readout">
              {enclosedCaption(iso.meta)}{" "}
              <Badge provenance={iso.meta.enclosed_fraction.provenance} />
            </div>
            <div className="ghost-readout">
              |psi|^2 = {iso.meta.level.value.toExponential(3)} bohr^-3 on a{" "}
              {iso.meta.resolution}^3 grid, {componentsCaption(iso.meta)}
            </div>
            <div className="ghost-readout">
              {iso.meta.escaped_fraction.value.toExponential(1)} of the electron is
              outside the box entirely
            </div>
            {/* Both error bars, because they measure different claims and the
                fraction one is nearly blind: the level hardly moves under a
                halved grid, so the fraction converges long before the shape
                does. Showing only "fraction ± 0" beside a surface that is half
                a percent off in size would read as exactness. */}
            {iso.meta.provenance.error_estimate !== null && (
              <div className="ghost-readout">
                halving the grid moves the enclosed fraction by{" "}
                {formatErrorScale(iso.meta.provenance.error_estimate)}
                {iso.meta.mesh_volume.provenance.error_estimate !== null && (
                  <>
                    {" "}
                    and the volume by{" "}
                    {(
                      (100 * iso.meta.mesh_volume.provenance.error_estimate) /
                      iso.meta.mesh_volume.value
                    ).toFixed(2)}
                    %
                  </>
                )}
              </div>
            )}
            <Badge provenance={ISOSURFACE_LIBERTY} />
          </div>
        )}
        {model === "hf" && <span className="orbital-claim">{HF_ORBITAL_CAPTION}</span>}
        {kind === "hydrogenic" && (
          <label className="ghost-toggle">
            <input
              type="checkbox"
              checked={ghost}
              onChange={(e) => setGhost(e.target.checked)}
            />
            Classical ghost
          </label>
        )}
        {kind === "screened" && (
          <span className="ghost-readout">
            No classical ghost here: the Kepler orbit needs a 1/r field, and the
            screening is the part this model adds.
          </span>
        )}
        {ghost && classicalStatus === "sampling" && (
          <span className="ghost-readout">loading classical orbits…</span>
        )}
        {ghost && classicalGhost && (
          <div className="ghost-hud">
            <div className="ghost-banner">
              Counterfactual: a classical electron would spiral in; real atoms do not
            </div>
            <GhostClock
              tauRef={ghostTauRef}
              collapseSeconds={classicalGhost.collapse_time_s.value}
            />
            <div className="ghost-readout">
              collapse in {formatSeconds(classicalGhost.collapse_time_s.value)}{" "}
              <Badge provenance={classicalGhost.collapse_time_s.provenance} />
            </div>
            <div className="ghost-readout">
              {Math.round(classicalGhost.orbit_count.value).toLocaleString()} orbits before
              collapse <Badge provenance={classicalGhost.orbit_count.provenance} />
            </div>
            <div className="ghost-readout">
              shown at ~{slowMotionFactor(classicalGhost.collapse_time_s.value).toExponential(1)}×
              slow motion <Badge provenance={CLASSICAL_SLOWMO} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
