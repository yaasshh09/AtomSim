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

/** The axis arm length, printed at a precision the number can support. */
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
  // My ghost is a Kepler orbit, which exists because the field is exactly 1/r.
  // A screened atom's whole content is that its field is not, so I have no
  // ghost to draw and /api/classical says so with a 422. I keep three states,
  // not two: I offer the toggle when I know the system is hydrogenic, say why
  // not when I know it is screened, and show neither before my systems table
  // arrives, so hydrogen does not flash the screened note on first render.
  const kind = systemKind(systems, system);
  // A deep link (?ghost=1) sets `ghost` in my initial state without going
  // through setGhost, and changing n or the system resets my ghost data to
  // idle while the toggle stays on. Either way, I fetch when the overlay is on
  // but the data is idle. Hiding the toggle is not enough on its own: the deep
  // link reaches `ghost` without ever touching it, and this effect is what it
  // reaches.
  useEffect(() => {
    if (kind === "hydrogenic" && ghost && classicalStatus === "idle") void loadClassical();
  }, [kind, ghost, classicalStatus, loadClassical]);
  // The live loop phase I share between my in-Canvas animation (which writes
  // each frame) and my HUD clock (which polls at 10 Hz), with no per-frame
  // React renders.
  const ghostTauRef = useRef(0);
  const colors = useMemo(
    () => buildCloudColors(colorMode, density, phase),
    [colorMode, density, phase],
  );
  const showSurface = surfaceMode !== "cloud";
  const showCloud = surfaceMode !== "surface";
  // I fetch the surface when someone asks me to show it, exactly like the
  // classical ghost: a deep link can arrive with ?surf=surface set without
  // ever passing through the toggle, and this is what it reaches.
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
  // In surface-only mode I have no cloud to frame, and a contour is smaller
  // than the cloud around it, so if I framed on <r> I would leave it small in
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
        {/* I match --stage in index.css here. My 3-D canvas paints its own
            opaque background, so this is the one place I do not read the stage
            colour from the stylesheet, and I have to keep the two in step by
            hand. */}
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
            {/* A lit material needs light, and my cloud never did. Both are
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
      {/* What I am showing on the stage, in its top-left corner. The design
          also put a random seed and a "1px = N pm" scale here; my sampler does
          not report a seed and my camera is a live orbit with no fixed pixel
          scale, so both would have been decoration reading as
          instrumentation. */}
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
        {/* My triad's scale. I draw it here rather than in 3-D so it cannot
            land on top of the z tip, and so I state the arm length in the
            data's own units next to the rest of my disclosures. */}
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
            solving for the level… {Math.round(isoProgress * 100)}%
          </span>
        )}
        {/* Deliberately not .ghost-hud: I border that box in counterfactual
            pink, and a contour of the real |psi|^2 is not a counterfactual. */}
        {showSurface && iso && (
          <div className="surface-hud">
            <div className="ghost-readout">
              {enclosedCaption(iso.meta)}{" "}
              <Badge provenance={iso.meta.enclosed_fraction.provenance} />
            </div>
            <div className="ghost-readout">
              |psi|^2 = {iso.meta.level.value.toExponential(3)} bohr^-3 on the{" "}
              {iso.meta.resolution}^3 grid I used, {componentsCaption(iso.meta)}
            </div>
            <div className="ghost-readout">
              {iso.meta.escaped_fraction.value.toExponential(1)} of the electron is
              outside my box entirely
            </div>
            {/* I show both error bars, because they measure different claims
                and the fraction one is nearly blind: the level hardly moves
                under a halved grid, so the fraction converges long before the
                shape does. If I showed only "fraction ± 0" beside a surface
                that is half a percent off in size it would read as
                exactness. */}
            {iso.meta.provenance.error_estimate !== null && (
              <div className="ghost-readout">
                halving my grid moves the enclosed fraction by{" "}
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
            I have no classical ghost here: the Kepler orbit needs a 1/r field,
            and the screening is the part this model adds.
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
              I am showing this at ~{slowMotionFactor(classicalGhost.collapse_time_s.value).toExponential(1)}×
              slow motion <Badge provenance={CLASSICAL_SLOWMO} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
