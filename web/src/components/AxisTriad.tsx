import { Html, Line } from "@react-three/drei";
import { PHYSICS_TO_SCREEN } from "../lib/frame";

/**
 * The coordinate frame I draw the cloud in: x, y and z through the origin.
 *
 * This is not decoration. An orbital's shape only means something against its
 * quantization axis, a 2p_z and a 2p_x are the same picture rotated, and
 * without an axis to read it against my point cloud cannot tell you which one
 * it is. My `RENDER_LIBERTIES` has always declared that I draw z
 * screen-vertical; here I draw the axis that claim is about.
 *
 * I draw both halves of each axis. A single arrow would suggest a preferred
 * direction, and the negative lobe of a p orbital is as real as the positive
 * one. I report the arm length in the canvas caption rather than in 3-D, so my
 * triad carries a scale without a label colliding with the z tip.
 */

const AXES: { key: "x" | "y" | "z"; dir: [number, number, number]; color: string }[] = [
  { key: "x", dir: [1, 0, 0], color: "#ff7a6b" },
  { key: "y", dir: [0, 1, 0], color: "#6cd98a" },
  // The quantization axis, which I give the shell's own accent: it is the one
  // of the three that the physics singles out.
  { key: "z", dir: [0, 0, 1], color: "#34e0a1" },
];

/** The arm length I use, as a fraction of the camera framing distance. */
const ARM = 0.42;

export function axisArmLength(distance: number): number {
  return distance * ARM;
}

export function AxisTriad({ distance }: { distance: number }) {
  const length = axisArmLength(distance);
  return (
    // I draw this inside the same rotation as the cloud and the isosurface, so
    // the vectors below are engine axes rather than three.js ones. Without it I
    // labelled the screen-vertical axis "y" here while the cloud beside it and
    // the liberty note under it both called that axis z.
    <group rotation={PHYSICS_TO_SCREEN}>
      {AXES.map((a) => (
        <Line
          key={a.key}
          points={[
            [-a.dir[0] * length, -a.dir[1] * length, -a.dir[2] * length],
            [a.dir[0] * length, a.dir[1] * length, a.dir[2] * length],
          ]}
          color={a.color}
          lineWidth={1}
          transparent
          opacity={0.5}
          // I blend the cloud additively and the axes not at all; without this
          // I would draw an axis behind a dense lobe over the top of it, and it
          // would read as a line in front.
          depthWrite={false}
        />
      ))}
      {AXES.map((a) => (
        <Html
          key={a.key}
          position={[
            a.dir[0] * length * 1.07,
            a.dir[1] * length * 1.07,
            a.dir[2] * length * 1.07,
          ]}
          center
          // I mean this as a reading aid over the data, never a click target:
          // the canvas underneath is an orbit control.
          style={{ pointerEvents: "none" }}
        >
          <span className="axis-label" style={{ color: a.color }}>
            {a.key}
          </span>
        </Html>
      ))}
    </group>
  );
}
