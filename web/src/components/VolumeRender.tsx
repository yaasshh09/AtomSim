import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { PHYSICS_TO_SCREEN } from "../lib/frame";
import { INFERNO } from "../lib/luts";
import { volumeTexels } from "../lib/volume";

const VERTEX = /* glsl */ `
out vec3 vLocal;
void main() {
  vLocal = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// Front-to-back emission-absorption along each ray, in the box's own frame,
// which is the physics frame: the mesh carries the same rotation as the cloud.
const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D vol;
uniform sampler2D lut;
uniform vec3 camLocal;
uniform float h;
uniform float n;
uniform float absorb;
in vec3 vLocal;
// With glslVersion GLSL3, three.js declares no fragment output of its own.
layout(location = 0) out highp vec4 fragColor;

vec2 hitBox(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (vec3(-h) - o) * inv;
  vec3 t1 = (vec3(h) - o) * inv;
  vec3 lo = min(t0, t1);
  vec3 hi = max(t0, t1);
  return vec2(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

void main() {
  vec3 dir = normalize(vLocal - camLocal);
  vec2 t = hitBox(camLocal, dir);
  t.x = max(t.x, 0.0);
  if (t.x >= t.y) discard;
  const int STEPS = 192;
  float dt = (t.y - t.x) / float(STEPS);
  vec3 col = vec3(0.0);
  float alpha = 0.0;
  for (int i = 0; i < STEPS; i++) {
    vec3 p = camLocal + dir * (t.x + (float(i) + 0.5) * dt);
    // Node q of N sits at texel centre (q + 0.5) / N; the box edges are nodes.
    vec3 uvw = (p + h) / (2.0 * h) * (n - 1.0) / n + 0.5 / n;
    // The engine's grid is [x][y][z] with z fastest, so texture u is z.
    float d = texture(vol, uvw.zyx).r;
    float a = 1.0 - exp(-absorb * d * dt / h);
    vec3 c = texture(lut, vec2(d, 0.5)).rgb;
    col += (1.0 - alpha) * a * c;
    alpha += (1.0 - alpha) * a;
    if (alpha > 0.98) break;
  }
  fragColor = vec4(col, alpha);
}
`;

interface Props {
  values: Float32Array;
  resolution: number;
  halfWidth: number;
  /** Absorption per box half-width at full density: how deep into the fog a ray sees. */
  absorb: number;
}

/**
 * The engine's density grid, ray-marched as a glowing fog.
 *
 * Everything between the grid and the pixel is presentation and rides on
 * VOLUME_LIBERTY: the GPU's trilinear filtering, the 8-bit gamma-compressed
 * texels, and the absorption, which sets how far into the cloud you see.
 */
export function VolumeRender({ values, resolution, halfWidth, absorb }: Props) {
  const meshRef = useRef<THREE.Mesh>(null);
  const textures = useMemo(() => {
    const vol = new THREE.Data3DTexture(
      volumeTexels(values, resolution),
      resolution,
      resolution,
      resolution,
    );
    vol.format = THREE.RedFormat;
    vol.type = THREE.UnsignedByteType;
    vol.minFilter = THREE.LinearFilter;
    vol.magFilter = THREE.LinearFilter;
    vol.unpackAlignment = 1;
    vol.needsUpdate = true;
    const rgba = new Uint8Array(INFERNO.length * 4);
    INFERNO.forEach(([r, g, b], i) => rgba.set([r, g, b, 255], 4 * i));
    const lut = new THREE.DataTexture(rgba, INFERNO.length, 1, THREE.RGBAFormat);
    lut.minFilter = THREE.LinearFilter;
    lut.magFilter = THREE.LinearFilter;
    lut.needsUpdate = true;
    return { vol, lut };
  }, [values, resolution]);
  useEffect(
    () => () => {
      textures.vol.dispose();
      textures.lut.dispose();
    },
    [textures],
  );
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        uniforms: {
          vol: { value: textures.vol },
          lut: { value: textures.lut },
          camLocal: { value: new THREE.Vector3() },
          h: { value: halfWidth },
          n: { value: resolution },
          absorb: { value: absorb },
        },
        vertexShader: VERTEX,
        fragmentShader: FRAGMENT,
        side: THREE.BackSide,
        transparent: true,
        premultipliedAlpha: true,
        depthWrite: false,
      }),
    [textures, halfWidth, resolution, absorb],
  );
  useEffect(() => () => material.dispose(), [material]);
  useFrame(({ camera }) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const cam = material.uniforms.camLocal.value as THREE.Vector3;
    cam.copy(camera.position);
    mesh.worldToLocal(cam);
  });
  return (
    <mesh ref={meshRef} rotation={PHYSICS_TO_SCREEN} material={material}>
      <boxGeometry args={[2 * halfWidth, 2 * halfWidth, 2 * halfWidth]} />
    </mesh>
  );
}
