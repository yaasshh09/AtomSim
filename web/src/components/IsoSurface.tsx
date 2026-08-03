import { useMemo } from "react";
import * as THREE from "three";

interface Props {
  vertices: Float32Array;
  triangles: Uint32Array;
  colors: Float32Array;
}

/**
 * The engine's mesh, drawn.
 *
 * Nothing is computed here beyond the vertex normals three.js needs to light
 * it: the positions are the engine's interpolated crossings and the indices are
 * the engine's triangles, both untouched. What the renderer adds is disclosed
 * in ISOSURFACE_LIBERTY, and the important part of it is the smoothing — a
 * finite grid really does produce facets, and averaging them away makes the
 * surface look more resolved than it is.
 *
 * DoubleSide because a contour is a shell and the camera goes inside it in
 * both-modes and on any orbital with a hollow. With FrontSide the far wall
 * vanishes from within and the shell reads as an open bowl.
 */
export function IsoSurface({ vertices, triangles, colors }: Props) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.setIndex(new THREE.BufferAttribute(triangles, 1));
    // VISUAL LIBERTY: averaged across the facets that share each vertex.
    g.computeVertexNormals();
    return g;
  }, [vertices, triangles, colors]);
  return (
    // Same rotation as the point cloud: physics z is drawn screen-vertical, and
    // a surface that disagreed with the cloud it sits inside would be worse
    // than either convention on its own.
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent
        opacity={0.62}
        roughness={0.45}
        metalness={0.0}
      />
    </mesh>
  );
}
