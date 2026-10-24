import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";

/**
 * The point cloud and the isosurface as one binary glTF, in bohr.
 *
 * glTF's convention is one unit per metre, so a viewer will show an atom tens
 * of metres across; the file name says bohr for that reason. Built from the
 * engine's arrays rather than read off the live scene, so what goes out is
 * the data and not whatever the camera happened to be doing.
 */
export function glbScene(
  positions: Float32Array | null,
  iso: { vertices: Float32Array; triangles: Uint32Array } | null,
): Promise<ArrayBuffer> {
  const scene = new THREE.Scene();
  if (positions) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 0.05 })));
  }
  if (iso) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(iso.vertices, 3));
    g.setIndex(new THREE.BufferAttribute(iso.triangles, 1));
    g.computeVertexNormals();
    scene.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x7dd3fc })));
  }
  return new Promise((resolve, reject) => {
    new GLTFExporter().parse(
      scene,
      (out) => (out instanceof ArrayBuffer ? resolve(out) : reject(new Error("expected binary glTF"))),
      (err) => reject(err),
      { binary: true },
    );
  });
}
