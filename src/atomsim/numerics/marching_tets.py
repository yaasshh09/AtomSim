"""How I extract an isosurface from a scalar field on a regular grid.

I march *tetrahedra*, not cubes, and I chose that deliberately.

The classical marching-cubes triangulation table has 256 rows and is
transcribed data: nothing in this repo can check it for me, and a single wrong
row makes a hole or a flipped facet in one corner configuration my tests may
never happen to visit. A tetrahedron has 16 cases and I can derive every one of
them in a sentence. Zero or four corners inside the region emit nothing; one
corner inside (or one outside, which is the mirror image) cuts the three edges
meeting it and emits one triangle; two inside cut four edges and emit a quad.
That is my whole algorithm, and I build `_TET_CASES` below from that statement
at import time rather than typing it in from a paper. Marching cubes'
ambiguous-face problem, which has to be settled by convention, never arises for
me.

It costs me roughly twice the triangles for the same grid, and slightly worse
triangle shapes. Neither touches any number I report.

I split each cell by the Kuhn decomposition into six tetrahedra that all share
the cell's main diagonal. I apply the rule identically in every cell, so two
neighbouring cells split the face between them along the same diagonal and my
surface comes out watertight: every edge is shared by exactly two triangles.

I orient outward, meaning my normals point away from the region where the field
is at or above the level, so `enclosed_volume` comes out positive. I know
nothing about atoms here; I take an array and return a mesh.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

# I index cell corners as (di, dj, dk) bits, by di*4 + dj*2 + dk.
_CORNER_OFFSETS = np.array(
    [(di, dj, dk) for di in (0, 1) for dj in (0, 1) for dk in (0, 1)], dtype=np.int64
)


def _kuhn_tetrahedra() -> np.ndarray:
    """The six tetrahedra of the Kuhn decomposition, as corner indices.

    Each is the path from corner (0,0,0) to corner (1,1,1) taking the three unit
    steps in one of the six orders. Every tetrahedron therefore contains the
    main diagonal, the six of them tile the cell exactly, and the split they
    induce on each cell face runs between that face's lowest and highest corner,
    which is a property of the face rather than of the cell, so my neighbours
    agree.
    """
    tets = []
    for order in permutations(range(3)):
        position = [0, 0, 0]
        path = [0]
        for axis in order:
            position[axis] = 1
            path.append(position[0] * 4 + position[1] * 2 + position[2])
        tets.append(path)
    return np.array(tets, dtype=np.int64)


_TETS = _kuhn_tetrahedra()

# The six edges of a tetrahedron, which I hold as pairs of local corner indices 0..3.
_TET_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64)
_EDGE_OF = {(a, b): e for e, (a, b) in enumerate(map(tuple, _TET_EDGES))}
_EDGE_OF.update({(b, a): e for (a, b), e in list(_EDGE_OF.items())})


def _tet_cases() -> np.ndarray:
    """My triangulation for each of the 16 inside/outside patterns.

    I return an int8 array of shape (16, 2, 3): up to two triangles per case,
    each as three tetrahedron-edge indices, with -1 where I emit no triangle.

    I derive this rather than tabulating it. The winding here is arbitrary;
    `marching_tets` fixes orientation per triangle against the inside corners,
    which is cheap and cannot be got wrong by a sign convention in a table.
    """
    cases = np.full((16, 2, 3), -1, dtype=np.int8)
    for code in range(16):
        inside = [c for c in range(4) if code & (1 << c)]
        outside = [c for c in range(4) if not code & (1 << c)]
        if not inside or not outside:
            continue
        if len(inside) == 1 or len(outside) == 1:
            # One corner alone on its side of the level: I cut the three edges
            # that meet it, and they carry a single triangle.
            alone = inside[0] if len(inside) == 1 else outside[0]
            others = outside if len(inside) == 1 else inside
            cases[code, 0] = [_EDGE_OF[(alone, other)] for other in others]
        else:
            # Two and two: I cut the four edges joining an inside corner to an
            # outside one, and they form a quad. Ordering it as
            # (a,c) (a,d) (b,d) (b,c) walks the quad's rim rather than its
            # diagonal, so splitting it 0-1-2 / 0-2-3 gives me two real
            # triangles.
            a, b = inside
            c, d = outside
            rim = [_EDGE_OF[(a, c)], _EDGE_OF[(a, d)], _EDGE_OF[(b, d)], _EDGE_OF[(b, c)]]
            cases[code, 0] = [rim[0], rim[1], rim[2]]
            cases[code, 1] = [rim[0], rim[2], rim[3]]
    return cases


_TET_CASES = _tet_cases()


@dataclass(frozen=True)
class Mesh:
    """A welded, oriented, indexed triangle mesh of mine.

    `vertices` is (M, 3) in the field's own coordinates; `triangles` is (K, 3)
    of indices into it. I make shared vertices one vertex, which is what makes
    watertightness and connected-component counts mean anything.
    """

    vertices: np.ndarray
    triangles: np.ndarray

    @property
    def is_empty(self) -> bool:
        return self.triangles.shape[0] == 0


def marching_tets(
    field: np.ndarray,
    level: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: float = 1.0,
    chunk: int = 24,
) -> Mesh:
    """I extract the surface where `field` crosses `level`.

    I take the enclosed region to be `field >= level`. I land vertices on cell
    edges by linear interpolation of the field, which is what makes my surface
    converge on the true level set as the grid refines rather than sitting on
    voxel corners.

    `chunk` is how many cell layers in x I process at once. A 128 cubed grid
    holds 12 million tetrahedra, and materializing their corner values all at
    once is pointless when the work is embarrassingly parallel in slabs.
    """
    field = np.ascontiguousarray(field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError(f"I need a 3-D field, got shape {field.shape}")
    nx, ny, nz = field.shape
    if min(nx, ny, nz) < 2:
        raise ValueError(f"I need at least 2 points per axis, got {field.shape}")

    flat = field.reshape(-1)
    strides = np.array([ny * nz, nz, 1], dtype=np.int64)
    origin_arr = np.asarray(origin, dtype=np.float64)

    keys: list[np.ndarray] = []
    points: list[np.ndarray] = []

    for i0 in range(0, nx - 1, chunk):
        i1 = min(i0 + chunk, nx - 1)
        cells = np.stack(
            np.meshgrid(
                np.arange(i0, i1, dtype=np.int64),
                np.arange(ny - 1, dtype=np.int64),
                np.arange(nz - 1, dtype=np.int64),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        # The global flat index I give each of the eight corners of each cell.
        corner_ids = (cells[:, None, :] + _CORNER_OFFSETS[None, :, :]) @ strides

        for tet in _TETS:
            ids = corner_ids[:, tet]  # (ncells, 4)
            values = flat[ids]
            inside = values >= level
            code = (
                inside[:, 0].astype(np.int64)
                | (inside[:, 1].astype(np.int64) << 1)
                | (inside[:, 2].astype(np.int64) << 2)
                | (inside[:, 3].astype(np.int64) << 3)
            )
            active = (code != 0) & (code != 15)
            if not active.any():
                continue

            ids = ids[active]
            values = values[active]
            inside = inside[active]
            tris = _TET_CASES[code[active]]  # (n, 2, 3)

            # I do both triangle slots at once; the second is empty for most cases.
            for slot in range(2):
                edges = tris[:, slot, :]
                live = edges[:, 0] >= 0
                if not live.any():
                    continue
                e = edges[live].astype(np.int64)
                sub_ids = ids[live]
                sub_values = values[live]
                sub_inside = inside[live]

                ends = _TET_EDGES[e]  # (n, 3, 2) local corner indices
                a_local = ends[..., 0]
                b_local = ends[..., 1]
                a_id = np.take_along_axis(sub_ids, a_local, axis=1)
                b_id = np.take_along_axis(sub_ids, b_local, axis=1)
                a_val = np.take_along_axis(sub_values, a_local, axis=1)
                b_val = np.take_along_axis(sub_values, b_local, axis=1)

                t = (level - a_val) / (b_val - a_val)
                a_pos = _unflatten(a_id, strides) * spacing + origin_arr
                b_pos = _unflatten(b_id, strides) * spacing + origin_arr
                pos = a_pos + t[..., None] * (b_pos - a_pos)

                # I point outward: away from the corners inside the region.
                inside_pos = _inside_centroid(sub_ids, sub_inside, strides, spacing, origin_arr)
                normal = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
                flip = np.einsum("ij,ij->i", normal, pos.mean(axis=1) - inside_pos) < 0
                order = np.where(flip[:, None], np.array([0, 2, 1]), np.array([0, 1, 2]))

                lo = np.minimum(a_id, b_id)
                hi = np.maximum(a_id, b_id)
                key = lo * flat.size + hi
                keys.append(np.take_along_axis(key, order, axis=1).reshape(-1))
                points.append(
                    np.take_along_axis(pos, order[:, :, None], axis=1).reshape(-1, 3)
                )

    if not keys:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    all_keys = np.concatenate(keys)
    all_points = np.concatenate(points)
    unique, first, inverse = np.unique(all_keys, return_index=True, return_inverse=True)
    return Mesh(all_points[first], inverse.reshape(-1, 3).astype(np.int32))


def _unflatten(flat_ids: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """I turn flat grid indices back into (i, j, k) as floats, for position arithmetic."""
    i, rest = np.divmod(flat_ids, strides[0])
    j, k = np.divmod(rest, strides[1])
    return np.stack([i, j, k], axis=-1).astype(np.float64)


def _inside_centroid(ids, inside, strides, spacing, origin):
    """The mean position of the tetrahedron corners inside the region.

    I use it only to point my normal outward. Any point strictly inside would
    do; this one is cheap and always on the correct side, because the triangle
    separates the inside corners from the outside ones by construction.
    """
    positions = _unflatten(ids, strides) * spacing + origin
    weights = inside.astype(np.float64)
    return (positions * weights[..., None]).sum(axis=1) / weights.sum(axis=1)[:, None]


def enclosed_volume(mesh: Mesh) -> float:
    """The volume the mesh encloses, which I get by the divergence theorem.

    I sum the signed tetrahedra from the origin to each facet. It comes out
    positive for the outward orientation I produce, and it does not depend on
    where the origin sits, which is what makes it a check on my triangles
    rather than on my coordinates.
    """
    if mesh.is_empty:
        return 0.0
    v = mesh.vertices[mesh.triangles]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def surface_area(mesh: Mesh) -> float:
    if mesh.is_empty:
        return 0.0
    v = mesh.vertices[mesh.triangles]
    return float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2)


def edge_use_counts(mesh: Mesh) -> np.ndarray:
    """How many triangles I put on each undirected edge. Watertight means all twos."""
    tri = mesh.triangles.astype(np.int64)
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    lo = pairs.min(axis=1)
    hi = pairs.max(axis=1)
    _, counts = np.unique(lo * (mesh.vertices.shape[0] + 1) + hi, return_counts=True)
    return counts


def connected_components(mesh: Mesh) -> int:
    """How many disconnected pieces I find, by union-find over my welded vertices.

    This means something only because I weld the mesh: an unwelded surface has
    one component per triangle, and I would report a p orbital as thousands of
    lobes.
    """
    if mesh.is_empty:
        return 0
    parent = np.arange(mesh.vertices.shape[0])

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in mesh.triangles:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[ru] = rv
    roots = {find(int(i)) for i in np.unique(mesh.triangles)}
    return len(roots)
