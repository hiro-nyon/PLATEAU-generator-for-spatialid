import itertools
import math
import os
import tempfile
import typing
import logging

import lxml.etree as etree
import numpy as np
import pygeos as geos

# Set writable caches early to avoid matplotlib/fontconfig penalties during pyvista import
_default_cache_dir = os.path.join(tempfile.gettempdir(), 'plateau_cache')
if 'MPLCONFIGDIR' not in os.environ:
    os.makedirs(_default_cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = _default_cache_dir
if 'XDG_CACHE_HOME' not in os.environ:
    os.environ['XDG_CACHE_HOME'] = _default_cache_dir

import pyvista as vista
try:
    import pyvista._vtk as vtk  # pyvista<=0.43
except ImportError:
    import vtk  # fallback for newer pyvista where _vtk is absent
from vtk.util import numpy_support as vtknp

Any = typing.Any
Set = typing.Set
Dict = typing.Dict
List = typing.List
Tuple = typing.Tuple
Union = typing.Union
Iterator = typing.Iterable
Optional = typing.Optional

logger = logging.getLogger(__name__)

# #region agent log
import json as _json, time as _time
_DBG_LOG = '/data/output/debug-43a43b.log'
def _dbg(msg, **data):
    try:
        with open(_DBG_LOG, 'a') as _f:
            _f.write(_json.dumps({'sessionId':'43a43b','timestamp':int(_time.time()*1000),'location':'grids.py','message':msg,'data':data},ensure_ascii=False)+'\n')
    except Exception:
        pass
# #endregion


def get(type: str, level: int = None, size: List[float] = None,
        crs: int = None) -> 'Grid':
    """get voxel grid instance

    Args:
        type (str): type of the voxel grid
        level (int, optional): zoom level of the voxel grid. Defaults to None.
        size (List[float], optional): size of the voxel grid. Defaults to None.
        crs (int, optional): coordinate reference system of the voxel grid. Defaults to None.

    Raises:
        ValueError: parameter error

    Returns:
        Grid: Voxel Grid instance
    """
    if type == 'zfxy':
        grid = ZFXYGrid(level=level)
    else:
        raise ValueError(f'Invalid grid type: {type}')
    return grid


class Grid(object):

    size_x: float = None
    size_y: float = None
    size_z: float = None

    crs: int = None
    output_crs: int = None

    def __init__(self, *args, **kwargs):
        self._data = {}
        self._underground_wall_only: bool = False
        self._underground_z_min: Optional[float] = None
        self._underground_z_max: Optional[float] = None

    @property
    def data(self) -> Dict[Tuple[float], Any]:
        return self._data

    def clear(self):
        self._data = {}

    def normalize_size(self, size: Union[float, List[float]]):
        if size is None:
            return None
        if isinstance(size, float):
            size = [size]
        normalized_size = size + (3 - len(size)) * [size[-1]]
        return normalized_size

    def encode_key(self, ix: int, iy: int, iz: int) -> str:
        return f'{ix}_{iy}_{iz}'

    def decode_key(self, key: str, full: bool = False) -> str:
        return (int(s) for s in key.split('_'))

    def _use_edge_only_voxelization(self, props: Dict[str, Any]) -> bool:
        fc_name = str(props.get('fc_name') or '')
        gml_id = str(props.get('gml_id') or '')
        return self._underground_wall_only or fc_name == 'UndergroundBuilding' or gml_id.startswith('ubld_')

    def _encode_keys_vectorized(self, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray = None, level: int = None) -> List[str]:
        if level is None:
            level = self.level
        # 2D/3D共通のグリッドインデックス変換
        gix = (ix + (1 << max(level - 1, 0))) % (1 << level)
        giy = (1 << max(level - 1, 0)) - iy - 1
        gix_s = gix.astype(str)
        giy_s = giy.astype(str)
        level_s = np.full(gix.shape, str(level), dtype=object)
        if iz is None:
            return [f'{lv}/{x}/{y}' for lv, x, y in zip(level_s, gix_s, giy_s)]
        giz = iz
        if giz.ndim == 1:
            giz = giz[:, None]
        giz_s = giz.astype(str)
        return [f'{lv}/{z}/{x}/{y}' for lv, x, y, z in zip(level_s, gix_s, giy_s, giz_s.flatten())]

    def load_geom_data(self,
                data: Iterator[Tuple[List[List[List[float]]], Dict[str, Any]]],
                interpolate: bool = False, fill_solid: bool = False,
                underground_wall_only: bool = False,
                underground_z_min: Optional[float] = None,
                underground_z_max: Optional[float] = None,
                merge: bool = False) -> None:
        self._underground_wall_only = underground_wall_only
        self._underground_z_min = underground_z_min
        self._underground_z_max = underground_z_max

        if fill_solid:
            # Surface/line を個別 voxelize し、後段で column infill を適用する。
            # interior ring は hole 候補として収集し、infill 由来の voxel から差し引く。
            hole_keys_by_gml: Dict[str, Set[str]] = {}
            with vista.utilities.VtkErrorCatcher(raise_errors=False, send_to_logging=False):
                for geom, props in data:
                    gml_id = props.get('gml_id')
                    if not gml_id:
                        continue
                    if props.get('geom_dim') != 3:
                        self._update_by_geom(geom, props)
                        continue

                    data_set = self._data.setdefault(gml_id, set())
                    if props.get('geom_line'):
                        self._update_by_geom_3d(data_set, geom, props)
                        continue

                    surface_keys, hole_keys = self._voxelize_polygon_3d_with_holes(geom, props)
                    data_set.update(surface_keys)
                    if hole_keys:
                        hole_keys_by_gml.setdefault(gml_id, set()).update(hole_keys)

            # 各 feature ごとに infill 実行後、hole column を削る。
            for gml_id, data_set in self._data.items():
                before_infill = set(data_set)
                self._infill_vertical_columns(data_set)
                hole_keys = hole_keys_by_gml.get(gml_id)
                if not hole_keys:
                    continue
                hole_span = self._expand_column_keys(hole_keys, include_edge=True)
                if not hole_span:
                    continue
                # 元の surface/wire voxel は保持し、infill で生成された分だけ除去する。
                hole_span.difference_update(before_infill)
                data_set.difference_update(hole_span)
        else:
            with vista.utilities.VtkErrorCatcher(raise_errors=False, send_to_logging=False):
                for geom, props in data:
                    self._update_by_geom(geom, props)

        # interpolate inner voxels (legacy)
        if interpolate:
            # #region agent log
            _pre_interp = {gid: len(s) for gid, s in self._data.items()}
            _dbg('interpolate_before', hypothesisId='B',
                 n_buildings=len(self._data),
                 sample_buildings={k: v for i, (k, v) in enumerate(_pre_interp.items()) if i < 5})
            # #endregion
            self.interpolate()
            # #region agent log
            _post_interp = {gid: len(s) for gid, s in self._data.items()}
            _sample_diffs = {}
            for _gid in list(_pre_interp)[:5]:
                _bef = _pre_interp.get(_gid, 0)
                _aft = _post_interp.get(_gid, 0)
                _sample_diffs[_gid] = {'before': _bef, 'after': _aft, 'added': _aft - _bef}
            _dbg('interpolate_after', hypothesisId='B', sample_diffs=_sample_diffs)
            # #endregion

        # merge small voxels
        if merge:
            self.merge()

    def _voxelize_closed_solid(self, data: Set[str],
                face_list: List[Tuple[List[List[List[float]]], Dict[str, Any]]]) -> None:
        """Voxelize a closed 3D solid by combining all surface faces into one mesh.

        Builds a single pyvista PolyData from all surface polygons, then uses
        pyvista.voxelize() to find all voxels inside or intersecting the closed solid.
        """
        # Build combined mesh from all faces
        all_points = []
        all_faces = []
        offset = 0

        for geom, props in face_list:
            for ring_group in [geom]:
                if not ring_group:
                    continue
                # Use only exterior ring (first element)
                exterior = ring_group[0] if ring_group else []
                if not exterior or len(exterior) < 3:
                    continue
                ring = list(exterior)
                if len(ring) > 1 and ring[0] == ring[-1]:
                    ring = ring[:-1]
                if len(ring) < 3:
                    continue
                n = len(ring)
                all_points.extend(ring)
                all_faces.append(n)
                all_faces.extend(range(offset, offset + n))
                offset += n

        if not all_points or offset < 3:
            return

        try:
            mesh = vista.PolyData(all_points, all_faces)
            mesh.points_to_double()
            mesh.triangulate(inplace=True)
        except Exception as e:
            logger.debug(f'failed to build combined mesh: {e}')
            return

        size_x = self.size_x
        size_y = self.size_y
        size_z = self.size_z

        bounds = mesh.bounds
        ix_min = math.floor(bounds[0] / size_x)
        ix_max = math.ceil(bounds[1] / size_x)
        iy_min = math.floor(bounds[2] / size_y)
        iy_max = math.ceil(bounds[3] / size_y)
        iz_min = math.floor(bounds[4] / size_z)
        iz_max = math.ceil(bounds[5] / size_z)

        # Offset to origin for numerical stability
        offset_x = ix_min * size_x
        offset_y = iy_min * size_y
        offset_z = iz_min * size_z
        mesh.translate((-offset_x, -offset_y, -offset_z), inplace=True)

        try:
            vox = mesh.voxelize(density=(size_x, size_y, size_z), check_surface=False)
        except Exception as e:
            logger.debug(f'pyvista voxelize failed: {e}')
            # Fallback: voxelize each face individually
            for geom, props in face_list:
                self._update_by_geom_3d(data, geom, props)
            return

        # Extract voxel center positions
        if vox.n_cells > 0:
            centers = vox.cell_centers().points
        elif vox.n_points > 0:
            centers = vox.points
        else:
            logger.debug('voxelize returned empty result')
            return

        if len(centers) == 0:
            return

        centers = np.asarray(centers, dtype=np.float64)
        ix_arr = np.floor(centers[:, 0] / size_x).astype(np.int64) + ix_min
        iy_arr = np.floor(centers[:, 1] / size_y).astype(np.int64) + iy_min
        iz_arr = np.floor(centers[:, 2] / size_z).astype(np.int64) + iz_min

        keys = self._encode_keys_vectorized(ix_arr, iy_arr, iz_arr, level=self.level)
        data.update(keys)
        logger.info(f'solid voxelization: {len(keys)} voxels from {len(face_list)} faces')

    def _update_by_geom(self, geom: List[List[List[float]]],
                props: Dict[str, Any]) -> None:
        gml_id = props.get('gml_id')
        if not gml_id:
            raise ValueError('`gml_id` not found')
        if len(geom) == 0:
            raise ValueError('geometry contains no points')
        data = self._data.setdefault(gml_id, set())
        if props.get('geom_dim') == 3:
            self._update_by_geom_3d(data, geom, props)
        else:
            self._update_by_geom_2d(data, geom, props)

    def _update_by_geom_3d(self, data: Set[str],
                coordinates: List[List[List[float]]],
                props: Dict[str, Any]) -> None:
        if props.get('geom_line'):
            line_coords: List[List[float]] = []
            if coordinates:
                first = coordinates[0]
                if (
                    first and
                    isinstance(first[0], (list, tuple)) and
                    len(first[0]) >= 3 and
                    isinstance(first[0][0], (int, float, np.floating))
                ):
                    line_coords = first  # [ [x,y,z], ... ]
                elif first and isinstance(first[0], (list, tuple)):
                    line_coords = first[0]  # [ [ [x,y,z], ... ] ]
            if line_coords:
                data.update(self._voxelize_wire_3d(line_coords, props))
            return

        if not self._use_edge_only_voxelization(props):
            exterior = coordinates[0]

            size_x = self.size_x
            size_y = self.size_y
            size_z = self.size_z

            # #region agent log
            _z_vals = [pt[2] for pt in exterior if len(pt) >= 3]
            _z_span = max(_z_vals) - min(_z_vals) if _z_vals else -1
            _z_tol = max(1e-6, size_z * 1e-6)
            _is_horiz = _z_span <= _z_tol
            _dbg('face_enter', hypothesisId='A', gml_id=props.get('gml_id'), fc_name=props.get('fc_name'),
                 z_span=round(_z_span, 9), z_tol=round(_z_tol, 9), is_horizontal=_is_horiz,
                 n_exterior_pts=len(exterior), data_size_before=len(data))
            # #endregion

            # FIX: Route horizontal faces to pygeos 2D intersection instead of
            # VTK collision detection, which misses interior voxels for flat polygons.
            # Coordinate transforms introduce z-noise up to ~10mm on horizontal faces;
            # use half-voxel threshold to safely separate from wall faces (z_span >> size_z).
            z_horiz_threshold = size_z * 0.5
            if _z_vals:
                _face_z_span = max(_z_vals) - min(_z_vals)
                if _face_z_span <= z_horiz_threshold:
                    z_value = sum(_z_vals) / len(_z_vals)
                    data.update(self._voxelize_horizontal_ring(exterior, z_value))
                    # #region agent log
                    _dbg('face_exit_horizontal_fix', hypothesisId='A_fix',
                         gml_id=props.get('gml_id'), z_value=round(z_value, 6),
                         z_span=round(_face_z_span, 9), threshold=round(z_horiz_threshold, 6),
                         data_size_after=len(data))
                    # #endregion
                    return

            abs_geom_vtk = self.build_geom_3d(exterior)

            bounds = abs_geom_vtk.bounds
            ix_min = math.floor(bounds[0] / size_x)
            ix_max = math.ceil(bounds[1] / size_x)
            iy_min = math.floor(bounds[2] / size_y)
            iy_max = math.ceil(bounds[3] / size_y)
            iz_min = math.floor(bounds[4] / size_z)
            iz_max = math.ceil(bounds[5] / size_z)
            offset_x = ix_min * size_x
            offset_y = iy_min * size_y
            offset_z = iz_min * size_z
            rel_geom_vtk = self.build_geom_3d([
                [
                    x - offset_x,
                    y - offset_y,
                    z - offset_z
                ]
                for x, y, z in exterior
            ])
            rel_bounds = rel_geom_vtk.bounds
            collision_filter = self.build_collision_filter(rel_geom_vtk)
            voxel_geom_vtk = self.build_box_3d(
                cx=size_x/2,
                cy=size_y/2,
                cz=size_z/2,
                sx=size_x,
                sy=size_y,
                sz=size_z
            )
            matrix = vtk.vtkMatrix4x4()
            collision_filter.SetInputData(1, voxel_geom_vtk)
            collision_filter.SetMatrix(1, matrix)
            # #region agent log
            _dbg_before = len(data)
            # #endregion
            for rx in range(ix_max - ix_min + 1):
                for ry in range(iy_max - iy_min + 1):
                    for rz in range(iz_max - iz_min + 1):
                        matrix.SetElement(0, 3, rx * size_x)
                        matrix.SetElement(1, 3, ry * size_y)
                        matrix.SetElement(2, 3, rz * size_z)
                        collision_filter.Update()
                        n_collision = collision_filter.GetNumberOfContacts()
                        if n_collision > 0:
                            key = self.encode_key(
                                ix_min + rx,
                                iy_min + ry,
                                iz_min + rz
                            )
                            data.add(key)
                        else:
                            voxel_bounds = (
                                rx * size_x,
                                rx * size_x + size_x,
                                ry * size_y,
                                ry * size_y + size_y,
                                rz * size_z,
                                rz * size_z + size_z
                            )
                            if (rel_bounds[0] >= voxel_bounds[0] and
                                rel_bounds[1] <= voxel_bounds[1] and
                                rel_bounds[2] >= voxel_bounds[2] and
                                rel_bounds[3] <= voxel_bounds[3] and
                                rel_bounds[4] >= voxel_bounds[4] and
                                rel_bounds[5] <= voxel_bounds[5]):

                                key = self.encode_key(
                                    ix_min + rx,
                                    iy_min + ry,
                                    iz_min + rz
                                )
                                data.add(key)
            # #region agent log
            _after = len(data)
            _dbg('face_exit_vtk', hypothesisId='A', gml_id=props.get('gml_id'),
                 is_horizontal=_is_horiz, voxels_added=_after - _dbg_before,
                 total_voxels=_after,
                 bbox_voxels=(ix_max-ix_min+1)*(iy_max-iy_min+1)*(iz_max-iz_min+1),
                 ix_span=ix_max-ix_min+1, iy_span=iy_max-iy_min+1, iz_span=iz_max-iz_min+1)
            # #endregion
            return

        keys, _ = self._voxelize_polygon_3d_with_holes(coordinates, props)
        data.update(keys)

    def _voxelize_polygon_3d_with_holes(
        self,
        coordinates: List[List[List[float]]],
        props: Dict[str, Any],
    ) -> Tuple[Set[str], Set[str]]:
        if len(coordinates) == 0:
            return set(), set()

        exterior = coordinates[0]
        # Polygon holes (interior rings) must be removed from voxelized result.
        interiors = coordinates[1:] if len(coordinates) > 1 else []

        keys = self._voxelize_ring_3d(exterior, props)
        hole_union: Set[str] = set()
        for interior in interiors:
            hole_keys = self._voxelize_ring_3d(interior, props)
            if hole_keys:
                keys.difference_update(hole_keys)
                hole_union.update(hole_keys)
        return keys, hole_union

    def _voxelize_wire_3d(self, ring: List[List[float]], props: Dict[str, Any]) -> Set[str]:
        keys: Set[str] = set()
        if not ring:
            return keys

        points = np.asarray(ring, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0:
            return keys
        if points.shape[0] > 1 and np.allclose(points[0], points[-1], atol=1e-9):
            points = points[:-1]
        if points.shape[0] == 0:
            return keys

        if points.shape[0] == 1:
            sample_points = points
        else:
            # 半ボクセル刻みで各セグメントをサンプリングして wire を voxel 化する。
            step = max(min(self.size_x, self.size_y, self.size_z) * 0.5, 1e-6)
            sampled_segments: List[np.ndarray] = []
            for idx in range(points.shape[0] - 1):
                p0 = points[idx]
                p1 = points[idx + 1]
                delta = p1 - p0
                distance = float(np.linalg.norm(delta))
                steps = max(1, int(math.ceil(distance / step)))
                ts = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
                seg_points = p0 + ts[:, None] * delta
                if idx > 0:
                    seg_points = seg_points[1:]
                sampled_segments.append(seg_points)
            sample_points = np.vstack(sampled_segments) if sampled_segments else points[:1]

        ix_arr = np.floor(sample_points[:, 0] / self.size_x).astype(np.int64)
        iy_arr = np.floor(sample_points[:, 1] / self.size_y).astype(np.int64)
        iz_arr = np.floor(sample_points[:, 2] / self.size_z).astype(np.int64)
        keys.update(self._encode_keys_vectorized(ix_arr, iy_arr, iz_arr, level=self.level))
        return keys

    def _collect_column_bounds(self, keys: Set[str]) -> Dict[Tuple[int, int], List[int]]:
        column_bounds: Dict[Tuple[int, int], List[int]] = {}
        for key in keys:
            ix, iy, iz = self.decode_key(key)
            if iz is None:
                continue
            col_key = (ix, iy)
            if col_key not in column_bounds:
                column_bounds[col_key] = [iz, iz]
            else:
                column_bounds[col_key][0] = min(column_bounds[col_key][0], iz)
                column_bounds[col_key][1] = max(column_bounds[col_key][1], iz)
        return column_bounds

    def _expand_column_keys(self, keys: Set[str], *, include_edge: bool = False) -> Set[str]:
        expanded: Set[str] = set()
        column_bounds = self._collect_column_bounds(keys)
        for (ix, iy), (iz_min, iz_max) in column_bounds.items():
            start = iz_min if include_edge else iz_min + 1
            end = iz_max + 1
            for iz in range(start, end):
                expanded.add(self.encode_key(ix, iy, iz))
        return expanded

    def _infill_vertical_columns(self, data: Set[str]) -> None:
        infill_keys = self._expand_column_keys(data, include_edge=False)
        if infill_keys:
            data.update(infill_keys)

    # ------------------------------------------------------------------
    # Improved solid fill: footprint-projected floor + z-slice flood fill
    # ------------------------------------------------------------------

    def _infill_solid_improved(self, data: Set[str]) -> None:
        """Two-phase interior fill that handles missing GroundSurface.

        Phase B: Identify the building footprint from wall-column bounds,
                 inject a virtual floor at the global iz_min for all
                 interior columns, then run the standard column fill.
        Phase D: For each z-slice, 2D flood-fill interior cells bounded
                 by surface voxels to catch any remaining gaps.
        """
        if not data:
            return

        # --- decode all existing voxels into (ix, iy, iz) tuples ---
        col_bounds: Dict[Tuple[int, int], List[int]] = {}
        all_cells: Set[Tuple[int, int, int]] = set()
        for key in data:
            ix, iy, iz = self.decode_key(key)
            if iz is None:
                continue
            all_cells.add((ix, iy, iz))
            col = (ix, iy)
            if col not in col_bounds:
                col_bounds[col] = [iz, iz]
            else:
                if iz < col_bounds[col][0]:
                    col_bounds[col][0] = iz
                if iz > col_bounds[col][1]:
                    col_bounds[col][1] = iz

        if not col_bounds:
            return

        # --- Phase B: virtual floor injection ---
        # Find global ground level from columns that have a real vertical
        # span (wall columns typically have iz_min much lower than iz_max).
        spans = [(col, b) for col, b in col_bounds.items() if b[1] > b[0]]
        if spans:
            global_iz_min = min(b[0] for _, b in spans)
        else:
            global_iz_min = min(b[0] for b in col_bounds.values())

        # Determine the 2D bounding box of all columns
        all_ix = [c[0] for c in col_bounds]
        all_iy = [c[1] for c in col_bounds]
        ix_lo, ix_hi = min(all_ix), max(all_ix)
        iy_lo, iy_hi = min(all_iy), max(all_iy)

        # Build a 2D occupancy grid of existing columns
        footprint_cols = set(col_bounds.keys())

        # For each column that only has voxels at the roof level
        # (iz_min == iz_max), inject a floor voxel at global_iz_min
        # if the column is within the footprint.
        injected = set()
        for col in footprint_cols:
            iz_min_col, iz_max_col = col_bounds[col]
            if iz_min_col == iz_max_col and iz_min_col > global_iz_min:
                # This column has only roof-level voxels — inject floor
                floor_key = self.encode_key(col[0], col[1], global_iz_min)
                injected.add(floor_key)
                all_cells.add((col[0], col[1], global_iz_min))
                # Update bounds for the standard column fill
                col_bounds[col][0] = global_iz_min

        if injected:
            data.update(injected)

        # Standard column fill (now with corrected bounds)
        self._infill_vertical_columns(data)

        # --- Phase D: z-slice 2D flood fill ---
        # Rebuild cell set after column fill
        all_cells_after: Set[Tuple[int, int, int]] = set()
        iz_set: Set[int] = set()
        for key in data:
            ix, iy, iz = self.decode_key(key)
            if iz is None:
                continue
            all_cells_after.add((ix, iy, iz))
            iz_set.add(iz)

        # For each z-level, flood-fill interior holes
        for z_level in iz_set:
            slice_cells = {(c[0], c[1]) for c in all_cells_after if c[2] == z_level}
            if len(slice_cells) < 3:
                continue
            filled = self._flood_fill_2d_slice(slice_cells, ix_lo, ix_hi, iy_lo, iy_hi)
            for (fx, fy) in filled:
                data.add(self.encode_key(fx, fy, z_level))

    @staticmethod
    def _flood_fill_2d_slice(
        boundary: Set[Tuple[int, int]],
        ix_lo: int, ix_hi: int,
        iy_lo: int, iy_hi: int,
    ) -> Set[Tuple[int, int]]:
        """Find interior cells in a 2D slice using outside-in flood fill.

        Flood-fills from the border of the bounding box (expanded by 1).
        Any cell NOT reached by the flood and NOT already in boundary
        is interior.
        """
        # Expand bounding box by 1 to ensure the flood can wrap around
        x_lo = ix_lo - 1
        x_hi = ix_hi + 1
        y_lo = iy_lo - 1
        y_hi = iy_hi + 1

        # BFS from all border cells that are not boundary voxels
        visited: Set[Tuple[int, int]] = set()
        queue = []
        for x in range(x_lo, x_hi + 1):
            for y in (y_lo, y_hi):
                if (x, y) not in boundary:
                    queue.append((x, y))
                    visited.add((x, y))
        for y in range(y_lo + 1, y_hi):
            for x in (x_lo, x_hi):
                if (x, y) not in boundary:
                    queue.append((x, y))
                    visited.add((x, y))

        idx = 0
        while idx < len(queue):
            cx, cy = queue[idx]
            idx += 1
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if nx < x_lo or nx > x_hi or ny < y_lo or ny > y_hi:
                    continue
                if (nx, ny) in visited or (nx, ny) in boundary:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))

        # Interior = cells inside bounding box that are neither boundary
        # nor reachable from outside
        interior: Set[Tuple[int, int]] = set()
        for x in range(ix_lo, ix_hi + 1):
            for y in range(iy_lo, iy_hi + 1):
                if (x, y) not in boundary and (x, y) not in visited:
                    interior.add((x, y))
        return interior

    def _voxelize_ring_3d(self, ring: List[List[float]], props: Dict[str, Any]) -> Set[str]:
        keys: Set[str] = set()

        size_x = self.size_x
        size_y = self.size_y
        size_z = self.size_z

        # target
        abs_geom_vtk = self.build_geom_3d(ring)

        bounds = abs_geom_vtk.bounds
        ix_min = math.floor(bounds[0] / size_x)
        ix_max = math.ceil(bounds[1] / size_x)
        iy_min = math.floor(bounds[2] / size_y)
        iy_max = math.ceil(bounds[3] / size_y)
        iz_min = math.floor(bounds[4] / size_z)
        iz_max = math.ceil(bounds[5] / size_z)
        span_x = bounds[1] - bounds[0]
        span_y = bounds[3] - bounds[2]
        span_z = bounds[5] - bounds[4]

        if self._underground_wall_only:
            z_tolerance = max(1e-6, size_z * 1e-6)
            if span_z <= z_tolerance:
                logger.debug('skip horizontal ring in underground-wall-only mode: %s', props.get('gml_id'))
                return keys

            # 壁面はZ値が2種類（上端/下端）の面として扱う。
            z_unique = set()
            for pt in ring:
                z_unique.add(round(pt[2], 6))
                if len(z_unique) > 2:
                    break
            if len(z_unique) <= 2:
                keys.update(
                    self._voxelize_wall_edges(
                        ring,
                        z_unique,
                        size_x,
                        size_y,
                        size_z,
                        full_span=False,
                        z_min=self._underground_z_min,
                        z_max=self._underground_z_max,
                    )
                )
            else:
                logger.debug('skip non-wall ring in underground-wall-only mode: %s', props.get('gml_id'))
            return keys

        # 微小な浮動小数ノイズは水平面として扱う。
        z_tolerance = max(1e-6, size_z * 1e-6)
        if span_z <= z_tolerance:
            z_value = ring[0][2]
            keys.update(self._voxelize_horizontal_ring(ring, z_value))
            return keys

        # 非水平ポリゴン（Z値が2個以上）はすべてedge-only方式で処理。
        # 壁面・斜面・階段状面いずれもXY投影に沿ったZ境界のみにvoxelを配置し、
        # VTK voxelizationによる中間z-levelへの全高fill を回避する。
        z_unique = set()
        for pt in ring:
            z_unique.add(round(pt[2], 6))
        if len(z_unique) >= 2:
            keys.update(self._voxelize_wall_edges(ring, z_unique, size_x, size_y, size_z))
            return keys

        # triangulate ベースの3D経路は、x一定/y一定の面（XY投影が線）を扱えない。
        if span_x <= 0 or span_y <= 0:
            logger.debug('degenerate bbox; skipping: %s', props.get('gml_id'))
            return keys

        if span_x <= size_x and span_y <= size_y and span_z <= size_z:
            key = self.encode_key(ix_min, iy_min, iz_min)
            keys.add(key)
            return keys
        offset_x = ix_min * size_x
        offset_y = iy_min * size_y
        offset_z = iz_min * size_z
        rel_geom_vtk = self.build_geom_3d([
            [
                x - offset_x,
                y - offset_y,
                z - offset_z
            ]
            for x, y, z in ring
        ])
        voxel_indices = None
        try:
            rx_count = ix_max - ix_min + 1
            ry_count = iy_max - iy_min + 1
            rz_count = iz_max - iz_min + 1
            voxel_indices = self._voxelize_to_indices(
                rel_geom_vtk,
                size_x,
                size_y,
                size_z,
                rx_count,
                ry_count,
                rz_count
            )
        except Exception as e:
            logger.debug(f'voxelize failed: {e}')
        if voxel_indices:
            voxel_indices = np.asarray(voxel_indices, dtype=np.int64)
            ix_arr = ix_min + voxel_indices[:, 0]
            iy_arr = iy_min + voxel_indices[:, 1]
            iz_arr = iz_min + voxel_indices[:, 2]
            keys.update(self._encode_keys_vectorized(ix_arr, iy_arr, iz_arr, level=self.level))
            return keys

        # fallback: collision check per voxel (slower)
        rel_bounds = rel_geom_vtk.bounds
        collision_filter = self.build_collision_filter(rel_geom_vtk)
        voxel_geom_vtk = self.build_box_3d(
            cx=size_x/2,
            cy=size_y/2,
            cz=size_z/2,
            sx=size_x,
            sy=size_y,
            sz=size_z
        )
        matrix = vtk.vtkMatrix4x4()
        collision_filter.SetInputData(1, voxel_geom_vtk)
        collision_filter.SetMatrix(1, matrix)
        for rx in range(ix_max - ix_min + 1):
            for ry in range(iy_max - iy_min + 1):
                for rz in range(iz_max - iz_min + 1):
                    voxel_minx = rx * size_x
                    voxel_maxx = voxel_minx + size_x
                    voxel_miny = ry * size_y
                    voxel_maxy = voxel_miny + size_y
                    voxel_minz = rz * size_z
                    voxel_maxz = voxel_minz + size_z
                    if (voxel_minx >= rel_bounds[1] or
                        voxel_maxx <= rel_bounds[0] or
                        voxel_miny >= rel_bounds[3] or
                        voxel_maxy <= rel_bounds[2] or
                        voxel_minz >= rel_bounds[5] or
                        voxel_maxz <= rel_bounds[4]):
                        continue
                    matrix.SetElement(0, 3, rx * size_x)
                    matrix.SetElement(1, 3, ry * size_y)
                    matrix.SetElement(2, 3, rz * size_z)
                    collision_filter.Update()
                    n_collision = collision_filter.GetNumberOfContacts()
                    if n_collision > 0:
                        key = self.encode_key(
                            ix_min + rx,
                            iy_min + ry,
                            iz_min + rz
                        )
                        keys.add(key)
                    else:
                        voxel_bounds = (
                            voxel_minx,
                            voxel_maxx,
                            voxel_miny,
                            voxel_maxy,
                            voxel_minz,
                            voxel_maxz
                        )
                        if (rel_bounds[0] >= voxel_bounds[0] and
                            rel_bounds[1] <= voxel_bounds[1] and
                            rel_bounds[2] >= voxel_bounds[2] and
                            rel_bounds[3] <= voxel_bounds[3] and
                            rel_bounds[4] >= voxel_bounds[4] and
                            rel_bounds[5] <= voxel_bounds[5]):

                            key = self.encode_key(
                                ix_min + rx,
                                iy_min + ry,
                                iz_min + rz
                            )
                            keys.add(key)

        return keys

    def _voxelize_horizontal_ring(self, ring: List[List[float]], z_value: float) -> Set[str]:
        keys: Set[str] = set()
        size_x = self.size_x
        size_y = self.size_y
        size_z = self.size_z

        coords_2d = [[x, y] for x, y, _ in ring]
        abs_geom_geos = geos.polygons(coords_2d)
        bounds = geos.bounds(abs_geom_geos)
        ix_min = math.floor(bounds[0] / size_x)
        ix_max = math.ceil(bounds[2] / size_x)
        iy_min = math.floor(bounds[1] / size_y)
        iy_max = math.ceil(bounds[3] / size_y)
        offset_x = ix_min * size_x
        offset_y = iy_min * size_y

        rel_coords_2d = [[x - offset_x, y - offset_y] for x, y, _ in ring]
        rel_geom_geos = geos.polygons(rel_coords_2d)
        geos.prepare(rel_geom_geos)
        rel_bounds = geos.bounds(rel_geom_geos)

        rx_count = ix_max - ix_min + 1
        ry_count = iy_max - iy_min + 1
        rxs = np.arange(rx_count, dtype=np.int64)
        rys = np.arange(ry_count, dtype=np.int64)
        grid_rx, grid_ry = np.meshgrid(rxs, rys, indexing='ij')
        voxel_minx = size_x * grid_rx
        voxel_miny = size_y * grid_ry
        voxel_maxx = voxel_minx + size_x
        voxel_maxy = voxel_miny + size_y
        mask_bounds = (
            (voxel_minx < rel_bounds[2]) &
            (voxel_maxx > rel_bounds[0]) &
            (voxel_miny < rel_bounds[3]) &
            (voxel_maxy > rel_bounds[1])
        )
        if not mask_bounds.any():
            return keys

        boxes = geos.box(
            voxel_minx[mask_bounds],
            voxel_miny[mask_bounds],
            voxel_maxx[mask_bounds],
            voxel_maxy[mask_bounds]
        )
        intersects = geos.intersects(rel_geom_geos, boxes)
        if isinstance(intersects, np.ndarray):
            hits = intersects
        else:
            hits = np.asarray(intersects, dtype=bool)
        if not hits.any():
            return keys

        hit_indices = np.argwhere(mask_bounds)
        hit_indices = hit_indices[hits]
        ix_arr = ix_min + hit_indices[:, 0]
        iy_arr = iy_min + hit_indices[:, 1]
        iz_arr = np.full(ix_arr.shape, math.floor(z_value / size_z), dtype=np.int64)
        keys.update(self._encode_keys_vectorized(ix_arr, iy_arr, iz_arr, level=self.level))
        return keys

    def _edge_z_indices_from_unique(
        self,
        z_unique: Set[float],
        size_z: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> List[int]:
        """壁面voxelに使うz-indexを決定する（上端/下端のみ）。"""
        if not z_unique:
            return []
        ring_z_min = min(z_unique)
        ring_z_max = max(z_unique)
        iz_min = math.floor(ring_z_min / size_z)
        iz_max = math.floor(ring_z_max / size_z)
        if z_min is not None:
            iz_min = max(iz_min, math.floor(z_min / size_z))
        if z_max is not None:
            iz_max = min(iz_max, math.floor(z_max / size_z))
        if iz_min > iz_max:
            return []
        if iz_min == iz_max:
            return [iz_min]
        return [iz_min, iz_max]

    def _full_span_z_indices_from_unique(
        self,
        z_unique: Set[float],
        size_z: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> List[int]:
        """壁面voxelに使うz-indexを連続レンジで返す。"""
        if not z_unique:
            return []
        iz_min = math.floor(min(z_unique) / size_z)
        iz_max = math.floor(max(z_unique) / size_z)
        if z_min is not None:
            iz_min = max(iz_min, math.floor(z_min / size_z))
        if z_max is not None:
            iz_max = min(iz_max, math.floor(z_max / size_z))
        if iz_min > iz_max:
            return []
        return list(range(iz_min, iz_max + 1))

    def _voxelize_wall_edges(self, ring: List[List[float]], z_unique: Set[float],
                             size_x: float, size_y: float, size_z: float,
                             *,
                             full_span: bool = False,
                             z_min: Optional[float] = None,
                             z_max: Optional[float] = None) -> Set[str]:
        """非水平面の壁エッジをZ境界（上端/下端）のみに配置する。

        壁面のXY投影はlinestring（線）になるため、
        linestring-box交差判定でヒットした(x,y)セルに対して
        壁面のZ境界（上端・下端）または全高さにvoxelを配置する。
        """
        keys: Set[str] = set()
        coords_2d = [[x, y] for x, y, _ in ring]
        line_geom = geos.linestrings(coords_2d)
        line_bounds = geos.bounds(line_geom)
        lix_min = math.floor(line_bounds[0] / size_x)
        lix_max = math.ceil(line_bounds[2] / size_x)
        liy_min = math.floor(line_bounds[1] / size_y)
        liy_max = math.ceil(line_bounds[3] / size_y)
        rx_count = lix_max - lix_min + 1
        ry_count = liy_max - liy_min + 1
        if rx_count <= 0 or ry_count <= 0:
            return keys
        offset_x = lix_min * size_x
        offset_y = liy_min * size_y
        rel_coords_2d = [[x - offset_x, y - offset_y] for x, y, _ in ring]
        rel_line = geos.linestrings(rel_coords_2d)
        geos.prepare(rel_line)
        rel_line_bounds = geos.bounds(rel_line)
        rxs = np.arange(rx_count, dtype=np.int64)
        rys = np.arange(ry_count, dtype=np.int64)
        grid_rx, grid_ry = np.meshgrid(rxs, rys, indexing='ij')
        voxel_minx = size_x * grid_rx
        voxel_miny = size_y * grid_ry
        voxel_maxx = voxel_minx + size_x
        voxel_maxy = voxel_miny + size_y
        mask = (
            (voxel_minx < rel_line_bounds[2]) &
            (voxel_maxx > rel_line_bounds[0]) &
            (voxel_miny < rel_line_bounds[3]) &
            (voxel_maxy > rel_line_bounds[1])
        )
        if not mask.any():
            return keys
        boxes = geos.box(
            voxel_minx[mask], voxel_miny[mask],
            voxel_maxx[mask], voxel_maxy[mask]
        )
        intersects = geos.intersects(rel_line, boxes)
        hits = np.asarray(intersects, dtype=bool) if not isinstance(intersects, np.ndarray) else intersects
        if not hits.any():
            return keys
        hit_idx = np.argwhere(mask)[hits]
        ix_arr = lix_min + hit_idx[:, 0]
        iy_arr = liy_min + hit_idx[:, 1]
        z_indices = (
            self._full_span_z_indices_from_unique(z_unique, size_z, z_min=z_min, z_max=z_max)
            if full_span
            else self._edge_z_indices_from_unique(z_unique, size_z, z_min=z_min, z_max=z_max)
        )
        if not z_indices:
            return keys
        for iz in z_indices:
            iz_arr = np.full(ix_arr.shape, iz, dtype=np.int64)
            keys.update(self._encode_keys_vectorized(ix_arr, iy_arr, iz_arr, level=self.level))
        return keys

    def _update_by_geom_2d(self, data: Set[str],
                coordinates: List[List[List[float]]],
                props: Dict[str, Any]) -> None:
        size_x = self.size_x
        size_y = self.size_y

        is_line = props.get('geom_line') or False

        # target
        abs_geom_geos = self.build_geom_2d(coordinates, is_line=is_line)

        bounds = geos.bounds(abs_geom_geos)
        ix_min = math.floor(bounds[0] / size_x)
        ix_max = math.ceil(bounds[2] / size_x)
        iy_min = math.floor(bounds[1] / size_y)
        iy_max = math.ceil(bounds[3] / size_y)
        offset_x = ix_min * size_x
        offset_y = iy_min * size_y
        rel_coordinates = [
            [
                [
                    x - offset_x,
                    y - offset_y,
                ]
                for x, y, _ in part
            ]
            for part in coordinates
        ]
        rel_geom_geos = self.build_geom_2d(rel_coordinates, is_line=is_line)
        geos.prepare(rel_geom_geos)
        rel_bounds = geos.bounds(rel_geom_geos)
        rx_count = ix_max - ix_min + 1
        ry_count = iy_max - iy_min + 1
        rxs = np.arange(rx_count, dtype=np.int64)
        rys = np.arange(ry_count, dtype=np.int64)
        grid_rx, grid_ry = np.meshgrid(rxs, rys, indexing='ij')
        voxel_minx = size_x * grid_rx
        voxel_miny = size_y * grid_ry
        voxel_maxx = voxel_minx + size_x
        voxel_maxy = voxel_miny + size_y
        mask_bounds = (
            (voxel_minx < rel_bounds[2]) &
            (voxel_maxx > rel_bounds[0]) &
            (voxel_miny < rel_bounds[3]) &
            (voxel_maxy > rel_bounds[1])
        )
        if not mask_bounds.any():
            return
        boxes = geos.box(
            voxel_minx[mask_bounds],
            voxel_miny[mask_bounds],
            voxel_maxx[mask_bounds],
            voxel_maxy[mask_bounds]
        )
        intersects = geos.intersects(rel_geom_geos, boxes)
        if isinstance(intersects, np.ndarray):
            hits = intersects
        else:
            # pygeos <0.12 may return list
            hits = np.asarray(intersects, dtype=bool)
        if not hits.any():
            return
        hit_indices = np.argwhere(mask_bounds)
        if hits.any():
            hit_indices = hit_indices[hits]
            ix_arr = ix_min + hit_indices[:, 0]
            iy_arr = iy_min + hit_indices[:, 1]
            keys = self._encode_keys_vectorized(ix_arr, iy_arr, None, level=self.level)
            data.update(keys)

    def build_collision_filter(self, geom: vista.PolyData
                              ) -> vtk.vtkCollisionDetectionFilter:
        alg = vtk.vtkCollisionDetectionFilter()
        alg.SetInputData(0, geom)
        alg.SetTransform(0, vtk.vtkTransform())
        alg.SetBoxTolerance(0.001)
        alg.SetCellTolerance(0.0)
        alg.SetNumberOfCellsPerNode(2)
        alg.SetCollisionMode(1)
        alg.SetGenerateScalars(False)
        return alg

    def _voxelize_to_indices(self, geom: vista.PolyData,
                             size_x: float, size_y: float, size_z: float,
                             nx: int, ny: int, nz: int
                             ) -> List[Tuple[int, int, int]]:
        """Fast voxelization using pyvista or vtk voxel modeller."""
        # path1: pyvista voxelize
        if hasattr(geom, 'voxelize'):
            vox = geom.voxelize(density=(size_x, size_y, size_z))
            scalars = vox.cell_data.active_scalars
            if scalars is None or len(scalars) == 0:
                scalars = vox.point_data.active_scalars
            if scalars is None or len(scalars) == 0:
                return None
            dims = vox.dimensions
            cell_dims = (max(dims[0]-1, 0), max(dims[1]-1, 0), max(dims[2]-1, 0))
            arr = np.asarray(scalars)
            if arr.size == cell_dims[0] * cell_dims[1] * cell_dims[2]:
                arr = arr.reshape(cell_dims, order='F')
            else:
                arr = arr.reshape(dims, order='F')[:-1, :-1, :-1]
            indices = np.argwhere(arr > 0)
            return [tuple(map(int, idx.tolist())) for idx in indices]

        # path2: vtkVoxelModeller
        modeller = vtk.vtkVoxelModeller()
        modeller.SetSampleDimensions(int(nx), int(ny), int(nz))
        modeller.SetModelBounds(0, nx * size_x, 0, ny * size_y, 0, nz * size_z)
        modeller.SetScalarTypeToUnsignedChar()
        modeller.SetInputData(geom)
        modeller.Update()
        image = modeller.GetOutput()
        dims = image.GetDimensions()
        scalars = image.GetPointData().GetScalars()
        if scalars is None:
            return None
        arr = vtknp.vtk_to_numpy(scalars)
        arr = arr.reshape(dims, order='F')
        # Convert point grid to voxel occupancy
        arr = arr[:-1, :-1, :-1]
        indices = np.argwhere(arr > 0)
        return [tuple(map(int, idx.tolist())) for idx in indices]

    def build_geom_3d(self, ring: List[List[float]]) -> vista.PolyData:
        if len(ring) > 1 and ring[0] == ring[-1]:
            points = ring[:-1]
        else:
            points = ring
        faces = [len(points)] + list(range(len(points)))
        geom = vista.PolyData(points, faces)
        geom.points_to_double()
        geom.triangulate(inplace=True)
        return geom

    def build_box_3d(self, cx: float, cy: float, cz: float,
                     sx: float, sy: float, sz: float) -> vista.PolyData:
        geom = vista.Cube(
            x_length=sx,
            y_length=sy,
            z_length=sz
        )
        geom.points_to_double()
        geom.translate((cx, cy, cz), inplace=True)
        geom.triangulate(inplace=True)
        return geom

    def build_geom_2d(self, coordinates: List[List[List[float]]],
                      is_line: bool = False) -> geos.Geometry:
        if is_line:
            geom = geos.linestrings(coordinates[0])
        else:
            interiors = [
                geos.linearrings(c)
                for c in coordinates[1:]
            ] or None
            geom = geos.polygons(coordinates[0], holes=interiors)
        return geom

    def build_box_2d(self, minx: float, miny: float,
                     maxx: float, maxy: float) -> geos.Geometry:
        geom = geos.box(minx, miny, maxx, maxy)
        return geom

    def interpolate(self) -> None:
        for _gml_id, data in self._data.items():
            _before = len(data)
            self._infill_solid_improved(data)
            _after = len(data)
            if _before > 50:
                logger.info('interpolate %s: %d -> %d (+%d)',
                            _gml_id, _before, _after, _after - _before)

    def merge(self) -> None:
        pass

    def fill_solid_interior(self) -> None:
        """Fill interior voxels of closed solids.

        For each gml_id, finds the min/max z for each (x,y) column
        and fills all intermediate z-levels. This converts surface-only
        voxels into solid filled volumes.
        """
        for data in self._data.values():
            self._infill_vertical_columns(data)

    def load_id_data(self, data: Iterator[Tuple[str, str, Any]],
                     level: int = None, decompose: bool = False) -> None:
        for gml_id, spatial_id, property in data:
            self._update_by_id(gml_id, spatial_id, property, level, decompose)

    def _update_by_id(self, gml_id: str, spatial_id: str, property: Any = None,
                      level: int = None, decompose: bool = False) -> None:
        data = self._data.setdefault(gml_id, set())
        data.add(spatial_id)

    def extract_ids(self, xml: etree.ElementTree) -> None:
        spatial_id_elms = xml.findall(
            '//{*}spatialID'
        )
        for spatial_id_elm in spatial_id_elms:
            level_elm = spatial_id_elm.getparent().find('{*}maxZoomLevel')
            if level_elm is not None:
                level = int(level_elm.text)
                decompose = True
            else:
                level = None
                decompose = False
            spatial_id = spatial_id_elm.text
            feature_elm = spatial_id_elm.getparent().getparent().getparent()
            gml_id = feature_elm.attrib['{http://www.opengis.net/gml}id']
            self._update_by_id(gml_id, spatial_id, level=level, decompose=decompose)

    def extrude(self, min_extrude: float, max_extrude: float) -> None:
        pass


class ZFXYGrid(Grid):

    lv_0_x: float = 2 * math.pi * 6378137.0
    lv_0_y: float = lv_0_x
    lv_0_z: float = 2 ** 25

    crs: int = 3857
    output_crs: int = 6668

    level: int = 20

    def __init__(self, *args, level: int = None, **kwargs):
        super().__init__(*args, level=level, **kwargs)
        if level is not None:
            self.level = level
        self.size_x = self.lv_0_x / 2 ** self.level
        self.size_y = self.lv_0_y / 2 ** self.level
        self.size_z = self.lv_0_z / 2 ** self.level

    def encode_key(self, ix: int, iy: int, iz: int, level: int = None) -> str:
        if level is None:
            level = self.level
        gix, giy, giz = self._get_grid_index(ix, iy, iz=iz, level=level)
        if giz is None:
            return f'{level}/{gix}/{giy}'
        else:
            return f'{level}/{giz}/{gix}/{giy}'

    def encode_key_simple(self, gix: int, giy: int, giz: int = None, level: int = None) -> str:
        if level is None:
            level = self.level
        if giz is None:
            return f'{level}/{gix}/{giy}'
        else:
            return f'{level}/{giz}/{gix}/{giy}'

    def decode_key(self, key: str, full: bool = False) -> str:
        components = [int(s) for s in key.split('/')]
        if len(components) == 4:
            level, giz, gix, giy = components
        elif len(components) == 3:
            level, gix, giy = components
            giz = None
        else:
            raise ValueError(f'Invalid key: {key}')
        ix, iy, iz = self._get_coordinate_index(gix, giy, giz, level)
        if full:
            return (ix, iy, iz, level)
        else:
            return (ix, iy, iz)

    def decode_key_simple(self, key: str, full: bool = False) -> str:
        components = [int(s) for s in key.split('/')]
        if len(components) == 4:
            level, giz, gix, giy = components
        elif len(components) == 3:
            level, gix, giy = components
            giz = None
        else:
            raise ValueError(f'Invalid key: {key}')
        if full:
            return (gix, giy, giz, level)
        else:
            return (gix, giy, giz)

    def _get_grid_index(self, ix: int, iy: int, iz: int = 0, level: int = None) -> Tuple[int, int, int]:
        if level is None:
            level = self.level
        gix = (ix + (1 << max(level - 1, 0))) % (1 << level)
        giy = (1 << max(level - 1, 0)) - iy - 1
        giz = iz
        return gix, giy, giz

    def _get_coordinate_index(self, gix: int, giy: int, giz: int = 0, level: int = None) -> Tuple[int, int, int]:
        if level is None:
            level = self.level
        ix = gix - int(2 ** (level - 1))
        iy = (1 << max(level - 1, 0)) - giy - 1
        iz = giz
        return ix, iy, iz

    def merge(self) -> None:
        for data in self._data.values():
            self._merge_sub(data)

    def _merge_sub(self, data: Set[str]) -> None:
        normal_data = set()
        parent_data = set()
        for key1 in data:
            gix1, giy1, giz1, lv = self.decode_key_simple(key1, full=True)
            if lv == 0:
                continue
            is_3d = giz1 is not None
            if is_3d:
                # parent
                px = gix1 >> 1
                py = giy1 >> 1
                pz = giz1 >> 1
                pkey = self.encode_key_simple(px, py, pz, level=lv-1)
                if pkey in parent_data:
                    continue
                # sibling
                gix2 = gix1 ^ 1
                giy2 = giy1 ^ 1
                giz2 = giz1 ^ 1
                skeys = [
                    self.encode_key_simple(gix1, giy1, giz1, level=lv),
                    self.encode_key_simple(gix1, giy1, giz2, level=lv),
                    self.encode_key_simple(gix1, giy2, giz1, level=lv),
                    self.encode_key_simple(gix1, giy2, giz2, level=lv),
                    self.encode_key_simple(gix2, giy1, giz1, level=lv),
                    self.encode_key_simple(gix2, giy1, giz2, level=lv),
                    self.encode_key_simple(gix2, giy2, giz1, level=lv),
                    self.encode_key_simple(gix2, giy2, giz2, level=lv),
                ]
                if all(skey in data for skey in skeys):
                    parent_data.add(pkey)
                else:
                    normal_data.add(key1)
            else:
                # parent
                px = gix1 >> 1
                py = giy1 >> 1
                pkey = self.encode_key_simple(px, py, level=lv-1)
                if pkey in parent_data:
                    continue
                # sibling
                gix2 = gix1 ^ 1
                giy2 = giy1 ^ 1
                skeys = [
                    self.encode_key_simple(gix1, giy1, level=lv),
                    self.encode_key_simple(gix1, giy2, level=lv),
                    self.encode_key_simple(gix2, giy1, level=lv),
                    self.encode_key_simple(gix2, giy2, level=lv),
                ]
                if all(skey in data for skey in skeys):
                    parent_data.add(pkey)
                else:
                    normal_data.add(key1)
        data.clear()
        if parent_data:
            self._merge_sub(parent_data)
        data.update(parent_data)
        data.update(normal_data)

    def _update_by_id(self, gml_id: str, spatial_id: str, property: Any = None,
                      level: int = None, decompose: bool = False) -> None:
        key = spatial_id
        ix, iy, iz, lv = self.decode_key(key, full=True)
        is_3d = iz is not None
        temp_data = {}
        if not decompose or lv == level:
            temp_data.setdefault(key, set()).add(gml_id)
        elif lv < level:
            # decompose
            temp_list = [(ix, iy, iz, lv)]
            if is_3d:
                for _ in range(level - lv):
                    temp_list = itertools.chain.from_iterable(
                        [
                            (_ix << 1    , _iy << 1    , _iz << 1    , _lv + 1),
                            (_ix << 1    , _iy << 1    , _iz << 1 ^ 1, _lv + 1),
                            (_ix << 1    , _iy << 1 ^ 1, _iz << 1    , _lv + 1),
                            (_ix << 1    , _iy << 1 ^ 1, _iz << 1 ^ 1, _lv + 1),
                            (_ix << 1 ^ 1, _iy << 1    , _iz << 1    , _lv + 1),
                            (_ix << 1 ^ 1, _iy << 1    , _iz << 1 ^ 1, _lv + 1),
                            (_ix << 1 ^ 1, _iy << 1 ^ 1, _iz << 1    , _lv + 1),
                            (_ix << 1 ^ 1, _iy << 1 ^ 1, _iz << 1 ^ 1, _lv + 1),
                        ]
                        for _ix, _iy, _iz, _lv in temp_list
                    )
            else:
                for _ in range(level - lv):
                    temp_list = itertools.chain.from_iterable(
                        [
                            (_ix << 1    , _iy << 1    , _iz    , _lv + 1),
                            (_ix << 1    , _iy << 1 ^ 1, _iz    , _lv + 1),
                            (_ix << 1 ^ 1, _iy << 1    , _iz    , _lv + 1),
                            (_ix << 1 ^ 1, _iy << 1 ^ 1, _iz    , _lv + 1),
                        ]
                        for _ix, _iy, _iz, _lv in temp_list
                    )
            for _ix, _iy, _iz, _lv in temp_list:
                temp_key = self.encode_key(_ix, _iy, _iz, _lv)
                temp_data.setdefault(temp_key, set()).add(gml_id)
        else:
            # compose
            n = lv - level
            temp_key = self.encode_key(
                ix >> n,
                iy >> n,
                iz >> n if is_3d else None,
                level
            )
            temp_data.setdefault(temp_key, set()).add(gml_id)
        data = self._data.setdefault(gml_id, set())
        for sid in temp_data.keys():
            data.add(sid)

    def extrude(self, min_extrude: float = None, max_extrude: float = None) -> None:
        if min_extrude is None or max_extrude is None:
            return
        new_data = {}
        iz_min = math.floor(min_extrude / self.size_z)
        iz_max = math.ceil(max_extrude / self.size_z)
        iz_max_extra = 1 if float(iz_max) == (max_extrude / self.size_z) else 0
        for gml_id, sid_set in self.data.items():
            new_sid_set = new_data.setdefault(gml_id, set())
            for sid in sid_set:
                ix, iy, _iz, level = self.decode_key(sid, full=True)
                if _iz is not None:
                    # 3D
                    new_sid_set.add(sid)
                else:
                    # 2D
                    for iz in range(iz_min, iz_max + iz_max_extra):
                        new_sid = self.encode_key(ix, iy, iz, level)
                        new_sid_set.add(new_sid)
        self._data = new_data
