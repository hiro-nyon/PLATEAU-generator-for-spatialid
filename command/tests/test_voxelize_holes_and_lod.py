from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_grids_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "grids.py"
    )
    spec = importlib.util.spec_from_file_location("plateau_spatialid_grids", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load grids.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_prepare_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "prepare.py"
    )
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("plateau_spatialid_prepare", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load prepare.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ring(x0: float, x1: float, y0: float, y1: float, z: float) -> list[list[float]]:
    return [
        [x0, y0, z],
        [x1, y0, z],
        [x1, y1, z],
        [x0, y1, z],
        [x0, y0, z],
    ]


def test_fill_solid_keeps_hole_columns_empty() -> None:
    grids = _load_grids_module()
    grid = grids.ZFXYGrid(level=25)
    sx = grid.size_x
    sy = grid.size_y

    outer_bottom = _ring(0.1 * sx, 3.9 * sx, 0.1 * sy, 3.9 * sy, -3.0)
    hole_bottom = _ring(1.1 * sx, 2.9 * sx, 1.1 * sy, 2.9 * sy, -3.0)
    outer_top = _ring(0.1 * sx, 3.9 * sx, 0.1 * sy, 3.9 * sy, -1.0)
    hole_top = _ring(1.1 * sx, 2.9 * sx, 1.1 * sy, 2.9 * sy, -1.0)

    props = {
        "gml_id": "ubld-hole",
        "geom_dim": 3,
        "geom_line": False,
    }
    geom_data = [
        ([outer_bottom, hole_bottom], props),
        ([outer_top, hole_top], props),
    ]
    grid.load_geom_data(iter(geom_data), fill_solid=True)

    decoded = {grid.decode_key(spatial_id) for spatial_id in grid.data["ubld-hole"]}

    # infillは維持される
    assert (0, 0, -2) in decoded
    # hole列は z=-3,-2,-1 の全てで空になる
    for ix in (1, 2):
        for iy in (1, 2):
            for iz in (-3, -2, -1):
                assert (ix, iy, iz) not in decoded


def test_voxelize_wire_3d_hits_endpoints() -> None:
    grids = _load_grids_module()
    grid = grids.ZFXYGrid(level=25)
    sx = grid.size_x
    sy = grid.size_y

    wire = [
        [0.2 * sx, 0.2 * sy, -3.0],
        [3.8 * sx, 0.2 * sy, -1.0],
    ]
    keys = grid._voxelize_wire_3d(wire, {"gml_id": "wire"})
    decoded = {grid.decode_key(spatial_id) for spatial_id in keys}

    assert (0, 0, -3) in decoded
    assert (3, 0, -1) in decoded
    assert len(decoded) >= 4


def test_update_by_geom_3d_accepts_nested_line_coordinates() -> None:
    grids = _load_grids_module()
    grid = grids.ZFXYGrid(level=25)
    sx = grid.size_x
    sy = grid.size_y

    nested_coords = [[
        [0.2 * sx, 0.2 * sy, -5.0],
        [1.8 * sx, 0.4 * sy, -3.0],
    ]]
    data: set[str] = set()
    grid._update_by_geom_3d(
        data,
        [nested_coords],
        {"gml_id": "nested-wire", "geom_line": True},
    )
    assert len(data) > 0


def test_arrange_path_supports_lod4() -> None:
    prepare = _load_prepare_module()
    lod, path = prepare.arrange_path("lod4Solid//surfaceMember")
    assert lod == 4
    assert "{*}lod4Solid" in path


def test_geometry_paths_include_lod4_for_ubld() -> None:
    prepare = _load_prepare_module()
    prepare.GEOMETRY_PATHS = None
    paths = prepare.get_geometry_paths("UndergroundBuilding")
    assert paths is not None
    assert 4 in paths
    lod4_paths = {geom_path for geom_path, _ in paths[4]}
    assert any("lod4Solid" in geom_path for geom_path in lod4_paths)
