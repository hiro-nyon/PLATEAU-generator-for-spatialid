#!/usr/bin/env python3
"""
Diagnostic: Compare GML polygon footprint with generated voxel positions.
Uses only stdlib + pyproj (no lxml).
"""
import sys
import math
import re

def main():
    gml_file = sys.argv[1] if len(sys.argv) > 1 else None
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not gml_file:
        print("Usage: python diagnose_ubld.py <gml_file> [csv_file]")
        sys.exit(1)

    # --- 1. Scan GML for coordinate bounds ---
    print("=" * 60)
    print("1. GML Coordinate Analysis (regex scan)")
    print("=" * 60)

    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')
    z_values = set()
    n_pos_lists = 0
    n_horiz = 0
    n_wall = 0

    pos_pattern = re.compile(r'<gml:posList>(.*?)</gml:posList>', re.DOTALL)

    with open(gml_file, encoding='utf-8') as f:
        buf = ''
        for line in f:
            buf += line
            if '</gml:posList>' in line:
                for m in pos_pattern.finditer(buf):
                    text = m.group(1).strip()
                    coords = text.split()
                    if len(coords) < 3 or len(coords) % 3 != 0:
                        continue
                    n_pos_lists += 1
                    local_z = set()
                    for i in range(0, len(coords), 3):
                        lat = float(coords[i])
                        lon = float(coords[i + 1])
                        z = float(coords[i + 2])
                        lat_min = min(lat_min, lat)
                        lat_max = max(lat_max, lat)
                        lon_min = min(lon_min, lon)
                        lon_max = max(lon_max, lon)
                        local_z.add(round(z, 3))
                    z_values.update(local_z)
                    if len(local_z) == 1:
                        n_horiz += 1
                    else:
                        n_wall += 1
                buf = ''

    print(f"  Total posList elements: {n_pos_lists}")
    print(f"  Surfaces: {n_horiz} horizontal, {n_wall} wall/non-horizontal")
    print(f"  Lat range: {lat_min:.8f} — {lat_max:.8f}")
    print(f"  Lon range: {lon_min:.8f} — {lon_max:.8f}")
    print(f"  Z values: {sorted(z_values)}")

    # --- 2. Transform like prepare.py ---
    print("\n" + "=" * 60)
    print("2. Coordinate Transform (same as prepare.py)")
    print("=" * 60)

    try:
        import pyproj as proj
        has_pyproj = True
    except ImportError:
        has_pyproj = False
        print("  [WARN] pyproj not available; using manual Web Mercator transform")

    if has_pyproj:
        crs_from = proj.CRS.from_epsg(4326)
        crs_to = proj.CRS.from_epsg(3857)
        transformer = proj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)
    else:
        # Manual Web Mercator
        def to_3857(lon, lat):
            x = lon * 20037508.342789244 / 180.0
            y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
            y = y * 20037508.342789244 / 180.0
            return x, y
        x_min, y_min = to_3857(lon_min, lat_min)
        x_max, y_max = to_3857(lon_max, lat_max)

    print(f"  EPSG:3857 bbox:")
    print(f"    x: {x_min:.2f} — {x_max:.2f}")
    print(f"    y: {y_min:.2f} — {y_max:.2f}")

    # --- 3. Compute expected voxel indices ---
    print("\n" + "=" * 60)
    print("3. Expected Voxel Indices (level 25)")
    print("=" * 60)

    lv_0 = 2 * math.pi * 6378137.0
    level = 25
    size_xy = lv_0 / (2 ** level)
    size_z = (2 ** 25) / (2 ** level)

    print(f"  Voxel size: {size_xy:.6f} m x {size_xy:.6f} m x {size_z:.6f} m")

    ix_min_v = math.floor(x_min / size_xy)
    ix_max_v = math.ceil(x_max / size_xy)
    iy_min_v = math.floor(y_min / size_xy)
    iy_max_v = math.ceil(y_max / size_xy)

    print(f"  ix range: {ix_min_v} — {ix_max_v} ({ix_max_v - ix_min_v} cells)")
    print(f"  iy range: {iy_min_v} — {iy_max_v} ({iy_max_v - iy_min_v} cells)")

    # Convert to gix/giy (ZFXY grid)
    half = 1 << max(level - 1, 0)
    gix_min_v = (ix_min_v + half) % (1 << level)
    gix_max_v = (ix_max_v + half) % (1 << level)
    giy_min_v = half - iy_max_v - 1
    giy_max_v = half - iy_min_v - 1

    print(f"  Expected gix (x in ZFXY): {gix_min_v} — {gix_max_v}")
    print(f"  Expected giy (y in ZFXY): {giy_min_v} — {giy_max_v}")

    for z_val in sorted(z_values):
        iz = math.floor(z_val / size_z)
        print(f"  z={z_val} → f={iz}")

    # --- 4. Analyze CSV output ---
    if csv_file:
        print("\n" + "=" * 60)
        print("4. CSV Voxel Analysis")
        print("=" * 60)

        csv_gix_min, csv_gix_max = float('inf'), float('-inf')
        csv_giy_min, csv_giy_max = float('inf'), float('-inf')
        csv_f_values = set()
        n_rows = 0

        with open(csv_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('PLATEAU') or line.startswith('gml_id'):
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                sid = parts[1]
                comps = sid.split('/')
                if len(comps) == 4:
                    fz, gx, gy = int(comps[1]), int(comps[2]), int(comps[3])
                elif len(comps) == 3:
                    gx, gy = int(comps[1]), int(comps[2])
                    fz = None
                else:
                    continue

                n_rows += 1
                csv_gix_min = min(csv_gix_min, gx)
                csv_gix_max = max(csv_gix_max, gx)
                csv_giy_min = min(csv_giy_min, gy)
                csv_giy_max = max(csv_giy_max, gy)
                if fz is not None:
                    csv_f_values.add(fz)

        print(f"  Total voxel rows: {n_rows}")
        print(f"  CSV gix (x) range: {csv_gix_min} — {csv_gix_max}")
        print(f"  CSV giy (y) range: {csv_giy_min} — {csv_giy_max}")
        print(f"  CSV f (z) values: min={min(csv_f_values)}, max={max(csv_f_values)}, count={len(csv_f_values)}")

        # Compare
        print("\n  --- Comparison ---")
        print(f"  Expected gix: {gix_min_v} — {gix_max_v}")
        print(f"  Actual   gix: {csv_gix_min} — {csv_gix_max}")
        gix_ok = (csv_gix_min >= gix_min_v - 1) and (csv_gix_max <= gix_max_v + 1)
        print(f"  X match: {'✓ OK' if gix_ok else '✗ MISMATCH!'}")

        print(f"  Expected giy: {giy_min_v} — {giy_max_v}")
        print(f"  Actual   giy: {csv_giy_min} — {csv_giy_max}")
        giy_ok = (csv_giy_min >= giy_min_v - 1) and (csv_giy_max <= giy_max_v + 1)
        print(f"  Y match: {'✓ OK' if giy_ok else '✗ MISMATCH!'}")

        # Reverse-geocode center voxel to lat/lon
        center_gix = (csv_gix_min + csv_gix_max) // 2
        center_giy = (csv_giy_min + csv_giy_max) // 2
        center_ix = center_gix - half
        center_iy = half - center_giy - 1
        center_x = (center_ix + 0.5) * size_xy
        center_y = (center_iy + 0.5) * size_xy

        if has_pyproj:
            transformer_inv = proj.Transformer.from_crs(crs_to, crs_from, always_xy=True)
            center_lon, center_lat = transformer_inv.transform(center_x, center_y)
        else:
            center_lon = center_x * 180.0 / 20037508.342789244
            center_lat = math.degrees(math.atan(math.exp(center_y * math.pi / 20037508.342789244)) * 2 - math.pi / 2)

        gml_center_lat = (lat_min + lat_max) / 2
        gml_center_lon = (lon_min + lon_max) / 2
        print(f"\n  Voxel center → lat={center_lat:.8f}, lon={center_lon:.8f}")
        print(f"  GML   center → lat={gml_center_lat:.8f}, lon={gml_center_lon:.8f}")

        dlat_m = (center_lat - gml_center_lat) * 111320
        dlon_m = (center_lon - gml_center_lon) * 111320 * math.cos(math.radians(gml_center_lat))
        dist = math.sqrt(dlat_m ** 2 + dlon_m ** 2)
        print(f"  Offset: Δlat={dlat_m:.1f}m, Δlon={dlon_m:.1f}m, total={dist:.1f}m")

        if dist > 10:
            print(f"\n  ⚠️  Significant spatial offset detected ({dist:.0f}m)!")
        else:
            print(f"\n  ✓ Spatial alignment looks reasonable ({dist:.1f}m)")


if __name__ == '__main__':
    main()
