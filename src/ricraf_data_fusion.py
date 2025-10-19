#!/usr/bin/env python3
"""
Utility functions for fusing road and climate datasets in the RICRAF project.
Supports `notebooks/ricraf_data_fusion.ipynb` and other project components.
Includes functions for road data merging (`fuse_road_data`) and road-climate integration (`merge_road_climate`).
"""


import geopandas as gpd
import pandas as pd
import numpy as np
import math
from shapely.geometry import LineString, MultiLineString
from typing import Optional, Tuple, List
from loguru import logger
import sys
import warnings
import os

# Turn off future warnings to keep the logs tidy
warnings.filterwarnings("ignore", category=FutureWarning)


def calc_travel_dir(geometry, precision: str = "cardinal") -> Optional[str]:
    """
    Works out which way a road is heading based on its shape.

    Args:
        geometry: The road shape (a single line or a bunch of lines).
        precision (str): Use 'cardinal' for directions like 'Northbound' (default) or 'degrees' for an angle like '45.0°'.

    Returns:
        str or None: The direction (e.g., 'Northbound' or '45.0°'), or None if it can’t figure it out.
    """
    try:
        geom_type = geometry.geom_type
    except AttributeError:
        logger.error("The road shape doesn’t have a type - something’s off!")
        return None

    if geom_type == 'LineString':
        coords = list(geometry.coords)
        if len(coords) < 2:
            logger.warning(f"Not enough points to work out direction: {len(coords)}")
            return None
        start_x, start_y = coords[0]
        end_x, end_y = coords[-1]
    elif geom_type == 'MultiLineString':
        total_length = 0
        weighted_bearing = 0
        for segment in geometry.geoms:
            coords = list(segment.coords)
            if len(coords) < 2:
                continue
            start_x, start_y = coords[0]
            end_x, end_y = coords[-1]
            delta_x = end_x - start_x
            delta_y = end_y - start_y
            if delta_x == 0 and delta_y == 0:
                continue
            bearing_rad = math.atan2(delta_x, delta_y)
            length = segment.length
            weighted_bearing += ((math.degrees(bearing_rad) + 360) % 360) * length
            total_length += length
        if total_length == 0:
            logger.warning("No decent bits in this multi-line road shape")
            return None
        bearing_deg = weighted_bearing / total_length
        bearing_deg = (bearing_deg + 360) % 360
        if precision == "degrees":
            return f"{bearing_deg:.1f}°"
        return _bearing_to_direction(bearing_deg)
    else:
        logger.warning(f"Can’t deal with this shape type: {geom_type}")
        return None

    if start_x == end_x and start_y == end_y:
        logger.warning("Start and end are the same spot - no direction to give!")
        return None

    bearing_rad = math.atan2(end_x - start_x, end_y - start_y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360
    return f"{bearing_deg:.1f}°" if precision == "degrees" else _bearing_to_direction(bearing_deg)


def _bearing_to_direction(bearing_deg: float) -> str:
    """
    Turns an angle into a simple direction name like 'Northbound' (used internally).

    Args:
        bearing_deg (float): Angle in degrees (0-360).

    Returns:
        str: A direction name like 'Northbound'.
    """
    directions = {
        (337.5, 22.5): "Northbound", (22.5, 67.5): "Northeast", (67.5, 112.5): "Eastbound",
        (112.5, 157.5): "Southeast", (157.5, 202.5): "Southbound", (202.5, 247.5): "Southwest",
        (247.5, 292.5): "Westbound", (292.5, 337.5): "Northwest"
    }
    for (lower, upper), direction in directions.items():
        if lower <= bearing_deg < upper or (lower > upper and (bearing_deg >= lower or bearing_deg < upper)):
            return direction
    return "Northbound"  # Backup if nothing fits


def shorten_road_name(name: str) -> str:
    """
    Cuts a road name down to size by dropping bits after 'btwn' and shortening the last word.

    Args:
        name (str): Full road name (e.g., 'Main Road btwn A and B').

    Returns:
        str: Short version (e.g., 'Main R').
    """
    road_part = name.split('btwn')[0].strip() if 'btwn' in name else name.strip()
    words = road_part.split()
    directions = {'EAST', 'WEST', 'NORTH', 'SOUTH', 'E', 'W', 'N', 'S'}
    if words and words[-1].upper() in directions:
        words = words[:-1]
    if len(words) > 1:
        base_name = ' '.join(words[:-1])
        return f"{base_name} {words[-1][0]}"
    return words[0]


def calculate_azimuth(geometry: LineString) -> float:
    """
    Figures out the angle (in degrees) of a single road line.

    Args:
        geometry (LineString): A single road line shape.

    Returns:
        float: Angle in degrees (0-360).
    """
    if not isinstance(geometry, LineString):
        raise ValueError(f"Needs a single line, got {type(geometry)} instead")
    coords = list(geometry.coords)
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    return np.degrees(np.arctan2(dx, dy)) % 360


def split_multilinestring(gdf: gpd.GeoDataFrame, azimuth_threshold: float = 60) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Splits complex road shapes into simpler bits if they change direction too much.

    Args:
        gdf (GeoDataFrame): Table of road data with shapes.
        azimuth_threshold (float): Max angle difference before splitting (default: 60 degrees).

    Returns:
        Tuple: Two tables - one with updated roads, one with just the split bits.
    """
    new_rows = []
    split_rows = []
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        if isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
            current_segment = [lines[0]]
            for i in range(1, len(lines)):
                prev_azimuth = calculate_azimuth(lines[i-1])
                curr_azimuth = calculate_azimuth(lines[i])
                azimuth_diff = min((curr_azimuth - prev_azimuth) % 360, (prev_azimuth - curr_azimuth) % 360)
                if azimuth_diff > azimuth_threshold:
                    logger.debug(f"Split MultiLineString for OBJECTID_1: {row.get('OBJECTID_1', 'unknown')}, azimuth_diff = {azimuth_diff:.2f}")
                    new_row = row.copy()
                    new_row['geometry'] = MultiLineString(current_segment) if len(current_segment) > 1 else current_segment[0]
                    new_rows.append(new_row)
                    split_rows.append(new_row.copy())
                    current_segment = [lines[i]]
                else:
                    current_segment.append(lines[i])
            new_row = row.copy()
            new_row['geometry'] = MultiLineString(current_segment) if len(current_segment) > 1 else current_segment[0]
            new_rows.append(new_row)
            split_rows.append(new_row.copy())
        else:
            new_rows.append(row.copy())
    return (gpd.GeoDataFrame(new_rows, crs=gdf.crs).reset_index(drop=True),
            gpd.GeoDataFrame(split_rows, crs=gdf.crs).reset_index(drop=True) if split_rows else gpd.GeoDataFrame())


def fuse_road_data(
        gdf_main: gpd.GeoDataFrame,
        gdf_supp: gpd.GeoDataFrame,
        output_dir: str,
        main_road_num_col: str,
        supp_road_num_col: str,
        main_road_name_cols: List[str],
        supp_road_name_col: str,
        verification_id_col: str = "OBJECTID",
        log_level_console: str = "INFO",
        log_level_file: str = "INFO",
        buffer_distance: float = 15,
        azimuth_threshold: float = 60,
        supp_columns_to_transfer: Optional[List[str]] = None,
        crs: str = "epsg:3111",
        output_format: Optional[str] = None,
        output_name: str = "gdf_matched"
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Merges extra road details into the main road data using map matching, splitting tricky shapes, and checking for issues.

    Args:
        gdf_main (GeoDataFrame): The main table of road info.
        gdf_supp (GeoDataFrame): The extra table of road info to add in.
        output_dir (str): Where to save the output files.
        main_road_num_col (str): Column with road numbers in the main table.
        supp_road_num_col (str): Column with road numbers in the extra table.
        main_road_name_cols (list): Two columns with road names in the main table [primary, declared].
        supp_road_name_col (str): Column with road names in the extra table.
        verification_id_col (str): Column to spot real records (default: 'OBJECTID').
        log_level_console (str): How much detail to show on screen (default: 'INFO').
        log_level_file (str): How much detail to save in the log file (default: 'INFO').
        buffer_distance (float): How close roads need to be to match up (default: 15 metres).
        azimuth_threshold (float): Max angle change before splitting road shapes (default: 60 degrees).
        supp_columns_to_transfer (list, optional): Which columns to copy from the extra table (default: all except road number and shape).
        crs (str): Map projection code (default: 'epsg:3111').
        output_format (str, optional): Save as 'Shapefile' or 'GeoJSON' (default: None, no saving).
        output_name (str): Name for the main output file (default: 'gdf_matched').

    Returns:
        Tuple: Six tables - matched roads, matched extra roads, unmatched extra roads, problem records, split main roads, split extra roads.
    """
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level=log_level_console, colorize=True)
    logger.add(sink=os.path.join(output_dir, f"{output_name}.log"), level=log_level_file,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", colorize=False)

    # Log where we’re starting
    gdf_main = gdf_main.copy()
    gdf_supp = gdf_supp.copy()
    logger.info(f"gdf_main records before splitting MultiLineStrings: {len(gdf_main)}")
    logger.info(f"gdf_supp records before splitting MultiLineStrings: {len(gdf_supp)}")
    logger.debug(f"gdf_main CRS: {gdf_main.crs}")
    logger.debug(f"gdf_supp CRS: {gdf_supp.crs}")

    # Make sure map projections match
    if gdf_main.crs != gdf_supp.crs:
        gdf_supp = gdf_supp.to_crs(gdf_main.crs)
        logger.info(f"Converted gdf_supp CRS to match gdf_main: {gdf_main.crs}")

    # Clear out old columns we’ll redo
    columns_to_check = ['Travel_Dir', 'Part_Rd_Name', 'Part_DRd_Name', 'Travel_Dir_supp', 'Part_Rd_Name_supp']
    base_cols = ['Travel_Dir', 'Part_Rd_Name', 'Part_DRd_Name']
    for df, df_name in [(gdf_main, 'gdf_main'), (gdf_supp, 'gdf_supp')]:
        suffixed_cols = [col for col in df.columns if any(col.startswith(base + '_') for base in base_cols)]
        cols_to_drop = [col for col in columns_to_check + suffixed_cols if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Removed existing columns from {df_name} before preprocessing: {cols_to_drop}")

    # Grab the columns we’ll copy from the extra table later
    base_supp_columns = [col for col in gdf_supp.columns if col not in (supp_road_num_col, 'geometry')]
    logger.info(f"base_supp_columns: {base_supp_columns}")

    # Shorten road names
    gdf_main['Part_Rd_Name'] = gdf_main[main_road_name_cols[0]].apply(shorten_road_name)
    gdf_main['Part_DRd_Name'] = gdf_main[main_road_name_cols[1]].apply(shorten_road_name)
    gdf_supp['Part_Rd_Name'] = gdf_supp[supp_road_name_col].apply(shorten_road_name)
    logger.info("Added Part Road Name for gdf_main and gdf_supp")

    # Make road number columns consistent
    gdf_main = gdf_main.rename(columns={main_road_num_col: 'road_num'})
    gdf_supp = gdf_supp.rename(columns={supp_road_num_col: 'road_num'})
    gdf_main['road_num'] = gdf_main['road_num'].astype(str)
    gdf_supp['road_num'] = gdf_supp['road_num'].astype(str)

    # Break up complex road shapes
    gdf_main, gdf_main_split_rows = split_multilinestring(gdf_main, azimuth_threshold)
    gdf_supp, gdf_supp_split_rows = split_multilinestring(gdf_supp, azimuth_threshold)
    logger.info(f"gdf_main records after splitting MultiLineStrings: {len(gdf_main)}")
    logger.info(f"gdf_supp records after splitting MultiLineStrings: {len(gdf_supp)}")

    # Split and keep only simple lines
    gdf_main = gdf_main.explode(index_parts=False).reset_index(drop=True)
    gdf_supp = gdf_supp.explode(index_parts=False).reset_index(drop=True)
    gdf_main = gdf_main[gdf_main['geometry'].geom_type == 'LineString']
    gdf_supp = gdf_supp[gdf_supp['geometry'].geom_type == 'LineString']
    logger.info(f"gdf_main records after explode/filter LineStrings: {len(gdf_main)}")
    logger.info(f"gdf_supp records after explode/filter LineStrings: {len(gdf_supp)}")

    # Add travel directions after splitting
    gdf_main['Travel_Dir'] = gdf_main['geometry'].apply(calc_travel_dir)
    gdf_supp['Travel_Dir'] = gdf_supp['geometry'].apply(calc_travel_dir)
    logger.info("Added Travel_Dir for gdf_main and gdf_supp after splitting")

    # Tidy up the split rows
    for df, name in [(gdf_main_split_rows, 'gdf_main_split_rows'), (gdf_supp_split_rows, 'gdf_supp_split_rows')]:
        if not df.empty:
            df = df.explode(index_parts=False).reset_index(drop=True)
            df = df[df['geometry'].geom_type == 'LineString']
            logger.info(f"{name} records after explode/filter LineStrings: {len(df)}")
        globals()[name] = df

    # Add angle and length measurements
    for df in [gdf_main, gdf_supp]:
        df['azimuth'] = df['geometry'].apply(calculate_azimuth)
        df['length'] = df['geometry'].length

    # Set up for map matching
    gdf_main_buffered = gdf_main.copy()
    gdf_main_buffered['main_index'] = gdf_main_buffered.index
    gdf_main_buffered['geometry'] = gdf_main_buffered['geometry'].buffer(buffer_distance)
    gdf_supp['supp_index'] = gdf_supp.index
    logger.debug(f"Assigned 'supp_index' as gdf_supp.index for matching traceability")

    # Match roads on the map
    matched = gpd.sjoin(gdf_supp, gdf_main_buffered, how='inner', predicate='intersects', lsuffix='_supp', rsuffix='_main')
    logger.info(f"Total spatial matches: {len(matched)}")

    # Filter the matches
    matched = matched[matched['road_num__supp'] == matched['road_num__main']].copy()
    matched['intersection_length'] = matched.apply(
        lambda row: gdf_supp.loc[row['supp_index'], 'geometry'].intersection(
            gdf_main.loc[row['main_index'], 'geometry'].buffer(buffer_distance)
        ).length, axis=1
    )
    matched['length_supp'] = matched['geometry'].length
    matched['azimuth_diff'] = matched.apply(
        lambda row: min((row['azimuth__supp'] - row['azimuth__main']) % 360, (row['azimuth__main'] - row['azimuth__supp']) % 360),
        axis=1
    )
    matched = matched[(matched['azimuth_diff'] < 45) | (matched['azimuth_diff'] > 315)].copy()
    logger.debug(f"Matches after road_num and azimuth filter: {len(matched)}")

    # Look for roads that might go both ways
    matched['is_bidirectional'] = matched.apply(
        lambda row: abs(row['azimuth__supp'] - row['azimuth__main'] - 180) < 45, axis=1
    )
    if matched['is_bidirectional'].sum() > 0:
        logger.warning(f"{matched['is_bidirectional'].sum()} potential bidirectional links detected!")

    # Decide which extra columns to copy
    if supp_columns_to_transfer is None:
        supp_columns_to_transfer = base_supp_columns
        if 'supp_index' not in supp_columns_to_transfer:
            supp_columns_to_transfer.append('supp_index')
        logger.info(f"supp_columns_to_transfer set to all gdf_supp columns: {supp_columns_to_transfer}")
    elif 'supp_index' not in supp_columns_to_transfer:
        supp_columns_to_transfer = supp_columns_to_transfer + ['supp_index']
        logger.warning(f"'supp_index' added to supp_columns_to_transfer: {supp_columns_to_transfer}")

    # Stick the extra info onto the main roads
    assigned_main_indices = set()
    matched_gdfs = []
    for main_idx, main_row in gdf_main.iterrows():
        agg_data = main_row.to_dict()
        agg_data.update({
            'geometry': main_row['geometry'],
            'supp_segment_azimuths': [],
            'supp_segment_lengths': [],
            'intersection_length': None,
            'Travel_Dir_supp': None,
            'Part_Rd_Name_supp': None
        })
        for col in gdf_supp.columns:
            if col not in agg_data and col != 'geometry':
                agg_data[col] = None

        matched_subset = matched[matched['main_index'] == main_idx].copy()
        if not matched_subset.empty and main_idx not in assigned_main_indices:
            matched_subset = matched_subset.sort_values('intersection_length', ascending=False)
            best_match = matched_subset.iloc[0]
            assigned_main_indices.add(main_idx)

            for col in supp_columns_to_transfer:
                supp_col = f"{col}__supp"
                if supp_col in matched_subset.columns:
                    agg_data[col] = best_match[supp_col]
                elif col in matched_subset.columns:
                    agg_data[col] = best_match[col]
                else:
                    supp_idx = best_match['supp_index']
                    if supp_idx in gdf_supp.index:
                        agg_data[col] = gdf_supp.loc[supp_idx, col]

            supp_idx = best_match['supp_index']
            if supp_idx in gdf_supp.index:
                travel_dir_supp = gdf_supp.loc[supp_idx, 'Travel_Dir']
                part_rd_name_supp = gdf_supp.loc[supp_idx, 'Part_Rd_Name']
                agg_data['Travel_Dir_supp'] = travel_dir_supp
                agg_data['Part_Rd_Name_supp'] = part_rd_name_supp
                logger.debug(
                    f"main_idx {main_idx}, road_num {main_row['road_num']}: "
                    f"Assigned supp_index {supp_idx}, Travel_Dir_supp={travel_dir_supp}, Part_Rd_Name_supp={part_rd_name_supp}"
                )

            agg_data['supp_segment_azimuths'] = matched_subset['azimuth__supp'].tolist()
            agg_data['supp_segment_lengths'] = matched_subset['length_supp'].tolist()
            agg_data['intersection_length'] = best_match['intersection_length']

        matched_gdfs.append(pd.DataFrame([agg_data]))

    # Put all the matched results together
    gdf_matched = gpd.GeoDataFrame(
        pd.concat([df for df in matched_gdfs if not df.empty and not df.isna().all().all()], ignore_index=True),
        geometry='geometry', crs=crs
    )
    logger.info(f"Non-null {verification_id_col} records in gdf_matched: {gdf_matched[verification_id_col].notnull().sum()}")

    # Check for stuff-ups
    gdf_matched['main_index'] = gdf_matched.index
    duplicates = gdf_matched[gdf_matched.duplicated(subset=['main_index'], keep=False)].copy()
    if not duplicates.empty:
        duplicates['Warning_Type'] = 'Duplicate'
        logger.warning(f"Found {len(duplicates)} duplicate main_index entries in gdf_matched")

    matched_rows = gdf_matched[gdf_matched[verification_id_col].notnull()].copy()
    required_cols = ['Travel_Dir', 'Travel_Dir_supp', 'Part_Rd_Name', 'Part_Rd_Name_supp', 'Part_DRd_Name']
    for col in required_cols:
        if col not in matched_rows.columns:
            matched_rows[col] = None

    non_matching = matched_rows[
        ((matched_rows['Travel_Dir'] != matched_rows['Travel_Dir_supp']) &
         (matched_rows['Travel_Dir'].notna() & matched_rows['Travel_Dir_supp'].notna())) |
        (matched_rows['Travel_Dir'].isna() & matched_rows['Travel_Dir_supp'].notna()) |
        (matched_rows['Travel_Dir'].notna() & matched_rows['Travel_Dir_supp'].isna()) |
        (((matched_rows['Part_Rd_Name'] != matched_rows['Part_Rd_Name_supp']) &
          (matched_rows['Part_DRd_Name'] != matched_rows['Part_Rd_Name_supp'])) &
         (matched_rows['Part_Rd_Name'].notna() & matched_rows['Part_Rd_Name_supp'].notna() &
          matched_rows['Part_DRd_Name'].notna()))
        ].copy()

    if not non_matching.empty:
        non_matching['Warning_Type'] = non_matching.apply(
            lambda row: 'Travel Direction Mismatch' if (
                    (row['Travel_Dir'] != row['Travel_Dir_supp'] and
                     pd.notna(row['Travel_Dir']) and pd.notna(row['Travel_Dir_supp'])) or
                    (pd.isna(row['Travel_Dir']) and pd.notna(row['Travel_Dir_supp'])) or
                    (pd.notna(row['Travel_Dir']) and pd.isna(row['Travel_Dir_supp']))
            ) else 'Road Name Mismatch', axis=1
        )
        logger.warning(f"Found {len(non_matching)} records with mismatched Travel_Dir or Part_Rd_Name/Part_DRd_Name")
    else:
        logger.info("No mismatches found in gdf_matched for Travel_Dir or road names")

    # Combine any issues into one table
    gdf_issues = gpd.GeoDataFrame(
        pd.concat([duplicates, non_matching], ignore_index=True),
        geometry='geometry', crs=crs
    )
    if gdf_issues.empty:
        logger.info("No issues (duplicates or mismatches) found in gdf_matched")

    # Tidy up the issues table columns
    if not gdf_issues.empty:
        issue_cols_to_end = ['Part_Rd_Name', 'Part_DRd_Name', 'Part_Rd_Name_supp', 'Travel_Dir', 'Travel_Dir_supp', 'Warning_Type', 'geometry']
        issue_other_cols = [col for col in gdf_issues.columns if col not in issue_cols_to_end]
        gdf_issues = gdf_issues[issue_other_cols + issue_cols_to_end]
        logger.debug(f"Reordered gdf_issues columns: {gdf_issues.columns.tolist()}")

    # Split the extra roads into matched and unmatched
    matched_supp_indices = matched['supp_index'].unique()
    gdf_supp_matched = gdf_supp[gdf_supp.index.isin(matched_supp_indices)].reset_index(drop=True)
    gdf_supp_unmatched = gdf_supp[~gdf_supp.index.isin(matched_supp_indices)].reset_index(drop=True)
    logger.info(f"Matched supp records: {len(gdf_supp_matched)}")
    logger.info(f"Unmatched supp records: {len(gdf_supp_unmatched)}")

    # Clean up the matched table
    columns_to_drop = [col for col in ['azimuth', 'length', 'supp_segment_azimuths', 'supp_segment_lengths', 'intersection_length', 'supp_index', 'main_index'] if col in gdf_matched.columns]
    gdf_matched = gdf_matched.drop(columns=columns_to_drop)
    gdf_matched = gdf_matched.rename(columns={'road_num': main_road_num_col})

    # Sort the matched table columns
    matched_cols_to_end = ['Part_Rd_Name', 'Part_DRd_Name', 'Part_Rd_Name_supp', 'Travel_Dir', 'Travel_Dir_supp', 'geometry']
    matched_other_cols = [col for col in gdf_matched.columns if col not in matched_cols_to_end]
    gdf_matched = gdf_matched[matched_other_cols + matched_cols_to_end]
    logger.debug(f"Reordered gdf_matched columns: {gdf_matched.columns.tolist()}")

    # Save the results if asked
    if output_format in ["Shapefile", "GeoJSON"]:
        os.makedirs(output_dir, exist_ok=True)
        ext = "shp" if output_format == "Shapefile" else "geojson"
        driver = "ESRI Shapefile" if output_format == "Shapefile" else "GeoJSON"
        for df, name in [
            (gdf_matched, output_name),
            (gdf_main_split_rows, "gdf_main_split_rows"),
            (gdf_supp_matched, "gdf_supp_matched"),
            (gdf_supp_unmatched, "gdf_supp_unmatched"),
            (gdf_issues, "gdf_issues"),
            (gdf_supp_split_rows, "gdf_supp_split_rows")
        ]:
            if not df.empty:
                df.to_file(os.path.join(output_dir, f"{name}.{ext}"), driver=driver)
                logger.info(f"Saved {name} to {output_dir}/{name}.{ext}")

    return gdf_matched, gdf_supp_matched, gdf_supp_unmatched, gdf_issues, gdf_main_split_rows, gdf_supp_split_rows


def merge_road_climate(gdf_road, gdf_clim, target_col=None):
    """
    Spatially merge road data with climate data, aggregating values and handling missing data via nearest neighbor.

    Parameters:
    gdf_road (GeoDataFrame): Road dataset.
    gdf_clim (GeoDataFrame): Climate dataset.
    target_col (str, optional): Target column name in output.

    Returns:
    GeoDataFrame: Updated road dataset with added climate column.
    """
    numeric_cols = [col for col in gdf_clim.columns if col != 'geometry' and pd.api.types.is_numeric_dtype(gdf_clim[col])]
    if not numeric_cols:
        raise ValueError(f"No numeric columns found in climate data: {gdf_clim.columns}")
    source_col = numeric_cols[0]
    target_col = target_col or source_col
    print(f"Mapping '{source_col}' to '{target_col}'")

    gdf_joined = gpd.sjoin(gdf_road, gdf_clim, how='left', predicate='intersects')
    all_indices = gdf_road.index.unique()
    gdf_aggregated = gdf_joined.groupby(gdf_joined.index).agg({source_col: 'mean'}).reindex(all_indices)
    gdf_result = gdf_road.copy()
    gdf_result[target_col] = gdf_aggregated[source_col]

    if gdf_result[target_col].isna().any():
        missing_mask = gdf_result[target_col].isna()
        missing_roads = gdf_result[missing_mask].copy()
        if len(missing_roads) > 0:
            road_centroids = missing_roads.geometry.centroid
            nearest_clim = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=road_centroids, crs=gdf_road.crs), gdf_clim, how='left', distance_col='distance')
            gdf_result.loc[missing_mask, target_col] = nearest_clim[source_col].values

    print(f"Number of non-null records for '{target_col}': {gdf_result[target_col].notna().sum()}")
    return gdf_result
