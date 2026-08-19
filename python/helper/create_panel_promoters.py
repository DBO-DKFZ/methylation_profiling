"""Create panel_promoters.csv by intersecting V1 (hg19) and V2 (hg38) promoter panels
on genome_coordinates via the genome_position_cpg_mapping.csv bridge file."""

import pandas as pd

v1 = pd.read_csv("data/curated_cpgs/Panel_promoters_V1_hg19_b04.csv", index_col=0)
v2 = pd.read_csv("data/curated_cpgs/Panel_promoters_V2_hg38_A1.csv", index_col=0)
mapping = pd.read_csv("data/genome_position_cpg_mapping.csv")

print(f"V1 rows: {len(v1)}, V2 rows: {len(v2)}, mapping rows: {len(mapping)}")

# One V1 probe can map to several V2 probes, listed comma-separated in the bridge file.
mapping["Probe_ID_v2"] = mapping["Probe_ID_v2"].str.split(",")
mapping_exploded = mapping.explode("Probe_ID_v2")
mapping_exploded["Probe_ID_v2"] = mapping_exploded["Probe_ID_v2"].str.strip()

# The two panels key on different probe-ID columns: V1 on cg_id, V2 on IlmnID.
v1_coords = v1.merge(
    mapping_exploded[["Probe_ID", "genome_coordinates"]].drop_duplicates(),
    left_on="cg_id",
    right_on="Probe_ID",
    how="inner",
).drop(columns="Probe_ID")

v2_coords = v2.merge(
    mapping_exploded[["Probe_ID_v2", "genome_coordinates"]].drop_duplicates(),
    left_on="IlmnID",
    right_on="Probe_ID_v2",
    how="inner",
).drop(columns="Probe_ID_v2")

common_coords = set(v1_coords["genome_coordinates"]) & set(v2_coords["genome_coordinates"])
print(f"Overlapping genome_coordinates: {len(common_coords)}")

v1_filtered = v1_coords[v1_coords["genome_coordinates"].isin(common_coords)]
v2_filtered = v2_coords[v2_coords["genome_coordinates"].isin(common_coords)]

# Keep all columns from both panels.
result = v1_filtered.merge(v2_filtered, on="genome_coordinates", suffixes=("_v1", "_v2"))

assert result["genome_coordinates"].notna().all(), "Null genome_coordinates found"
assert result["cg_id_v1"].notna().all(), "Null cg_id found"
assert result["IlmnID"].notna().all(), "Null IlmnID found"
print(f"Output rows: {len(result)}, columns: {list(result.columns)}")

result.to_csv("data/curated_cpgs/panel_promoters.csv", index=False)
print("Written to data/curated_cpgs/panel_promoters.csv")
