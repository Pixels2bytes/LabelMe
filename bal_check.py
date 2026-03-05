"""Check if dataset is balanced"""

import pandas as pd
import plotly.express as px
from itertools import combinations

FILE_DIR = "create dataset/dangerous_weapons"
CSV_FILE = f"{FILE_DIR}/image_annotations_this.csv"
OBJECT_PIE_GRAPH_FILE = f"{FILE_DIR}/object_balance_pie_graph.png"
OBJECT_BAR_GRAPH_FILE = f"{FILE_DIR}/object_balance_bar_graph.png"
COMBO_BAR_GRAPH_FILE = f"{FILE_DIR}/image_object_combinations_bar_graph.png"
COMBO_PIE_GRAPH_FILE = f"{FILE_DIR}/image_object_combinations_pie_graph.png"


PIE_CSV = f"{FILE_DIR}/object_count_pie.csv"
BAR_CSV = f"{FILE_DIR}/combination_counts_bar.csv"

df = pd.read_csv(CSV_FILE)

# Object Frequency
label_counts = df["Label"].value_counts().reset_index()
label_counts.columns = ["Label", "Count"]

obj_fig_pie = px.pie(
    label_counts,
    names="Label",
    values="Count",
    title="Object Distribution Across All Images"
)
obj_fig_pie.write_image(OBJECT_PIE_GRAPH_FILE)

obj_fig_bar = px.bar(
    label_counts,
    x="Label",
    y="Count",
    title="Object Distribution Across All Images",
    color="Label",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
obj_fig_bar.update_layout(xaxis_tickangle=-45)
obj_fig_bar.write_image(OBJECT_BAR_GRAPH_FILE)

# Group labels per image
image_labels = df.groupby("Frame")["Label"].apply(lambda x: sorted(set(x))).reset_index()

combination_list = []

for labels in image_labels["Label"]:
    
    # Single class images
    if len(labels) == 1:
        combination_list.append(labels[0])
    
    # Multi-class combinations
    else:
        for r in range(1, len(labels)+1):
            for combo in combinations(labels, r):
                combination_list.append("-".join(combo))

# Count objects per image and their combinations
combination_counts = pd.Series(combination_list).value_counts().reset_index()
combination_counts.columns = ["Combination", "Image_Count"]

# Optional: Remove single-label counts if only want multi-class combos
# combination_counts = combination_counts[combination_counts["Combination"].str.contains("-")]

combo_fig_bar = px.bar(
    combination_counts,
    x="Combination",
    y="Image_Count",
    title="Image-Label Combinations",
    color="Combination",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

combo_fig_bar.update_layout(xaxis_tickangle=-45)
combo_fig_pie = px.pie(
    combination_counts,
    names="Combination",
    values="Image_Count",
    title="Image-Label Combinations"
)
combo_fig_bar.write_image(COMBO_BAR_GRAPH_FILE)
combo_fig_pie.write_image(COMBO_PIE_GRAPH_FILE)

# Save the combination counts and object counts to csv files
combination_counts.to_csv(BAR_CSV, index=False)
label_counts.to_csv(PIE_CSV, index=False)