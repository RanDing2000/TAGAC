import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the seen objects data from occ_level_sr.json
seen_objects_data = {
    "[0,0.1)": 0.915, "[0.1,0.2)": 0.869, "[0.2,0.3)": 0.868,
    "[0.3,0.4)": 0.871, "[0.4,0.5)": 0.834, "[0.5,0.6)": 0.827,
    "[0.6,0.7)": 0.815, "[0.7,0.8)": 0.800, "[0.8,0.9)": 0.789
}

# Define the data for each model scenario
data = {
    "Occlusion Level": ["[0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", 
                        "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)"] * 2,
    "Successful Rate": [
        # TARGO-Net (Unseen objects) - original TARGO-Net data
        0.8697863682604272, 0.8109756097560976, 0.8372093023255814, 
        0.8064516129032258, 0.8185483870967742, 0.8062563067608476, 
        0.8076923076923077, 0.796576032225579, 0.7977755308392316,
        # TARGO-Net (Seen objects) - from occ_level_sr.json
        0.915, 0.869, 0.868, 0.871, 0.834, 0.827, 0.815, 0.800, 0.789
    ],
    "Model": ["TARGO-Net (Unseen objects)"]*9 + ["TARGO-Net (Seen objects)"]*9
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the plot size
plt.figure(figsize=(12, 8))

# Define the markers, colors, and line styles
markers = {
    "TARGO-Net (Unseen objects)": "o",
    "TARGO-Net (Seen objects)": "s"
}
palette = {
    "TARGO-Net (Unseen objects)": "#EF8A43",
    "TARGO-Net (Seen objects)": "#4865A9"
}
dashes = {
    "TARGO-Net (Unseen objects)": "",
    "TARGO-Net (Seen objects)": ""
}

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Plot using Seaborn
sns.lineplot(data=df, x="Occlusion Level", y="Successful Rate", 
            hue="Model", style="Model", 
            markers=markers, palette=palette, dashes=dashes)

plt.xlabel("Occlusion Level", fontsize=38)
plt.ylabel("Successful Rate", fontsize=38)

# Set x-ticks to show every other label
plt.xticks(ticks=range(0, len(df['Occlusion Level'].unique()), 2), 
           labels=df['Occlusion Level'].unique()[::2], fontsize=32)
plt.yticks(fontsize=32)
plt.grid(False)

# Customize legend
plt.legend(fontsize=25, frameon=False)

# Adjust the layout
plt.subplots_adjust(top=0.95)

# Save the figure
plt.savefig('seen_vs_unseen_full.pdf', bbox_inches='tight')

# Show the plot
# plt.show()