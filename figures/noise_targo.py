import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the data for each noise scenario, excluding "Noise 0.00"
data = {
    "Occlusion Level": ["[0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", 
                        "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)"] * 2,
    "Successful Rate": [
        # TARGO-Net (Noise 0.02) - Original data from benchmark_targo_syn.py lines 18-20
        0.8697863682604272, 0.8109756097560976, 0.8372093023255814, 
        0.8064516129032258, 0.8185483870967742, 0.8062563067608476, 
        0.8076923076923077, 0.796576032225579, 0.7977755308392316,
        # TARGO-Net (Noise 0.05) - New data provided by user
        0.8520, 0.7970, 0.8240, 0.7960, 0.8290, 
        0.8110, 0.7950, 0.7940, 0.7930
    ],
    "Model": ["TARGO-Net (Noise 0.02)"]*9 + ["TARGO-Net (Noise 0.05)"]*9
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the plot size
plt.figure(figsize=(12, 8))

# Define the markers, colors, and line styles
markers = {
    "TARGO-Net (Noise 0.02)": "s", 
    "TARGO-Net (Noise 0.05)": "D"
}
palette = {
    "TARGO-Net (Noise 0.02)": "#4865A9",  # Blue
    "TARGO-Net (Noise 0.05)": "#B2B2B2"   # Gray
}
dashes = {
    "TARGO-Net (Noise 0.02)": (2, 2),     # Dashed line
    "TARGO-Net (Noise 0.05)": (1, 1)      # Dotted line
}

# Set font to Times New Roman with a generic serif fallback
plt.rcParams["font.family"] = ["Times New Roman", "serif"]

# Plot using Seaborn
sns.lineplot(data=df, x="Occlusion Level", y="Successful Rate", hue="Model", style="Model", markers=markers, palette=palette, dashes=dashes)

# Customize labels and ticks
plt.xlabel("Occlusion Level", fontsize=38)
plt.ylabel("Successful Rate", fontsize=38)

# Set x-ticks to show every other label
plt.xticks(ticks=range(0, len(df['Occlusion Level'].unique()), 2), 
           labels=df['Occlusion Level'].unique()[::2], fontsize=32)
plt.yticks(fontsize=32)
plt.grid(False)  # Turn off the grid

# Customize legend
plt.legend(fontsize=25, frameon=False)

# Adjust the layout to remove the space above the plot
plt.subplots_adjust(top=0.95)

# Save the figure
plt.savefig('noise_targo.pdf', bbox_inches='tight')

# Show the plot (optional, can be commented out)
# plt.show()

