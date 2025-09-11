import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## VGN (Raw data)
# 0.6876907426246185, 0.5711382113821138, 0.6076845298281092, 
# 0.5675403225806451, 0.5574596774193549, 0.5771947527749748, 
# 0.5850202429149798, 0.5770392749244713, 0.5662285136501517,
# 0.7273652085452695, 0.6371951219512195, 0.621840242669363, 
# 0.6058467741935484, 0.5877016129032258, 0.5842583249243188, 
# 0.5718623481781376, 0.5317220543806647, 0.5015166835187057,
# Define the data for each model scenario
data = {
    "Occlusion Level": ["[0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", 
                        "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)"] * 6,
    "Successful Rate": [
        # TARGO-Net (Ours)
        0.8697863682604272, 0.8109756097560976, 0.8372093023255814, 
        0.8064516129032258, 0.8185483870967742, 0.8062563067608476, 
        0.8076923076923077, 0.796576032225579, 0.7977755308392316,
        # GIGA_HR
        0.8128179043743642, 0.7347560975609756, 0.7280080889787665, 
        0.6955645161290323, 0.6945564516129032, 0.656912209889001, 
        0.5910931174089069, 0.6112789526686808, 0.5763397371081901,
        # GIGA
        0.797558494404883, 0.717479674796748, 0.6905965621840243, 
        0.6693548387096774, 0.6834677419354839, 0.649848637739657, 
        0.6265182186234818, 0.6324269889224572, 0.5995955510616785,
        ## VGN
        0.7273652085452695, 0.6371951219512195, 0.621840242669363, 
        0.6058467741935484, 0.5877016129032258, 0.5842583249243188, 
        0.5718623481781376, 0.5317220543806647, 0.5015166835187057,
        # 0.6876907426246185, 0.5711382113821138, 0.6076845298281092, 
        # 0.5675403225806451, 0.5574596774193549, 0.5771947527749748, 
        # 0.5850202429149798, 0.5770392749244713, 0.5662285136501517,
        # VN-EdgeGraspNet
        0.638600, 0.594000, 0.601800, 0.597600, 0.584000, 
        0.572800, 0.507615, 0.445409, 0.387691,
        # EdgeGraspNet
        0.602667, 0.580333, 0.584000, 0.585333, 0.551333, 
        0.531000, 0.433534, 0.376636, 0.272006
    ],
    "Model": ["TARGO-Net (Ours)"]*9 + ["GIGA_HR"]*9 + ["GIGA"]*9 + ["VGN"]*9 + ["VN-EdgeGraspNet"]*9 + ["EdgeGraspNet"]*9
}


# Create a DataFrame
df = pd.DataFrame(data)

# Set the plot size
plt.figure(figsize=(12, 8))

# Define the markers, colors, and line styles
markers = {
    "TARGO-Net (Ours)": "o", 
    "GIGA_HR": "s", 
    "GIGA": "D", 
    "VGN": "^", 
    "VN-EdgeGraspNet": "v", 
    "EdgeGraspNet": "p"
}
palette = {
    "TARGO-Net (Ours)": "#EF8A43", 
    "GIGA_HR": "#4865A9", 
    "GIGA": "#4865A9", 
    "VGN": "#B2B2B2", 
    "VN-EdgeGraspNet": "#74A0A1", 
    "EdgeGraspNet": "#74A0A1"
}
dashes = {
    "TARGO-Net (Ours)": "", 
    "GIGA_HR": "", 
    "GIGA": (2, 2), 
    "VGN": "", 
    "VN-EdgeGraspNet": "", 
    "EdgeGraspNet": (2, 2)
}

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Plot using Seaborn
sns.lineplot(data=df, x="Occlusion Level", y="Successful Rate", hue="Model", style="Model", markers=markers, palette=palette, dashes=dashes)
# plt.title("Occlusion Level vs Successful Rate", fontsize=24)
plt.xlabel("Occlusion Level", fontsize=38)
plt.ylabel("Successful Rate", fontsize=38)
# Set x-ticks to show every other label
plt.xticks(ticks=range(0, len(df['Occlusion Level'].unique()), 2), 
           labels=df['Occlusion Level'].unique()[::2], fontsize=32)
plt.yticks(fontsize=32)
plt.grid(False)  # Turn off the grid

# plt.grid(False)  # Turn off the grid

# Customize legend
plt.legend(fontsize=25, frameon=False)



# Adjust the layout to remove the space above the plot
plt.subplots_adjust(top=0.95)

# Save the figure
plt.savefig('benchmark_targo_syn.pdf', bbox_inches='tight')

# Show the plot (optional, can be commented out)
plt.show()