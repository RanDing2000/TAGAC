import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Font configuration for display
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # Multiple font fallbacks
plt.rcParams['axes.unicode_minus'] = False

# Data preparation (prices as of March 21, 2024)
try:
    companies = ['Microsoft', 'Amazon', 'Google']
    start_prices = [376.04, 151.54, 138.04]  # Prices on Jan 2, 2024
    current_prices = [428.74, 178.75, 150.67]  # Current prices
    gains = [(current - start) / start * 100 for current, start in zip(current_prices, start_prices)]

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Tech Stocks Performance Comparison 2024', fontsize=16, pad=20)

    # Plot percentage gains
    bars = ax1.bar(companies, gains, color=['#90EE90', '#ADD8E6', '#FFB6C1'])
    ax1.set_title('Percentage Gains')
    ax1.set_ylabel('Gain (%)')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    # Plot price comparison
    x = np.arange(len(companies))
    width = 0.35

    ax2.bar(x - width/2, start_prices, width, label='Initial Price', color='lightgray')
    ax2.bar(x + width/2, current_prices, width, label='Current Price', color='skyblue')
    ax2.set_title('Price Comparison')
    ax2.set_ylabel('Stock Price (USD)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(companies)
    ax2.legend()

    # Add price labels
    for i, price in enumerate(start_prices):
        ax2.text(i - width/2, price, f'${price}', ha='center', va='bottom')
    for i, price in enumerate(current_prices):
        ax2.text(i + width/2, price, f'${price}', ha='center', va='bottom')

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig('tech_stocks_comparison_2024.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

except Exception as e:
    print(f"An error occurred: {str(e)}") 