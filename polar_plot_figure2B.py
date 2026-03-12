# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:29:13 2026

@author: Margaux Thieury
"""

"""
Script for Figure 2B
Polar plots script for the mean presence rate (proportion of minutes per day containing vocalizations) 
for dominant taxa during day and night across the five habitats. 

Generates radar plots from presence_rate_acoustic_data.xlsx file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source file (created by consolidate_data.py)
INPUT_FILE = "presence_rate_acoustic_data.xlsx"

# Month order
MONTH_ORDER = ["March", "May", "July", "October", "December"]
MONTH_SHORT = ["Mar", "May", "Jul", "Oct", "Dec"]

# Color palettes
PALETTE_DAY = {
    "Amphibians (audible)": "#6A706E",
    "Birds (audible)": "#B4B3E3",
    "Insects (audible)": "#361354",
    "Insects (ultrasonic)": "#E54B4B"
}

PALETTE_NIGHT = {
    "Amphibians (audible)": "#6A706E",
    "Birds (audible)": "#B4B3E3",
    "Insects (audible)": "#361354",
    "Bats (ultrasonic)": "#E29578",
    "Insects (ultrasonic)": "#E54B4B"
}

# Radar angles (optimized for 5 months)
ANGLES = [1.0472, 2.0944, 3.14159, 4.71239, 5.75959, 1.0472]

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def load_consolidated_data(filepath):
    """Loads and prepares consolidated data"""
    df_audible = pd.read_excel(filepath, sheet_name='Audible_Aggregated')
    df_ultrason = pd.read_excel(filepath, sheet_name='Ultrasonic_Aggregated')
    
    # Restructure to format expected by radar plot
    data = []
    
    # Audible data
    for _, row in df_audible.iterrows():
        data.append({
            'Location': row['Location'],
            'Month': row['Month'],
            'Period': row['Period'],
            'Source': 'Birds (audible)',
            'Presence_rate': row['Bird_rate']
        })
        data.append({
            'Location': row['Location'],
            'Month': row['Month'],
            'Period': row['Period'],
            'Source': 'Insects (audible)',
            'Presence_rate': row['Insect_rate']
        })
        data.append({
            'Location': row['Location'],
            'Month': row['Month'],
            'Period': row['Period'],
            'Source': 'Amphibians (audible)',
            'Presence_rate': row['Amphibian_rate']
        })
    
    # Ultrasonic data
    for _, row in df_ultrason.iterrows():
        data.append({
            'Location': row['Location'],
            'Month': row['Month'],
            'Period': row['Period'],
            'Source': 'Bats (ultrasonic)',
            'Presence_rate': row['Bat_rate']
        })
        data.append({
            'Location': row['Location'],
            'Month': row['Month'],
            'Period': row['Period'],
            'Source': 'Insects (ultrasonic)',
            'Presence_rate': row['Insect_rate']
        })
    
    return pd.DataFrame(data)


def create_radar_plot(df_all, location, period):
    """Creates a radar plot for a given location and period"""
    df_plot = df_all[(df_all["Location"] == location) & (df_all["Period"] == period)]
    
    # Filter sources (no bats during day)
    sources = df_plot["Source"].unique()
    if period == "day":
        sources = [s for s in sources if s != "Bats (ultrasonic)"]
    
    palette = PALETTE_DAY if period == "day" else PALETTE_NIGHT
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot each source
    for source in sources:
        values = df_plot[df_plot["Source"] == source].set_index("Month").reindex(MONTH_ORDER)["Presence_rate"].fillna(0).values
        values = np.concatenate((values, [values[0]]))  # Close polygon
        
        ax.plot(ANGLES, values, label=source, color=palette.get(source, "grey"), 
                linewidth=1, marker='o', markersize=3)
        ax.fill(ANGLES, values, alpha=0.4, color=palette.get(source, "grey"))
    
    # Visual configuration
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels(MONTH_SHORT, fontsize=32, fontweight='bold')
    ax.tick_params(axis='x', pad=20)
    
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("Loading consolidated data...")
    df_all = load_consolidated_data(INPUT_FILE)
    
    locations = df_all["Location"].unique()
    periods = df_all["Period"].unique()
    
    print(f"Available locations: {', '.join(locations)}")
    print(f"Available periods: {', '.join(periods)}")
    print("\nGenerating radar plots...")
    
    # Generate all plots
    for location in locations:
        for period in periods:
            print(f"  - {location} ({period})")
            fig = create_radar_plot(df_all, location, period)
            
            # Save
            filename = f"radar_{location}_{period}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {filename}")
    
    print("\n✓ All plots have been generated!")


# ============================================================================
# FUNCTION FOR SINGLE PLOT
# ============================================================================

def plot_single_radar(location="Lagoon", period="night"):
    """
    Generates a single radar plot (simplified interface)
    
    Args:
        location: "Wetgrassland", "Savannah", "Forest", "Lake", "Lagoon"
        period: "day" or "night"
    """
    print(f"Generating radar for {location} ({period})...")
    df_all = load_consolidated_data(INPUT_FILE)
    fig = create_radar_plot(df_all, location, period)
    plt.show()


if __name__ == "__main__":
    # Uncomment the desired line:
    
    # Option 1: Generate all plots and save them
    main()
    
    # Option 2: Display a single interactive plot
    # plot_single_radar(location="Lagoon", period="night")