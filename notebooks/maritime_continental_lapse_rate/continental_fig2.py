# %%
# ============================================================================
# FIGURE 2: Continental Sites Analysis (matching Figure 1 layout)
# ============================================================================

# Configuration - easy to modify
selected_continental_sites = ["Lee2", "Lee1", "A10"]  # Sites to include in plots (plotting order)
continental_row_order = [0, 1, 2]  # Can change order: 0=full period, 1=zoom periods, 2=lapse rates

# Plot layout (matching Figure 1)
fig = plt.figure(figsize=(18, 12), dpi=300, layout='constrained')
gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.05, wspace=0, figure=fig)

# Get temperature data for selected continental sites
n_cont_sites = len(selected_continental_sites)
# Use middle 50% of the cmocean matter colormap (25% to 75%)
cont_colors = get_colormap_range(cmocean.cm.matter, n_cont_sites, start_frac=0.25, end_frac=0.75)

continental_sites_data = {}
continental_site_elevations = {}
for i, site in enumerate(selected_continental_sites):
    site_data = ds_hourly.where(ds_hourly.site_id == site, drop=True)
    if len(site_data.sensor_idx) > 0:
        continental_sites_data[site] = site_data.temp_c.mean("sensor_idx")
        continental_site_elevations[site] = float(site_data.elevation.mean().values)

# Row arrangement based on continental_row_order
for row_idx, config_idx in enumerate(continental_row_order):
    config = row_configs[config_idx]

    if config["setup"] == "full_period":
        # Full period timeseries (spans both columns)
        ax = fig.add_subplot(gs[row_idx, :])

        # Add shaded regions for zoom periods
        ax.axvspan(june_end_start, june_end_end, alpha=0.3, color='lightgrey', zorder=2)
        ax.axvspan(july_mid_start, july_mid_end, alpha=0.3, color='lightgrey', zorder=2)

        for i, (site, temp_data) in enumerate(continental_sites_data.items()):
            elev = continental_site_elevations[site]
            temp_data.plot(ax=ax, color=cont_colors[i], label=f"{site} - {elev:.0f}m", zorder=3)

        ax.set_title("Continental Sites Temperature Timeseries - Full Period")
        ax.set_ylabel("Temperature (°C)")
        ax.set_xlabel("")
        ax.legend()
        ax.set_xlim(full_period_start, full_period_end)
        ax.grid(True, alpha=0.3, which='major')
        ax.grid(True, alpha=0.1, which='minor')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set x-axis locators and formatting (like Figure 4)
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_minor_locator(HourLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter('%b %d'))

        # Get all the tick labels and set every other one to empty
        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 == 1:  # Hide every other label (odd indices)
                label.set_visible(False)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    elif config["setup"] == "zoom_periods":
        # Two side-by-side zoom periods
        ax_left = fig.add_subplot(gs[row_idx, 0])
        ax_right = fig.add_subplot(gs[row_idx, 1], sharey=ax_left)

        zoom_axes = [ax_left, ax_right]
        zoom_periods = [(june_end_slice, "Last 5 Days of June"), (july_mid_slice, "July 17-22")]

        for ax, (time_slice, title_suffix) in zip(zoom_axes, zoom_periods):

            for i, (site, temp_data) in enumerate(continental_sites_data.items()):
                elev = continental_site_elevations[site]
                temp_data.sel(datetime=time_slice).plot(
                    ax=ax, color=cont_colors[i], label=f"{site} - {elev:.0f}m", zorder=3
                )

            ax.set_title(f"Continental Sites Temperature - {title_suffix}")
            ax.set_ylabel("Temperature (°C)")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3, which='major')
            ax.grid(True, alpha=0.1, which='minor')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Set axis locators and formatting (like Figure 4)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.set_major_locator(DayLocator())
            ax.xaxis.set_minor_locator(HourLocator(interval=6))
            ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
            # Rotate labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    elif config["setup"] == "lapse_rates":
        # Lapse rate timeseries (spans both columns)
        ax = fig.add_subplot(gs[row_idx, :])

        # Add grey zero line (only for lapse rate plots)
        ax.axhline(y=0, color="grey", linewidth=1.5, zorder=1)

        # Plot overall continental lapse rate
        continental_lapse = ds_hourly.continental_lapse_rate * 1000
        continental_lapse.plot(ax=ax, color="black", label="All continental sites regression",
                          linewidth=2, alpha=0.8, zorder=3)
        ax.fill_between(ds_hourly.datetime, 0, continental_lapse, where=(continental_lapse>0), color='lightgrey', label="Inverted", zorder=0.1)

        ylims = (-20, 20)
        ax.set_ylim(ylims)
        ax.set_ylabel("Lapse Rate (°C/km)")
        ax.set_xlabel("Time")
        ax.legend(loc="upper right")
        ax.set_xlim(full_period_start, full_period_end)
        ax.grid(True, alpha=0.3, which='major')
        ax.grid(True, alpha=0.1, which='minor')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set axis locators and formatting (like Figure 4)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_minor_locator(HourLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
        # Hide every other label for cleaner appearance
        for i, label in enumerate(ax.xaxis.get_majorticklabels()):
            if i % 2 == 1:
                label.set_visible(False)
        # Rotate labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.savefig(output_dir / "Fig2_continental_sites_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("Continental sites analysis completed!")
