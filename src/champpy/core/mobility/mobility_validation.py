import base64
import pandas as pd
import numpy as np
import webbrowser
import logging

from pathlib import Path
from importlib.resources import files
from plotly import graph_objs as go, express as px, subplots
from typing import Literal, Optional
from pydantic import ConfigDict, validate_call
from pydantic.dataclasses import dataclass as pydantic_dataclass
from rich.progress import Progress
from dataclasses import field
from champpy.core.mobility.mobility_data import MobData, MobDataExtended
from champpy.utils.time_utils import TypeDays, get_week_index
from champpy.utils.data_utils import get_plot_path

logger = logging.getLogger(__name__)


@pydantic_dataclass
class UserParamsMobPlotter:
    filename: str = "mob_plots.html"
    font_family: str = "Segoe UI"
    save_plot: bool = True  # Option to control whether plots are saved to file
    show: bool = True
    font_size: int = 18
    rgb_color: Optional[list] = field(
        default_factory=lambda: [  # RGB color matrix for plotting clusters
            [0.2078, 0.4235, 0.6471],
            [0.9686, 0.8353, 0.0275],
            [0.5412, 0.7098, 0.8824],
            [0.6706, 0.1490, 0.1490],
            [0.1216, 0.3059, 0.4745],
            [0.9255, 0.5765, 0.0078],
            [0.4784, 0.1098, 0.1098],
        ]
    )
    location_temp_res: Optional[int] = 1  # Temporal resolution in hours, only relevant for location profile plots
    location_order: Optional[list] = field(default_factory=lambda: [])
    clustering: Optional[bool] = False


class MobPlotter:
    """Class for plotting mobility characteristics."""

    def __init__(self, user_params: Optional[UserParamsMobPlotter] = UserParamsMobPlotter()):
        # Define a global RGB color matrix
        self._filename = user_params.filename
        self._rgb_color = user_params.rgb_color
        self._font_family = user_params.font_family
        self._show = user_params.show
        self._font_size = user_params.font_size
        self._temp_res = user_params.location_temp_res
        self._location_order = user_params.location_order
        self._save_plot = user_params.save_plot
        self._clustering = user_params.clustering

        # Placeholder for temporary variables
        self._location_order = []
        self._location_labels = []
        self._clusters = []
        self._legend_clusters = []
        self._label_positions = []

    def plot_mob_data(self, mob_data: MobData | MobDataExtended):
        """
        Generate a combined HTML file with plots from plot_mob_char, plot_hist, and plot_location_profile_week.

        Parameters:
        mob_data (MobData | MobDataExtended): Input data for the plots.
        """
        logger.info("Generate plot of mobility profiles")

        # Parse mob_data to ensure it is MobDataExtended
        mob_data_ext = parse_mob_data(mob_data, splitdays=True, clustering=self._clustering)

        # Disable individual plot showing
        cache_show = self._show
        self._show = False

        # Generate individual plots
        fig_mob_char = self.plot_mob_char(mob_data_ext)
        fig_hist = self.plot_hist(mob_data_ext)
        fig_location_profile = self.plot_location_profile_week(mob_data_ext)

        # Ensure the output_file path is absolute and properly formatted
        # Ensure the output_file has .html extension, replacing any existing extension
        output_path = Path(self._filename)
        if output_path.suffix != ".html" and self._save_plot:
            logger.warning(
                f"Can only export as html. Fileformat of output file '{self._filename}'is changed to .html extension."
            )
        output_path = output_path.with_suffix(".html")

        # Debug output
        logger.debug(f"Output filename: {self._filename}")
        logger.debug(f"Output path (before): {output_path}, is_absolute: {output_path.is_absolute()}")

        # If path is relative, resolve based on configuration
        if not output_path.is_absolute():
            output_path = get_plot_path(output_path)
        logger.debug(f"Output path (after get_plot_path): {output_path}")
        output_file = str(output_path.resolve())
        logger.debug(f"Output file (final): {output_file}")

        # Combine all figures into a single HTML file if save_plot is True
        if self._save_plot:
            # Create directory if it doesn't exist
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            # Load logo from package resources and encode as base64
            logo_path = files("champpy").joinpath("data/ffe_logo.svg")
            logo_svg = logo_path.read_text(encoding="utf-8")
            logo_base64 = base64.b64encode(logo_svg.encode("utf-8")).decode("utf-8")
            logo_data_uri = f"data:image/svg+xml;base64,{logo_base64}"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"<html><head><title>CHAMPPy Mobility plots</title>")
                f.write(f"<style>body {{ font-family: {self._font_family}; margin: 0; padding: 20px; }} ")
                f.write(".header { display: flex; align-items: center; gap: 1050px; } ")
                f.write(".logo { height: 50px; width: auto; }</style></head><body>\n")
                f.write('<div class="header"><h1>CHAMPPy mobility plots</h1>')
                f.write(f'<img src="{logo_data_uri}" class="logo" alt="FfE Logo"></div>\n')
                f.write("<h2>📊 Mobility characteristics</h2>")
                f.write(fig_mob_char.to_html(full_html=False, include_plotlyjs="cdn"))
                f.write("<h2>📈 Histogram of mobility characteristics</h2>\n")
                f.write(fig_hist.to_html(full_html=False, include_plotlyjs=False))
                f.write("<h2>📍 Location profile over the week</h2>\n")
                f.write(fig_location_profile.to_html(full_html=False, include_plotlyjs=False))
                f.write("</body></html>")

        # Restore the original show setting
        self._show = cache_show

        # Open the HTML file in the default web browser
        if self._show:
            webbrowser.open(f"file://{output_file}")

    def plot_mob_char(self, mob_data: MobData | MobDataExtended) -> go.Figure:
        """
        Plot the mobility characteristics: daily kilometrage, daily triptime, and number of trips per day.

        Parameters:
        mob_data (MobData | MobDataExtended): Mobility data to plot.

        Returns:
        fig: Plotly figure object.
        """
        logger.info("Create plot of mobility characteristics")
        # Parse mob_data to ensure it is MobDataExtended
        mob_data_ext = parse_mob_data(mob_data, splitdays=True, clustering=self._clustering)
        self._store_locations_clusters(mob_data_ext)

        # Calculate mobility characteristics for the current cluster
        mob_char_df_week_weekdend = MobilityCharacteristics(
            mob_data_ext,
            method="mean",
            typedays=TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]]),
            clustering=self._clustering,
            calc_share_at_locations=False,
        ).df
        mob_char_df_week = MobilityCharacteristics(
            mob_data_ext,
            method="mean",
            typedays=TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]),
            clustering=self._clustering,
            calc_share_at_locations=False,
        ).df

        # Append mobility characteristics of week and weekend to one dataframe
        mob_char_df = pd.concat([mob_char_df_week_weekdend, mob_char_df_week], ignore_index=True)

        # Define typedays and metrics for plotting
        typedays = ["Mon-Fri", "Sat-Sun", "Mon-Sun"]
        metrics = ["daily_kilometrage", "daily_journey_time", "number_journeys_per_day"]
        name_metric = ["Daily kilometrage in km", "Daily journey time in h", "Number of journeys per day"]

        # Create subplot
        fig = subplots.make_subplots(rows=1, cols=3, horizontal_spacing=0.15)

        # Plot for each cluster
        for idx_cluster, cluster in enumerate(mob_data_ext.clusters):
            # Filter data for the current cluster and type of days
            cluster_data = mob_char_df[mob_char_df["id_cluster"] == cluster] if self._clustering else mob_char_df
            # select color for the cluster
            cluster_color = f"rgb({self._rgb_color[idx_cluster][0] * 255},{self._rgb_color[idx_cluster][1] * 255},{self._rgb_color[idx_cluster][2] * 255})"

            for idx_metrix, metric in enumerate(metrics):
                values = cluster_data[metric].tolist()
                show_legend = (
                    True if idx_metrix == 0 and len(self._clusters) > 1 else False
                )  # Show legend only for the first metric
                fig.add_trace(
                    go.Bar(
                        y=typedays,
                        x=values,
                        marker_color=cluster_color,
                        orientation="h",
                        name=mob_data_ext.labels_clusters[idx_cluster],
                        legendgroup=mob_data_ext.labels_clusters[idx_cluster],
                        showlegend=show_legend,
                        text=[f"{val:.2f}" for val in values],
                        textposition="auto",
                        insidetextanchor="start",
                        textangle=0,
                    ),
                    row=1,
                    col=idx_metrix + 1,
                )

        # Update axes and layout
        for idx_metrix, metric in enumerate(metrics):
            fig.update_xaxes(title_text=metric, row=1, col=idx_metrix + 1)

        fig.update_layout(
            showlegend=True,
            barmode="group",
            height=300,
            width=1500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family=self._font_family, size=self._font_size),
            legend=dict(font=dict(size=self._font_size, family=self._font_family)),
            margin=dict(l=10, r=10, t=25, b=10),  # Reduce the top margin
        )

        # Update x-axis and y-axis for all subplots to show zero lines
        for i in range(1, 4):  # Assuming there are 3 subplots (columns)
            fig.update_xaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                layer="above traces",
                title_text=name_metric[i - 1],
                title_font=dict(size=self._font_size, family=self._font_family),
                tickfont=dict(size=self._font_size, family=self._font_family),
                row=1,
                col=i,
            )
            fig.update_yaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                title_text="Type of days",
                layer="above traces",
                title_font=dict(size=self._font_size, family=self._font_family),
                tickfont=dict(size=self._font_size),
                row=1,
                col=i,
            )

        # Show the plot
        if self._show:
            fig.show()

        return fig

    def plot_hist(self, mob_data: MobData | MobDataExtended) -> go.Figure:
        """
        Plot the histogram of the mobility characteristics from mobility data.

        Parameters:
        mob_data ( MobData | MobDataExtended): Mobility data to plot.

        Returns:
        fig: Plotly figure object.
        """
        logger.info("Create plot of mobility histograms")
        mob_data_ext = parse_mob_data(mob_data, splitdays=False, clustering=self._clustering)
        self._store_locations_clusters(mob_data_ext)

        # Get data per day
        t_mob_char_day = MobilityCharacteristics(
            mob_data_ext,
            typedays=TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]),
            grouping="day",
            clustering=self._clustering,
            calc_share_at_locations=False,
        ).df
        daily_mileage_per_day = t_mob_char_day["daily_kilometrage"]
        triptime_per_day = t_mob_char_day["daily_journey_time"]
        n_trips_per_day = t_mob_char_day["number_journeys_per_day"]

        # Get data per vehicle
        t_mob_char_vehicle = MobilityCharacteristics(
            mob_data_ext,
            typedays=TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]),
            grouping="vehicle",
            clustering=self._clustering,
            calc_share_at_locations=False,
        ).df
        daily_mileage_per_vehicle = t_mob_char_vehicle["daily_kilometrage"]
        triptime_per_vehicle = t_mob_char_vehicle["daily_journey_time"]
        n_trips_per_vehicle = t_mob_char_vehicle["number_journeys_per_day"]

        # Get data per journey
        mask_trips = mob_data_ext.df["location"] == 0
        trips_df = mob_data_ext.df.loc[mask_trips]
        grouped = trips_df.groupby("id_cluster")
        mileage_per_trip = pd.Series([g["distance"].tolist() for _, g in grouped])
        duration_per_trip = pd.Series([g["duration"].tolist() for _, g in grouped])
        speed_per_trip = pd.Series([g["speed"].tolist() for _, g in grouped])

        # Create the plot
        fig = subplots.make_subplots(
            rows=3,
            cols=3,
            horizontal_spacing=0.15,
            vertical_spacing=0.2,
            subplot_titles=(
                "Daily kilometrage per day",
                "Daily journey time per day",
                "Number of journeys per day",
                "Daily kilometrage per vehicle",
                "Daily journey time per vehicle",
                "Number of journeys per vehicle",
                "Kilometrage per journey",
                "Duration per journey",
                "Speed per journey",
            ),
        )

        # Update the formatting of subplot titles
        for i, annotation in enumerate(fig["layout"]["annotations"]):
            annotation["font"] = dict(
                size=self._font_size + 5, family=self._font_family
            )  # Customize font size, color, family, and make bold
            annotation["y"] += 0.02  # Adjust the y-position to move the title higher

        # create histograms per day
        fig = self._plot_sub_hist(
            fig=fig, data=daily_mileage_per_day, row=1, col=1, string_xlabel="Daily kilometrage in km", step=20
        )
        fig = self._plot_sub_hist(fig=fig, data=triptime_per_day, row=1, col=2, string_xlabel="Daily journey time in h")
        fig = self._plot_sub_hist(fig=fig, data=n_trips_per_day, row=1, col=3, string_xlabel="Number of journey")
        fig = self._plot_sub_hist(
            fig=fig, data=daily_mileage_per_vehicle, row=2, col=1, string_xlabel="Daily kilometrage in km", step=10
        )
        fig = self._plot_sub_hist(
            fig=fig, data=triptime_per_vehicle, row=2, col=2, string_xlabel="Daily journey time in h"
        )
        fig = self._plot_sub_hist(fig=fig, data=n_trips_per_vehicle, row=2, col=3, string_xlabel="Number of journey")
        fig = self._plot_sub_hist(
            fig=fig, data=mileage_per_trip, row=3, col=1, string_xlabel="kilometrage per journey in km", step=20
        )
        fig = self._plot_sub_hist(
            fig=fig, data=duration_per_trip, row=3, col=2, string_xlabel="Duration per journey in h"
        )
        fig = self._plot_sub_hist(
            fig=fig, data=speed_per_trip, row=3, col=3, string_xlabel="Speed per journey in km/h", step=10
        )

        # Update layout
        fig.update_layout(
            height=1000,
            width=1500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=self._font_size, family=self._font_family),
            showlegend=True if len(self._clusters) > 1 else False,
            legend=dict(font=dict(size=self._font_size, family=self._font_family)),
        )

        # Show the plot
        if self._show:
            fig.show()

        return fig

    def _plot_sub_hist(
        self,
        fig: go.Figure,
        data: pd.Series | list,
        row: int,
        col: int,
        string_xlabel: Optional[str] = None,
        step: Optional[int] = 1,
    ) -> go.Figure:
        """Internal function to create subplots of the histogram of mobility characteristics.
        Parameters:
        fig (go.Figure): Plotly figure object to add the subplot to.
        data (list of np.ndarray): List containing data arrays for each cluster.
        legends (list of str): List of legend labels for each cluster.
        row (int): Row index for the subplot.
        col (int): Column index for the subplot.
        string_xlabel (str, optional): Label for the x-axis. Defaults to None.
        string_title (str, optional): Title for the subplot. Defaults to None.
        step (int, optional): Step size for histogram bins. Defaults to 1.
        Returns:
        go.Figure: Updated Plotly figure object with the new subplot."""

        legends = self._legend_clusters
        max_y = 0  # Initialize max_y

        # determine min and max for x-axis
        min_x = 0
        max_x = step * np.ceil(max(max(d) for d in data) / step)

        # Histogramme und Treppenlinien für jede Datenreihe hinzufügen
        for i, datagroup in enumerate(data):
            edges = np.arange(min_x, max_x, step)  # Define edges based on step size
            hist, _ = np.histogram(datagroup, bins=edges, density=True)

            # save maximum of hist
            max_y = max(hist) if max_y < max(hist) else max_y
            # Add a zero line at the beginning and end
            extended_x = np.concatenate(([edges[0]], np.repeat(edges, 2)[1:-1], [edges[-1]]))
            extended_y = np.concatenate(([0], np.repeat(hist, 2), [0]))

            # plot stairs
            fig.add_trace(
                go.Scatter(
                    x=extended_x,  # Extended X values with zero at start and end
                    y=extended_y,  # Extended Y values with zero at start and end
                    mode="lines",
                    text=legends[i],  # Add legend text
                    legendgroup=legends[i],
                    textposition="top center",  # Position text above the subplot
                    textfont=dict(size=self._font_size),  # Set font size for the text
                    line=dict(
                        color=f"rgb({self._rgb_color[i][0] * 255},{self._rgb_color[i][1] * 255},{self._rgb_color[i][2] * 255})",
                        width=2,
                    ),
                    showlegend=True if row == 1 and col == 1 else False,  # Show legend only for the first subplot
                    name=legends[i],  # Use legend_groups for the name
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            ticks="outside",
            showline=True,
            linecolor="black",
            linewidth=1,
            title_text=string_xlabel,
            layer="above traces",
            title_font=dict(size=self._font_size),
            tickfont=dict(size=self._font_size),
            range=[min_x - max_x * 0.05, max_x * 1.05],  # Set maximum and mimum with 10% margin
            row=row,
            col=col,
        )
        fig.update_yaxes(
            ticks="outside",
            showline=True,
            linecolor="black",
            linewidth=1,
            layer="above traces",
            title_text="Relative frequency",
            title_font=dict(size=self._font_size),
            tickfont=dict(size=self._font_size),
            range=[0, max_y * 1.1],  # Set minimum to 0, maximum remains dynamic
            row=row,
            col=col,
        )

        return fig

    def plot_location_profile_week(self, mob_data: MobData | MobDataExtended) -> go.Figure:
        """
        Create a plot showing the average presence of a vehicle fleet at different locations.

        Parameters:
        t_location (pd.DataFrame): Table containing location data with columns ['id_vehicle', 'dt_start', 'dt_end', 'location'].
        temp_res (int, optional): Temporal resolution in hours. Default is 1.
        location_order (list, optional): Order of locations in the plot from bottom to top. Default is determined by frequency.
        location_labels (list, optional): Labels for the locations. Default is derived from location_order.

        Returns:
        fig: Matplotlib figure object.
        share_loc: Matrix containing the share at locations for each timestep of the week.
        """
        logger.info("Create plot of location profile over the week")
        # Parse mob_data to ensure it is MobDataExtended
        mob_data_ext = parse_mob_data(mob_data, splitdays=True, clustering=self._clustering)
        mob_data_ext_df = mob_data_ext.df

        # Check clusters and legend of clusters
        self._store_locations_clusters(mob_data_ext)
        n_clusters = len(self._clusters)

        # Determine unique locations and set default location order
        unique_locations = np.array(self._unique_locations)
        if (
            self._location_order is None
            or len(self._location_order) == 0
            or len(self._location_order) != len(unique_locations)
        ):
            self._location_order = [0] + list(unique_locations[(unique_locations != 0) & (unique_locations != 1)]) + [1]

        # Set default labels
        if (
            self._location_labels is None
            or len(self._location_labels) == 0
            or len(self._location_labels) != len(self._location_order)
        ):
            self._location_labels = [f"Location = {str(loc)}" for loc in self._location_order]
            index_home = self._location_order.index(1)
            index_driving = self._location_order.index(0)
            self._location_labels[index_home] = "Home"
            self._location_labels[index_driving] = "Driving"

        # Initialize variables
        temp_res = self._temp_res
        n_timesteps_day = int(24 / temp_res)
        n_timesteps_week = n_timesteps_day * 7
        n_locations = len(self._location_order)
        array_share_week = np.zeros((n_timesteps_week, n_locations))

        # Extend array_share_week to include an additional dimension for clusters
        array_share_week = np.zeros((n_timesteps_week, n_locations, n_clusters))

        mob_data_ext_df["start_index"] = get_week_index(mob_data_ext_df["start_dt"], temp_res)
        mob_data_ext_df["end_index"] = get_week_index(mob_data_ext_df["end_dt"], temp_res)

        # Vektorisierte Berechnung der Aufenthaltsmatrix für alle Cluster und Locations
        for cluster_idx, cluster in enumerate(self._clusters):
            cluster_data = mob_data_ext_df[mob_data_ext_df["id_cluster"] == cluster]
            starts = cluster_data["start_index"].values
            ends = cluster_data["end_index"].values - 1
            ends[ends < 0] = n_timesteps_week - 1
            locs = cluster_data["location"].values
            # Korrigiere End-Indizes, falls nötig (optional, je nach Datenstruktur)
            lengths = ends - starts + 1
            mask = lengths > 0
            if np.any(mask):
                all_indices = np.concatenate([np.arange(s, e + 1) for s, e in zip(starts[mask], ends[mask])])
                all_locs = np.repeat(locs[mask], lengths[mask])
                # Mapping location zu Index in location_order
                loc_indices = np.array([self._location_order.index(loc) for loc in all_locs])
                # Zähle Aufenthalte pro Zeitindex und Location
                np.add.at(array_share_week, (all_indices, loc_indices, cluster_idx), 1)

        # Normalize the share at locations
        self._array_share_week_norm = array_share_week / array_share_week.sum(axis=1, keepdims=True) * 100

        # Define x-ticks and labels for plotting
        self._x_ticks = list(range(0, n_timesteps_week, n_timesteps_day))  # Every 24 hours
        self._label_positions = [tick + n_timesteps_day // 2 for tick in self._x_ticks]  # Midpoints between ticks
        self._weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # Create the plot
        if n_clusters == 1:
            fig = self._plot_location_profile_week_1cluster()
        else:
            fig = self._plot_location_profile_week_ncluster()

        if self._show:
            fig.show()
        return fig

    def _plot_location_profile_week_1cluster(self) -> go.Figure:
        # create dataframe for share at location
        n_timesteps_week = self._array_share_week_norm.shape[0]
        df_share_loc = pd.DataFrame(self._array_share_week_norm[:, :, 0], columns=self._location_labels)

        fig = px.area(
            df_share_loc,
            x=df_share_loc.index,
            y=df_share_loc.columns,
            color_discrete_sequence=[f"rgba({r*255},{g*255},{b*255},1.0)" for r, g, b in self._rgb_color],
        )

        # Set explicit colors with full opacity to each trace
        for idx, (trace, color) in enumerate(zip(fig.data, self._rgb_color)):
            rgba_color = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1.0)"
            trace.fillcolor = rgba_color
            trace.line.color = rgba_color
            trace.opacity = 1.0

        # Formatting the plot
        fig.update_layout(
            xaxis=dict(
                visible=True,
                tickmode="array",
                showgrid=True,
                tickvals=self._x_ticks,  # Ticks remain at 0, 24, 48, ...
                ticktext=[""] * len(self._x_ticks),  # Empty tick labels
                title=dict(
                    text="Time of week",
                    standoff=35,  # Abstand zwischen Titel und Achse
                ),
                range=[0, n_timesteps_week],  # Set x-axis range from 0 to 168 (7 days * 24 hours)
                showticklabels=False,
                ticks="outside",
                zeroline=True,  # Ensure a line at y=0
                zerolinecolor="black",  # Set the color of the zero line
                zerolinewidth=1,  # Set the width of the zero line
                title_font=dict(size=self._font_size),
                tickfont=dict(size=self._font_size),
            ),
            font=dict(
                family=self._font_family,  # Schriftart für den gesamten Plot
                size=self._font_size,  # Standard-Schriftgröße
            ),
            yaxis=dict(
                visible=True,
                showgrid=True,
                gridcolor="white",  # Optional: Set gridline color
                gridwidth=1,  # Optional: Set gridline width
                title="Share in %",
                range=[0, 100],  # Set y-axis range from 0 to 100%
                tickmode="array",
                tickvals=[0, 20, 40, 60, 80, 100],  # Define y-ticks
                ticktext=list(range(0, 101, 20)),  # Define y-tick labels
                showticklabels=True,
                ticks="outside",
                zeroline=True,  # Ensure a line at y=0
                zerolinecolor="black",  # Set the color of the zero line
                zerolinewidth=1,  # Set the width of the zero line
                title_font=dict(size=self._font_size),
                tickfont=dict(size=self._font_size),
            ),
            legend=dict(
                traceorder="reversed",  # reverse order of legend items
                title_text="",
                font=dict(size=self._font_size, family=self._font_family),
            ),
            width=1300,  # width of the plot in Pixel
            height=400,  # high of the plot in Pixel
            plot_bgcolor="white",  # background of the plot area
            paper_bgcolor="white",  # background of the entire figure
        )

        # Add weekday labels at midpoints
        for i, label_pos in enumerate(self._label_positions):
            fig.add_annotation(
                x=label_pos,
                y=0,  # Adjust this value to position the labels below the x-axis
                yshift=-15,
                text=self._weekdays[i % 7],
                showarrow=False,
                yref="y",
                font=dict(size=self._font_size, family=self._font_family),
            )

        return fig

    def _plot_location_profile_week_ncluster(self) -> go.Figure:
        # Create subplots for each cluster
        n_timesteps_week = self._array_share_week_norm.shape[0]
        fig = subplots.make_subplots(
            rows=len(self._location_order),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.22,
            subplot_titles=[self._location_labels[i] for i in range(len(self._location_order))],
        )

        # Update the formatting of subplot titles
        for i, annotation in enumerate(fig["layout"]["annotations"]):
            annotation["font"] = dict(size=self._font_size + 5, color="black", family=self._font_family)
            annotation["y"] += 0.012  # Adjust the y-position to move the title higher

        # Plot each location as a separate row
        for loc_idx, loc in enumerate(self._location_order):
            for cluster_idx, cluster in enumerate(self._clusters):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_timesteps_week)),
                        y=self._array_share_week_norm[:, loc_idx, cluster_idx],
                        mode="lines",
                        line=dict(
                            color=f"rgb({self._rgb_color[cluster_idx][0] * 255},{self._rgb_color[cluster_idx][1] * 255},{self._rgb_color[cluster_idx][2] * 255})",
                            width=2,
                        ),
                        name=(
                            self._legend_clusters[cluster_idx]
                            if self._legend_clusters
                            else f"Cluster {cluster_idx + 1}"
                        ),
                        legendgroup=f"Cluster {cluster_idx + 1}",
                        showlegend=(loc_idx == 0),  # Show legend only for the first row
                    ),
                    row=loc_idx + 1,
                    col=1,
                )

            # get the y-axis range from of share_loc
            y_min = 5 * np.floor(self._array_share_week_norm[:, loc_idx, :].min() / 5)
            y_max = 5 * np.ceil(self._array_share_week_norm[:, loc_idx, :].max() / 5)

            fig.update_xaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                tickvals=self._x_ticks,  # Ticks remain at 0, 24, 48, ...
                ticktext=[""] * len(self._x_ticks),  # Empty tick labels
                title=dict(
                    text="Time of week",
                    standoff=35,  # Abstand zwischen Titel und Achse
                ),
                range=[0, n_timesteps_week],  # Set x-axis range from 0 to 168 (7 days * 24 hours)
                showticklabels=False,
                layer="above traces",
                title_font=dict(size=self._font_size),
                row=loc_idx + 1,
                col=1,
            )
            fig.update_yaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                layer="above traces",
                title_text="Share in %",
                title_font=dict(size=self._font_size),
                tickfont=dict(size=self._font_size),
                range=[y_min, y_max],
                row=loc_idx + 1,
                col=1,
            )

            # Add weekday labels at midpoints for each subplot
            for i, label_pos in enumerate(self._label_positions):
                fig.add_annotation(
                    x=label_pos,
                    y=y_min,  # y_min - (y_max-y_min)*0.01,  # Use the minimum from the y-axis range
                    yshift=-15,  # Adjust this value to position the labels below the x-axis
                    text=self._weekdays[i % 7],
                    showarrow=False,
                    yref=f"y{loc_idx + 1}",  # Reference the y-axis of the current subplot
                    xref="x",
                    font=dict(size=self._font_size, family=self._font_family),
                )

        fig.update_layout(
            width=1300,  # width of the plot in Pixel
            height=250 * len(self._location_order),  # high of the plot in Pixel
            plot_bgcolor="white",  # background of the plot area
            paper_bgcolor="white",  # background of the entire figure
            legend=dict(font=dict(size=self._font_size, family=self._font_family)),
        )

        return fig

    def _store_locations_clusters(self, mob_data_ext):
        """Internal function to store the locations and clusters."""
        self._clusters = mob_data_ext.clusters
        self._legend_clusters = mob_data_ext.labels_clusters
        self._unique_locations = mob_data_ext.locations
        self._legend_locations = mob_data_ext.labels_locations

    def show_rgb_colors(self):
        """Show the RGB colors used in the plots as a bar chart."""
        df = pd.DataFrame(
            {
                "Color": [f"Color {i+1}" for i in range(len(self._rgb_color))],
                "Value": [1] * len(self._rgb_color),
                "RGB": [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in self._rgb_color],
            }
        )
        fig = px.bar(df, x="Color", y="Value", color="Color", color_discrete_sequence=df["RGB"], text="Color")
        fig.update_traces(marker_line_color="black", marker_line_width=1, textposition="outside")
        fig.update_layout(
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            xaxis=dict(showgrid=False, zeroline=False),
            showlegend=False,
            title="RGB Colors",
            width=800,
            height=200,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        if self._show:
            fig.show()


class MobilityCharacteristics:
    """
    Class to calculate mobility characteristics from mobility data as dataframe.

    Parameters:
    ----------
    mob_data (MobData | MobDataExtended): Mobility data instance.
    typedays (TypeDays): Define type of days. Default is weekdays and weekend.
    grouping (str): The output table can be grouped by 'none', 'vehicle', or 'day'. Default is 'none'.
    method (str): Method to determine the characteristics: 'mean', 'max', 'min'. Default is 'mean'.
    clustering (bool): Whether to calculate the characteristics for clusters. Default is False.
    calc_share_at_locations (bool): Whether to calculate the variable share_at_locations. Default is True.

    Returns:
    ----------
    pd.DataFrame: Overview table with mobility characteristics for the defined type of days.
    1. daily_kilometrage (float): Daily kilometrage in km.
    2. daily_journey_time (float): Daily journey time in h.
    3. number_journeys_per_day (int): Number of journeys per day.
    4. share_days_with_journeys (float): Share of days with journeys in %.
    5. locations (list): Locations where the vehicle stayed.
    6. share_of_time_at_locations (list[float]): Share of time spent at the locations in %.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        mob_data: MobData | MobDataExtended,
        typedays: TypeDays = TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]),
        grouping: Literal["none", "vehicle", "day"] = "none",
        method: Literal["mean", "max", "min"] = "mean",
        clustering: Optional[bool] = False,
        calc_share_at_locations: bool = True,
    ):

        # Save time of splitting days if all weekdays are in one group
        if len(typedays.groups) == 1:
            splitdays = False
        else:
            splitdays = True

        mob_data_ext = parse_mob_data(mob_data, splitdays=splitdays)

        if calc_share_at_locations and method != "mean":
            logger.warning('The variable <share_of_time_at_locations> can only be calculated for method = "mean".')
            calc_share_at_locations = False

        # Calculate mobility characteristics
        self.df = self._calc_mob_char(
            mob_data_ext,
            typedays=typedays,
            grouping=grouping,
            method=method,
            clustering=clustering,
            calc_share_at_locations=calc_share_at_locations,
        )

    def _calc_mob_char(
        self,
        mob_data_ext: MobDataExtended,
        typedays: TypeDays,
        grouping: Literal["none", "vehicle", "day"] = "none",
        method: Literal["mean", "max", "min"] = "mean",
        clustering: Optional[bool] = False,
        calc_share_at_locations: bool = True,
    ) -> pd.DataFrame:
        """
        Internal Function to calculate mobility characteristics and save the values in an overview dataframe.
        """
        # Prepare extended mob data dataframe
        mob_data_ext_df = mob_data_ext.df
        mob_data_ext_df["weekday"] = mob_data_ext_df["start_dt"].dt.dayofweek  # Monday=0, Sunday=6
        mob_data_ext_df["index_typeday"] = mob_data_ext_df["weekday"].apply(typedays.weekday2typeday)
        mob_data_ext_df["date"] = mob_data_ext_df["start_dt"].dt.normalize()
        # Add a new column 'duration_driving' where 'duration' is retained if 'location' is 0, otherwise 0
        mob_data_ext_df["duration_driving"] = np.where(mob_data_ext_df["location"] == 0, mob_data_ext_df["duration"], 0)
        if clustering:
            unique_id_cluster = mob_data_ext_df["id_cluster"].unique()
        else:
            unique_id_cluster = [1]

        if method == "mean":
            pd_method = pd.Series.mean
        elif method == "min":
            pd_method = pd.Series.min
        elif method == "max":
            pd_method = pd.Series.max
        else:
            raise ValueError("Method must be one of ['mean', 'min', 'max']")

        mob_char = []

        for id_cluster in unique_id_cluster:
            for index_typeday in typedays.index:
                typeday_label = typedays.names[index_typeday]

                # Filter rows for current type of days
                mask_days = mob_data_ext_df["index_typeday"] == index_typeday
                mask_clusters = mob_data_ext_df["id_cluster"] == id_cluster
                mob_data_filtered = mob_data_ext_df[mask_days & mask_clusters]

                # Group by vehicle and day
                group = mob_data_filtered.groupby(["id_vehicle", "date"])
                group_vehicles = mob_data_filtered.groupby(["id_vehicle"])

                # Vektorisierte Berechnungen
                daily_mileage = group["distance"].sum()
                daily_triptime = group["duration_driving"].sum()
                daily_n_trips = group["location"].apply(lambda x: (x == 0).sum())
                daily_log_trips = group["location"].apply(lambda x: (x == 0).any())

                # grouping
                if grouping == "none":
                    if calc_share_at_locations:
                        share_at_locations, locations = self._calc_share_of_time_at_locations(mob_data_filtered)
                    stat_daily_mileage = pd_method(daily_mileage)
                    stat_daily_triptime = pd_method(daily_triptime)
                    stat_n_trips = pd_method(daily_n_trips)
                    share_days_with_trips = pd_method(daily_log_trips)

                elif grouping == "day":
                    if calc_share_at_locations:
                        share_at_locations, locations = zip(
                            *group.apply(lambda x: self._calc_share_of_time_at_locations(x))
                        )
                    stat_daily_mileage = daily_mileage
                    stat_daily_triptime = daily_triptime
                    stat_n_trips = daily_n_trips
                    share_days_with_trips = daily_log_trips

                elif grouping == "vehicle":
                    if calc_share_at_locations:
                        share_at_locations, locations = zip(
                            *group_vehicles.apply(lambda x: self._calc_share_of_time_at_locations(x))
                        )
                    stat_daily_mileage = daily_mileage.groupby(level="id_vehicle").agg(pd_method)
                    stat_daily_triptime = daily_triptime.groupby(level="id_vehicle").agg(pd_method)
                    stat_n_trips = daily_n_trips.groupby(level="id_vehicle").agg(pd_method)
                    share_days_with_trips = daily_log_trips.groupby(level="id_vehicle").agg(pd_method)

                # save results
                mob_char.append(
                    {
                        "typeday": typeday_label,
                        "id_cluster": id_cluster,
                        "daily_kilometrage": stat_daily_mileage.tolist(),
                        "daily_journey_time": stat_daily_triptime.tolist(),
                        "number_journeys_per_day": stat_n_trips.tolist(),
                        "share_days_with_journeys": share_days_with_trips.tolist(),
                        "locations": locations if calc_share_at_locations else None,
                        "share_of_time_at_locations": share_at_locations if calc_share_at_locations else None,
                    }
                )

        df_mob_char = pd.DataFrame(mob_char)
        if clustering == False:
            df_mob_char.drop(columns=["id_cluster"], inplace=True)
        return df_mob_char

    @staticmethod
    def _calc_share_of_time_at_locations(mob_data_ext_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the share of time spent at each location.

        Parameters:
        location (np.ndarray | pd.Series): Array of location indices.

        Returns:
        tuple: share percentages and corresponding unique locations.
        """

        # Get total hours per vehicle

        total_hours = mob_data_ext_df.duration.sum()

        # group by location and sum duration
        location_duration_df = mob_data_ext_df.groupby(["location"])["duration"].sum().reset_index()

        location_duration_df.sort_values(by="location", inplace=True)

        share_at_locations = location_duration_df["duration"] / total_hours * 100
        locations = location_duration_df["location"]

        return share_at_locations.to_numpy(), locations.to_numpy()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def parse_mob_data(
    mob_data: MobData | MobDataExtended, splitdays: Optional[bool] = False, clustering: Optional[bool] = False
) -> MobDataExtended:
    """
    Utility function to ensure the input mobility data is of type MobDataExtended.

    Parameters:
    mob_data (MobData | MobDataExtended): Input mobility data.
    splitdays (Optional[bool]): Whether to split trips that span multiple days when converting MobData to MobDataExtended.
                        Default is False.
    clustering (Optional[bool]): Whether to include cluster information in the output. Default is False.

    Returns:
    MobDataExtended: Parsed mobility data.
    """
    if isinstance(mob_data, MobDataExtended):
        return mob_data
    elif isinstance(mob_data, MobData):
        return MobDataExtended(mob_data, splitdays=splitdays, clustering=clustering)
    else:
        raise TypeError("mob_data must be an instance of MobData or MobDataExtended.")
