import pandas as pd
import webbrowser
import logging
import base64
from importlib.resources import files
from pathlib import Path

from plotly import graph_objs as go, subplots
from typing import Literal, Optional
from pydantic import ConfigDict, validate_call
from pydantic.dataclasses import dataclass as pydantic_dataclass
from dataclasses import field
from champpy.core.charging.charging_model import ChargingData
from champpy.utils.time_utils import TypeDays, get_week_index
from champpy.utils.data_utils import get_plot_path

logger = logging.getLogger(__name__)


@pydantic_dataclass
class UserParamsChargingPlotter:
    filename: str = "plots\\charging_plots.html"
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
    load_temp_res: Optional[int] = 1  # Temporal resolution in hours, only relevant for load profile plot
    clustering: Optional[bool] = (
        False  # Option to control whether plots are created for clusters (if defined in the data)
    )


class ChargingPlotter:
    """Class for plotting charging characteristics."""

    def __init__(self, user_params: Optional[UserParamsChargingPlotter] = UserParamsChargingPlotter()):
        # Unpack user parameters
        self._filename = user_params.filename
        self._rgb_color = user_params.rgb_color
        self._font_family = user_params.font_family
        self._show = user_params.show
        self._font_size = user_params.font_size
        self._temp_res = user_params.load_temp_res
        self._save_plot = user_params.save_plot
        self._clustering = user_params.clustering

        # Placeholder for temporary variables
        self._clusters = []
        self._legend_clusters = []
        self._label_positions = []

    def plot_charging_profiles(self, charge_data: ChargingData):
        """
        Generate a combined HTML file with plots from plot_charging_char, plot_load_week.

        Parameters:
        charge_data (ChargingData): Input data for the plots.
        """
        logger.info("Generate plot of charging profiles")

        # Disable individual plot showing
        cache_show = self._show
        self._show = False

        # Generate individual plots
        fig_charging_char = self.plot_charging_char(charge_data)
        fig_load_week = self.plot_load_week(charge_data)

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
            # Load logo from package resources and encode as base64
            logo_path = files("champpy").joinpath("data/ffe_logo.svg")
            logo_svg = logo_path.read_text(encoding="utf-8")
            logo_base64 = base64.b64encode(logo_svg.encode("utf-8")).decode("utf-8")
            logo_data_uri = f"data:image/svg+xml;base64,{logo_base64}"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"<html><head><title>CHAMPPy Charging plots</title>")
                f.write(f"<style>body {{ font-family: {self._font_family}; margin: 0; padding: 20px; }} ")
                f.write(".header { display: flex; align-items: center; gap: 1050px; } ")
                f.write(".logo { height: 50px; width: auto; }</style></head><body>\n")
                f.write('<div class="header"><h1>CHAMPPy charging plots</h1>')
                f.write(f'<img src="{logo_data_uri}" class="logo" alt="FfE Logo"></div>\n')
                f.write("<h2>📊 Charging characteristics</h2>")
                f.write(fig_charging_char.to_html(full_html=False, include_plotlyjs="cdn"))
                f.write("<h2>📈 Total load profile of the fleet over the week</h2>\n")
                f.write(fig_load_week.to_html(full_html=False, include_plotlyjs=False))
                f.write("</body></html>")

        # Restore the original show setting
        self._show = cache_show

        # Open the HTML file in the default web browser
        if self._show:
            webbrowser.open(f"file://{output_file}")

    def plot_charging_char(self, charge_data: ChargingData) -> go.Figure:
        """
        Plot charging characteristics: daily driving consumption, daily charging hours, daily charging energy, daily connected hours.

        Parameters:
        charge_data (ChargingData): Charging data to plot.

        Returns:
        fig: Plotly figure object.
        """
        logger.info("Create plot of charging characteristics")

        # Cluster-Handling analog zu plot_mob_char
        if self._clustering and len(charge_data.vehicles.df.id_cluster.unique()) > 1:
            clusters = charge_data.clusters.df["id_cluster"].tolist()
            labels_clusters = charge_data.clusters.df["label"].tolist()
            n_clusters = len(clusters)
        else:
            clusters = [1]
            labels_clusters = [None]
            n_clusters = 1

        char_df_week_weekend = ChargingCharacteristics(
            charge_data, method="mean", typedays=TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]]), clustering=self._clustering
        ).df
        char_df_week = ChargingCharacteristics(
            charge_data, method="mean", typedays=TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]), clustering=self._clustering
        ).df

        # Append mobility characteristics of week and weekend to one dataframe
        charge_char_df = pd.concat([char_df_week_weekend, char_df_week], ignore_index=True)

        # Extract data for plotting
        typedays = ["Mon-Fri", "Sat-Sun", "Mon-Sun"]
        metrics = [
            "daily_charging_hours",
            "daily_connected_hours",
            "simultaneous_factor",
            "daily_driving_consumption",
            "daily_charging_energy",
            "daily_missing_energy",
        ]
        metrics_labels = [
            "Daily charging hours in h",
            "Daily connected hours h",
            "Simultaneity charging factor",
            "Daily driving consumption in kWh",
            "Daily charging energy in kWh",
            "Daily missing energy in kWh",
        ]
        n_subplots = len(metrics)
        n_cols = 3
        n_rows = (n_subplots - 1) // n_cols + 1

        # Create subplot
        fig = subplots.make_subplots(rows=n_rows, cols=n_cols, horizontal_spacing=0.15, vertical_spacing=0.2)

        # Plot for each cluster
        for idx_cluster, cluster in enumerate(clusters):
            # Filter data for the current cluster and type of days
            cluster_data = (
                charge_char_df[charge_char_df["id_cluster"] == cluster] if self._clustering else charge_char_df
            )
            # select color for the cluster
            cluster_color = f"rgb({self._rgb_color[idx_cluster][0] * 255},{self._rgb_color[idx_cluster][1] * 255},{self._rgb_color[idx_cluster][2] * 255})"

            for idx_metrix, metric in enumerate(metrics):
                values = cluster_data[metric].tolist()
                show_legend = (
                    True if idx_metrix == 0 and n_clusters > 1 else False
                )  # Show legend only for the first metric
                row_index = idx_metrix // n_cols + 1
                col_index = idx_metrix % n_cols + 1
                fig.add_trace(
                    go.Bar(
                        y=typedays,
                        x=values,
                        marker_color=cluster_color,
                        orientation="h",
                        name=labels_clusters[idx_cluster],
                        legendgroup=labels_clusters[idx_cluster],
                        showlegend=show_legend,
                        text=[f"{val:.2f}" for val in values],
                        textposition="auto",
                        insidetextanchor="start",
                        textangle=0,
                        legendrank=n_clusters - idx_cluster,
                    ),
                    row=row_index,
                    col=col_index,
                )

        # Update axes and layout
        for idx_metrix, metric in enumerate(metrics):
            row_index = idx_metrix // n_cols + 1
            col_index = idx_metrix % n_cols + 1
            fig.update_xaxes(title_text=metrics_labels[idx_metrix], row=row_index, col=col_index)

        fig.update_layout(
            showlegend=True,
            barmode="group",
            height=600,
            width=1500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family=self._font_family, size=self._font_size),
            legend=dict(font=dict(size=self._font_size, family=self._font_family)),
            margin=dict(l=10, r=10, t=25, b=10),  # Reduce the top margin
        )

        # Update x-axis and y-axis for all subplots to show zero lines
        for i in range(n_subplots):
            row_index = i // n_cols + 1
            col_index = i % n_cols + 1
            fig.update_xaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                layer="above traces",
                title_font=dict(size=self._font_size),
                tickfont=dict(size=self._font_size),
                row=row_index,
                col=col_index,
            )
            fig.update_yaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                title_text="Type of days",
                layer="above traces",
                title_font=dict(size=self._font_size),
                tickfont=dict(size=self._font_size),
                row=row_index,
                col=col_index,
            )

        # Show the plot
        if self._show:
            fig.show()

        return fig

    # ...existing code...
    def plot_load_week(self, charge_data: ChargingData) -> go.Figure:
        """Plot the charging load profile over the course of an average week based on the charging data.
        Parameters:
        ----------
        charge_data (ChargingData): Charging data to plot.
        temp_res (float): Temporal resolution in hours for the load profile. Default is 1 hour.
        fleet (bool): If True, the load profile is aggregated for the entire fleet.
        clustering (bool): If True, separate load profiles are plotted for each cluster defined in the charging data. Default is False.
        Returns:
        ----------
        fig: Plotly figure object.
        """
        # Unpack charging data
        logger.info("Create plot of load profile over the course of a week")
        charge_df = charge_data.charging_timeseries.df.loc[:, ["id_vehicle", "datetime", "power_charging_kw"]]

        # add week index
        charge_df["week_index"] = get_week_index(charge_df["datetime"], temp_res=self._temp_res)

        if self._clustering and len(charge_data.vehicles.df.id_cluster.unique()) > 1:
            # Merge the cluster labels into the charge_df based on id_vehicle
            charge_df = charge_df.merge(
                charge_data.vehicles.df[["id_vehicle", "id_cluster"]], on="id_vehicle", how="left"
            )
            unique_id_cluster = charge_df["id_cluster"].unique()
            cluster_labels = charge_data.clusters.df["label"].tolist()
        else:
            # If clustering is not enabled, assign all data to a single cluster without label
            charge_df["id_cluster"] = 1
            unique_id_cluster = [1]
            cluster_labels = [None]

        # Aggregate charging power for each timestamp, week index, and cluster
        charge_df = charge_df.groupby(["datetime", "week_index", "id_cluster"])["power_charging_kw"].sum().reset_index()

        # Aggregate load profile by week index and cluster
        load_group = charge_df.groupby(["week_index", "id_cluster"])["power_charging_kw"]
        load_min = load_group.min().reset_index()
        load_max = load_group.max().reset_index()
        load_profile_mean = load_group.mean().reset_index()
        load_profile_90_quantile = load_group.quantile(0.9).reset_index()
        load_profile_10_quantile = load_group.quantile(0.1).reset_index()

        max_week_index = int(charge_df["week_index"].max())
        index_week = list(range(max_week_index + 1))
        n_timesteps_day = int(24 / self._temp_res)
        x_ticks = index_week[::n_timesteps_day]
        label_positions = [tick + n_timesteps_day // 2 for tick in x_ticks]  # Midpoints between ticks
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        max_value = load_max["power_charging_kw"].max()

        # Calculate vertical spacing based on absolute pixel distance
        subplot_height = 170  # Height per subplot in pixels
        absolute_spacing = 130  # Absolute spacing between subplots in pixels
        n_clusters = len(unique_id_cluster)
        total_height = n_clusters * subplot_height + (n_clusters) * absolute_spacing
        vertical_spacing = absolute_spacing / total_height

        # Create one subplot per cluster
        fig = subplots.make_subplots(
            rows=len(unique_id_cluster),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            subplot_titles=[
                "" if cluster_labels[idx] is None else f"{cluster_labels[idx]}" for idx in range(len(unique_id_cluster))
            ],
        )

        fig.update_annotations(font=dict(size=self._font_size + 5, family=self._font_family))

        for idx_cluster, cluster in enumerate(unique_id_cluster):
            row_idx = idx_cluster + 1
            cluster_color = f"rgb({self._rgb_color[cluster][0] * 255},{self._rgb_color[cluster][1] * 255},{self._rgb_color[cluster][2] * 255})"

            def _series(df):
                return (
                    df[df["id_cluster"] == cluster]
                    .set_index("week_index")
                    .reindex(index_week)["power_charging_kw"]
                    .tolist()
                )

            y_min = _series(load_min)
            y_max = _series(load_max)
            y_p10 = _series(load_profile_10_quantile)
            y_p90 = _series(load_profile_90_quantile)
            y_mean = _series(load_profile_mean)

            # 100% band (min-max)
            fig.add_trace(
                go.Scatter(
                    x=index_week + index_week[::-1],
                    y=y_min + y_max[::-1],
                    fill="toself",
                    fillcolor=f"rgba({self._rgb_color[2][0] * 255},{self._rgb_color[2][1] * 255},{self._rgb_color[2][2] * 255},0.3)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="100% of the values",
                    showlegend=(idx_cluster == 0),
                ),
                row=row_idx,
                col=1,
            )

            # 80% band (10-90)
            fig.add_trace(
                go.Scatter(
                    x=index_week + index_week[::-1],
                    y=y_p10 + y_p90[::-1],
                    fill="toself",
                    fillcolor=f"rgba({self._rgb_color[2][0] * 255},{self._rgb_color[2][1] * 255},{self._rgb_color[2][2] * 255},1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="80% of the values",
                    showlegend=(idx_cluster == 0),
                ),
                row=row_idx,
                col=1,
            )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=index_week,
                    y=y_mean,
                    mode="lines",
                    line=dict(
                        color=f"rgb({self._rgb_color[6][0] * 255},{self._rgb_color[6][1] * 255},{self._rgb_color[6][2] * 255})",
                        width=2,
                    ),
                    name="Mean",
                    showlegend=(idx_cluster == 0),
                ),
                row=row_idx,
                col=1,
            )

        fig.update_layout(
            height=total_height,
            width=1300,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family=self._font_family, size=self._font_size),
            legend=dict(font=dict(size=self._font_size, family=self._font_family)),
            margin=dict(l=40, r=20, t=30, b=40),
        )

        fig.update_xaxes(
            visible=True,
            title_text="",
            tickmode="array",
            tickvals=x_ticks,
            ticktext=[""] * len(x_ticks),  # Empty tick labels
            title=dict(
                text="Time of week",
                standoff=35,  # Abstand zwischen Titel und Achse
            ),
            range=[0, max_week_index],  # Set x-axis range from 0 to 168 (7 days * 24 hours)
            showticklabels=False,
            ticks="outside",
            zeroline=False,  # Ensure a line at y=0
            zerolinecolor="black",  # Set the color of the zero line
            zerolinewidth=1,  # Set the width of the zero line
            title_font=dict(size=self._font_size),
            tickfont=dict(size=self._font_size),
            layer="above traces",
        )

        fig.update_yaxes(
            title_text="Charging power in kW",
            showline=True,
            linecolor="black",
            linewidth=1,
            showticklabels=True,
            range=[0, max_value],
            ticks="outside",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            title_font=dict(size=self._font_size),
            tickfont=dict(size=self._font_size),
            layer="above traces",
        )

        # Add weekday labels at midpoints for each cluster subplot
        for idx_cluster in range(len(unique_id_cluster)):
            row_idx = idx_cluster + 1
            for i, label_pos in enumerate(label_positions):
                fig.add_annotation(
                    x=label_pos,
                    y=0,  # Adjust this value to position the labels below the x-axis
                    yshift=-15,
                    text=weekdays[i % 7],
                    showarrow=False,
                    yref="y",
                    font=dict(size=self._font_size, family=self._font_family),
                    row=row_idx,
                    col=1,
                )

        if self._show:
            fig.show()

        return fig


class ChargingCharacteristics:
    """
    Class to calculate charging characteristics from charging data as dataframe.

    Parameters:
    ----------
    mob_data (ChargingData): Charging data instance.
    typedays (TypeDays): Define type of days. Default is weekdays and weekend.
    grouping (str): The output table can be grouped by 'none', 'vehicle', or 'day'. Default is 'none'.
    method (str): Method to determine the characteristics: 'mean', 'max', 'min'. Default is 'mean'.

    Returns:
    pd.DataFrame: Overview table with charging characteristics for the defined type of days.
    1. daily_driving_consumption (float): Daily driving consumption in kWh.
    2. daily_charging_hours (float): Daily charging time in h.
    3. daily_charging_energy (float): Daily charging energy in kWh.
    4. daily_connected_hours (float): Daily connected time in h.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        charge_data: ChargingData,
        typedays: TypeDays = TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]),
        grouping: Literal["none", "vehicle", "day"] = "none",
        method: Literal["mean", "max", "min"] = "mean",
        clustering: Optional[bool] = False,
    ):
        """Initialize the ChargingCharacteristics class and calculate the characteristics based on the charging data."""
        self.df = self._calc_charge_char(charge_data, typedays, grouping, method, clustering)

    @staticmethod
    def _calc_charge_char(
        charge_data: ChargingData,
        typedays: TypeDays,
        grouping: Literal["none", "vehicle", "day"],
        method: Literal["mean", "max", "min"],
        clustering: Optional[bool] = False,
    ) -> pd.DataFrame:
        # Prepare data once
        charge_df = charge_data.charging_timeseries.df.copy()
        charge_df["weekday"] = charge_df["datetime"].dt.dayofweek
        charge_df["index_typeday"] = charge_df["weekday"].apply(typedays.weekday2typeday)
        charge_df["date"] = charge_df["datetime"].dt.normalize()

        if clustering:
            charge_df = charge_df.merge(
                charge_data.vehicles.df[["id_vehicle", "id_cluster"]], on="id_vehicle", how="left"
            )
            unique_id_cluster = sorted(charge_data.clusters.df["id_cluster"].unique())
        else:
            charge_df["id_cluster"] = 1
            unique_id_cluster = [1]

        temp_res = (charge_df.loc[1, "datetime"] - charge_df.loc[0, "datetime"]).total_seconds() / 3600

        # Map method
        method_map = {"mean": pd.Series.mean, "min": pd.Series.min, "max": pd.Series.max}
        pd_method = method_map[method]

        charge_char = []

        # Einmal groupby pro Cluster+Typtag combo statt vielen kleinen
        for id_cluster in unique_id_cluster:
            for index_typeday in typedays.index:
                typeday_label = typedays.names[index_typeday]

                # Filter einmal
                mask = (charge_df["index_typeday"] == index_typeday) & (charge_df["id_cluster"] == id_cluster)
                charge_df_filtered = charge_df[mask]

                if charge_df_filtered.empty:
                    continue

                # Einmal großes groupby statt einzelne Operationen
                group = charge_df_filtered.groupby(["id_vehicle", "date"])
                daily_stats = group.agg(
                    {
                        "energy_consumption_kwh": "sum",
                        "power_charging_kw": ["sum", lambda x: (x > 0).sum()],
                        "connected": "sum",
                        "energy_missing_kwh": "sum",
                    }
                )

                # Flatten column names
                daily_stats.columns = [
                    "energy_consumption",
                    "power_sum",
                    "power_count",
                    "connected_sum",
                    "energy_missing",
                ]

                # Berechne die Metriken
                daily_driving_consumption = daily_stats["energy_consumption"]
                daily_charging_hours = daily_stats["power_count"]
                daily_charging_energy = daily_stats["power_sum"] * temp_res
                daily_connected_hours = daily_stats["connected_sum"] * temp_res
                daily_missing_energy = daily_stats["energy_missing"]

                # Simultaneity factor - vektorisiert
                n_vehicles = charge_df_filtered["id_vehicle"].nunique()
                simultaneous_charging = charge_df_filtered.groupby("datetime")["power_charging_kw"].apply(
                    lambda x: (x > 0).sum()
                )
                simultaneous_factor = simultaneous_charging / n_vehicles if n_vehicles > 0 else 0

                # Apply grouping mode
                if grouping == "none":
                    result = {
                        "daily_driving_consumption": pd_method(daily_driving_consumption),
                        "daily_charging_hours": pd_method(daily_charging_hours),
                        "daily_charging_energy": pd_method(daily_charging_energy),
                        "daily_connected_hours": pd_method(daily_connected_hours),
                        "daily_missing_energy": pd_method(daily_missing_energy),
                        "simultaneous_factor": simultaneous_factor.mean(),
                    }
                elif grouping == "day":
                    result = {
                        "daily_driving_consumption": daily_driving_consumption.tolist(),
                        "daily_charging_hours": daily_charging_hours.tolist(),
                        "daily_charging_energy": daily_charging_energy.tolist(),
                        "daily_connected_hours": daily_connected_hours.tolist(),
                        "daily_missing_energy": daily_missing_energy.tolist(),
                        "simultaneous_factor": simultaneous_factor.tolist(),
                    }
                elif grouping == "vehicle":
                    result = {
                        "daily_driving_consumption": daily_driving_consumption.groupby(level="id_vehicle")
                        .agg(pd_method)
                        .tolist(),
                        "daily_charging_hours": daily_charging_hours.groupby(level="id_vehicle")
                        .agg(pd_method)
                        .tolist(),
                        "daily_charging_energy": daily_charging_energy.groupby(level="id_vehicle")
                        .agg(pd_method)
                        .tolist(),
                        "daily_connected_hours": daily_connected_hours.groupby(level="id_vehicle")
                        .agg(pd_method)
                        .tolist(),
                        "daily_missing_energy": daily_missing_energy.groupby(level="id_vehicle")
                        .agg(pd_method)
                        .tolist(),
                        "simultaneous_factor": [simultaneous_factor.mean()],
                    }

                charge_char.append({"typeday": typeday_label, "id_cluster": id_cluster, **result})

        df_mob_char = pd.DataFrame(charge_char)
        if not clustering:
            df_mob_char.drop(columns=["id_cluster"], inplace=True)
        return df_mob_char
