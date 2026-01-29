import pandas as pd
import numpy as np
import webbrowser
import os
import logging

from plotly import graph_objs as go, express as px, subplots
from typing import Literal, Optional
from pydantic import ConfigDict, validate_call
from pydantic.dataclasses import dataclass as pydantic_dataclass
from rich.progress import Progress
from dataclasses import field
from champpy.core.mobility_data import MobData, MobDataExtended
from champpy.utils.time_utils import TypeDays, get_week_index

logger = logging.getLogger(__name__)

@pydantic_dataclass
class UserParamsMobPlotter:
    filename: str = "mob_plots.html"
    font_family: str = "Segoe UI"
    show: bool = True
    font_size: int = 18
    rgb_color: Optional[list] = field(default_factory=lambda:[ # RGB color matrix for plotting clusters
            [0.2078, 0.4235, 0.6471],
            [0.9686, 0.8353, 0.0275],
            [0.5412, 0.7098, 0.8824],
            [0.6706, 0.1490, 0.1490],
            [0.1216, 0.3059, 0.4745],
            [0.9255, 0.5765, 0.0078],
            [0.4784, 0.1098, 0.1098]
        ])
    location_temp_res: Optional[int] = 1 # Temporal resolution in hours, only relevant for location profile plots
    location_order: Optional[list] = None # Order of locations for plotting location profiles
    location_labels: Optional[list[str]] = None # Labels for locations for plotting location profiles

class MobPlotter:
    """ Class for plotting mobility characteristics. """
    def __init__(self, user_params: UserParamsMobPlotter):
        # Define a global RGB color matrix
        self._filename = user_params.filename
        self._rgb_color = user_params.rgb_color
        self._font_family = user_params.font_family 
        self._show = user_params.show
        self._font_size = user_params.font_size
        self._temp_res = user_params.location_temp_res
        self._location_order = user_params.location_order
        self._location_labels = user_params.location_labels

        # Placeholder for temporary variables
        self._clusters = []
        self._legend_clusters = []
        self._location_order = []
        self._location_labels = []

    def plot_mob_data(self, mob_data):
        """
        Generate a combined HTML file with plots from plot_mob_char, plot_hist, and plot_location_profile_week.

        Parameters:
        t_location (pd.DataFrame): Input data for the plots.
        output_file (str): Path to the output HTML file. Default is "combined_plots.html".
        font_size (int): Font size for the plots. Default is 18.
        """

        # Parse mob_data to ensure it is MobDataExtended
        mob_data_ext = self._parse_mob_data(mob_data)

        # Disable individual plot showing
        cache_show = self._show
        self._show = False 
        
        # Generate individual plots
        with Progress() as progress:
            task = progress.add_task("Generating plots...", total=3)
            fig_mob_char = self.plot_mob_char(mob_data_ext)
            progress.update(task, advance=1)
            fig_hist = self.plot_hist(mob_data_ext)
            progress.update(task, advance=1)
            fig_location_profile = self.plot_location_profile_week(mob_data_ext)
            progress.update(task, advance=1)

        # Ensure the output_file path is absolute and properly formatted
        output_file = os.path.abspath(self._filename)

        # Combine all figures into a single HTML file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"<html><head><title>Combined Plots</title><style>body {{ font-family: {self._font_family}; }}</style></head><body>\n")
            f.write("<h1>Combined Plots</h1>\n")
            f.write("<h2>📊 Mobility Characteristics</h2>")
            f.write(fig_mob_char.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("<h2>📈 Histogram of Mobility Characteristics</h2>\n")
            f.write(fig_hist.to_html(full_html=False, include_plotlyjs=False))
            f.write("<h2>📍 Location Profile Over the Week</h2>\n")
            f.write(fig_location_profile.to_html(full_html=False, include_plotlyjs=False))
            f.write("</body></html>")

        # Restore the original show setting
        self._show = cache_show

        # Open the HTML file in the default web browser
        if self._show:
            webbrowser.open(f"file://{output_file}")

    def _parse_mob_data(self, mob_data: MobData | MobDataExtended) -> MobDataExtended:
        """
        Ensure the input mobility data is of type MobDataExtended.

        Parameters:
        mob_data (MobData | MobDataExtended): Input mobility data.

        Returns:
        MobDataExtended: Parsed mobility data.
        """
        if isinstance(mob_data, MobDataExtended):
            return mob_data
        elif isinstance(mob_data, MobData):
            return MobDataExtended(mob_data)
        else:
            raise TypeError("mob_data must be an instance of MobData or MobDataExtended.")
    
    @staticmethod
    def calc_share_of_time_at_locations(location: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the share of time spent at each location.

        Parameters:
        location (np.ndarray | pd.Series): Array of location indices.

        Returns:
        tuple: share percentages and corresponding unique locations.
        """
        # Check location array
        if not isinstance(location, np.ndarray) and not isinstance(location, pd.Series):
            raise TypeError("location must be a numpy ndarray or pandas Series.")
        if location.ndim != 1:
            raise ValueError("location array must be one-dimensional.")
        if location.size == 0:
            raise ValueError("location array must not be empty.")

        unique_locations, indices_locations = np.unique(location, return_inverse=True)
        counts = np.bincount(indices_locations)
        total_elements = location.size
        share_at_locations = counts / total_elements * 100
        return share_at_locations, unique_locations
    
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def calc_mob_char(self, 
                mob_data: MobData | MobDataExtended,
                typedays: TypeDays = TypeDays(groups=[[0,1,2,3,4],[5],[6]]),
                grouping: Literal['none', 'vehicle', 'day'] = 'none',
                method: Literal['mean', 'max', 'min'] = 'mean',
                calc_share_at_locations: bool = True) -> pd.DataFrame:
        """
        Calculate mobility characteristics and save the values in an overview dataframe.
        
        Parameters:
        ----------
        mob_data (MobData | MobDataExtended): Mobility data instance.
        typedays (TypeDays): Define type of days. Default is weekdays and weekend.
        grouping (str): The output table can be grouped by 'none', 'vehicle', or 'day'. Default is 'none'.
        method (str): Method to determine the characteristics: 'mean', 'max', 'min'. Default is 'mean'.
        calc_share_at_locations (bool): Whether to calculate the variable share_at_locations. Default is True.
        
        Returns:
        pd.DataFrame: Overview table with mobility characteristics for the defined type of days.
        """

        # Parse mob_data to ensure it is MobDataExtended
        mob_data_ext = self._parse_mob_data(mob_data).df

        if calc_share_at_locations and method != 'mean':
            logger.warning('The variable <share_at_locations> can only be calculated for method = "mean".')
            calc_share_at_locations = False

        # Prepare extended mob data dataframe
        mob_data_ext["weekday"] = mob_data_ext["start_dt"].dt.dayofweek # Monday=0, Sunday=6
        mob_data_ext["index_typeday"] = mob_data_ext["weekday"].apply(typedays.weekday2typeday)
        mob_data_ext['date'] = mob_data_ext['start_dt'].dt.normalize()
        # Add a new column 'duration_driving' where 'duration' is retained if 'location' is 0, otherwise 0
        mob_data_ext["duration_driving"] = np.where(mob_data_ext['location'] == 0, mob_data_ext['duration'], 0)
        unique_id_vehicle = mob_data_ext['id_vehicle'].unique()

        if method == 'mean':
            pd_method = pd.Series.mean
        elif method == 'min':
            pd_method = pd.Series.min
        elif method == 'max':
            pd_method = pd.Series.max
        else:
            raise ValueError("Method must be one of ['mean', 'min', 'max']")

        mob_char = []

        for index_typeday in typedays.index:
            typeday_label = typedays.names[index_typeday]

            # Filter rows for current type of days
            mask_days = mob_data_ext['index_typeday'] == index_typeday
            mob_data_filtered = mob_data_ext[mask_days]

            # Group by vehicle and day
            group = mob_data_filtered.groupby(['id_vehicle', 'date'])
            group_vehicles = mob_data_filtered.groupby(['id_vehicle'])

            # Vektorisierte Berechnungen
            daily_mileage = group['distance'].sum()
            daily_triptime = group['duration_driving'].sum() 
            daily_n_trips = group['location'].apply(lambda x: (x == 0).sum())
            daily_log_trips = group['location'].apply(lambda x: (x == 0).any())

            # grouping
            if grouping == 'none':
                if calc_share_at_locations:
                    share_at_locations, locations = self.calc_share_of_time_at_locations(mob_data_filtered["location"].to_numpy())
                stat_daily_mileage = pd_method(daily_mileage)
                stat_daily_triptime = pd_method(daily_triptime)
                stat_n_trips = pd_method(daily_n_trips)
                share_days_with_trips = pd_method(daily_log_trips)

            elif grouping == 'day':
                if calc_share_at_locations:
                    share_at_locations, locations = zip(*group["location"].apply(lambda x: self.calc_share_of_time_at_locations(x)))
                stat_daily_mileage = daily_mileage
                stat_daily_triptime = daily_triptime
                stat_n_trips = daily_n_trips
                share_days_with_trips = daily_log_trips

            elif grouping == 'vehicle':
                if calc_share_at_locations:
                    share_at_locations, locations = zip(*group_vehicles["location"].apply(lambda x: self.calc_share_of_time_at_locations(x)))
                stat_daily_mileage = daily_mileage.groupby(level='id_vehicle').agg(pd_method)
                stat_daily_triptime = daily_triptime.groupby(level='id_vehicle').agg(pd_method)
                stat_n_trips = daily_n_trips.groupby(level='id_vehicle').agg(pd_method)
                share_days_with_trips = daily_log_trips.groupby(level='id_vehicle').agg(pd_method)

            # save results
            mob_char.append({
                'typeday': typeday_label,
                'daily_kilometrage': stat_daily_mileage.tolist(),
                'daily_journey_time': stat_daily_triptime.tolist(),
                'number_journeys_per_day': stat_n_trips.tolist(),
                'share_days_with_journeys': share_days_with_trips.tolist(),
                'locations': locations if calc_share_at_locations else None,
                'share_of_time_at_locations': share_at_locations if calc_share_at_locations else None
            })

        return pd.DataFrame(mob_char)

    def plot_mob_char(self, mob_data: MobData | MobDataExtended):
        """
        Plot the mobility characteristics: daily kilometrage, daily triptime, and number of trips per day.

        Parameters:
        mob_data (MobData | MobDataExtended): Mobility data to plot.

        Returns:
        fig: Plotly figure object.
        """
        # Parse mob_data to ensure it is MobDataExtended
        mob_data_ext = self._parse_mob_data(mob_data)
        self._get_legend_clusters(mob_data_ext)
        
        # Split t_lb into clusters and apply mob_char for each cluster
        mob_char_df_clusters = []
        number_of_clusters = mob_data_ext.df['cluster'].nunique()
        for cluster in range(1, number_of_clusters+1):
            mob_data_cluster = mob_data_ext.extract_cluster(cluster_id=cluster, copy=False)
            # Call calc_mob_char for the current cluster
            mob_char_df_week_weekdend = self.calc_mob_char(
                            mob_data_cluster,
                            method='mean',
                            typedays=TypeDays(groups=[[0,1,2,3,4],[5,6]]),
                            calc_share_at_locations=False
                            )
            mob_char_df_week = self.calc_mob_char(
                            mob_data_cluster,
                            method='mean',
                            typedays=TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]),
                            calc_share_at_locations=False
                            )
        
            # Add a column to identify the cluster in the result
            mob_char_df_week_weekdend['cluster'] = cluster
            mob_char_df_week['cluster'] = cluster
            # Append the result to the list
            mob_char_df_clusters.append(mob_char_df_week_weekdend)
            mob_char_df_clusters.append(mob_char_df_week)

        # Combine all results into a single DataFrame
        mob_char_df = pd.concat(mob_char_df_clusters, ignore_index=True)

        # Extract data for plotting
        typedays = ["Weekend", "Weekdays", "Total"]
        fig = subplots.make_subplots(rows=1, cols=3, horizontal_spacing=0.15)

        # Plot for each cluster
        for cluster in self._clusters:
            cluster_data = mob_char_df[mob_char_df['cluster'] == cluster] if cluster is not None else mob_char_df
            daily_mileage = cluster_data['daily_kilometrage'].tolist()
            daily_triptime = cluster_data['daily_journey_time'].tolist()
            n_trips_per_day = cluster_data['number_journeys_per_day'].tolist()

            # select color for the cluster
            cluster_color = f"rgb({self._rgb_color[cluster][0] * 255},{self._rgb_color[cluster][1] * 255},{self._rgb_color[cluster][2] * 255})"

            # Add traces for each metric
            fig.add_trace(
                go.Bar(
                    y=typedays,
                    x=daily_mileage,
                    marker_color=cluster_color,
                    orientation='h',
                    name=self._legend_clusters[cluster-1],
                    legendgroup=self._legend_clusters[cluster-1],
                    showlegend=True,
                    text=[f"{val:.2f}" for val in daily_mileage],
                    textposition='auto',
                    insidetextanchor='start',
                    textangle=0
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(
                    y=typedays,
                    x=daily_triptime,
                    marker_color=cluster_color,
                    orientation='h',
                    name=self._legend_clusters[cluster-1],
                    showlegend=False,
                    legendgroup=self._legend_clusters[cluster-1],
                    text=[f"{val:.2f}" for val in daily_triptime],
                    textposition='auto',
                    insidetextanchor='start',
                    textangle=0
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Bar(
                y=typedays,
                x=n_trips_per_day,
                marker_color=cluster_color,
                orientation='h',
                name=self._legend_clusters[cluster-1],
                legendgroup=self._legend_clusters[cluster-1],
                showlegend=False,
                text=[f"{val:.2f}" for val in n_trips_per_day],
                textposition='auto', 
                insidetextanchor='start',
                textangle=0
                ),
                row=1, col=3
            )

        # Update axes and layout
        fig.update_xaxes(title_text="Daily kilometrage in km", row=1, col=1)
        fig.update_xaxes(title_text="Daily journey time in h", row=1, col=2)
        fig.update_xaxes(title_text="Daily number of journeys", row=1, col=3)

        fig.update_layout(
            showlegend=True,
            barmode='group', 
            height=300,
            width=1500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(
                family=self._font_family,
                size=self._font_size
            ),
            legend=dict(
            font=dict(
                size=self._font_size, 
                family=self._font_family
            )),
            margin=dict(
                l=10,
                r=10,
                t=25,  # Reduce the top margin
                b=10
            )
        )

        # Update x-axis and y-axis for all subplots to show zero lines
        for i in range(1, 4):  # Assuming there are 3 subplots (columns)
            fig.update_xaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                layer="above traces",
                title_font=dict(size=self._font_size), 
                tickfont=dict(size=self._font_size),
                row=1, col=i
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
                row=1, col=i
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
        mob_data_ext = self._parse_mob_data(mob_data)

        # legt clusters and legend of clusters
        self._get_legend_clusters(mob_data_ext)
        clusters = self._clusters

        # Initialize data containers
        # Initialize lists with the correct size based on the number of clusters
        daily_mileage_per_day = [None] * len(clusters)
        triptime_per_day = [None] * len(clusters)
        n_trips_per_day = [None] * len(clusters)

        daily_mileage_per_vehicle = [None] * len(clusters)
        triptime_per_vehicle = [None] * len(clusters)
        n_trips_per_vehicle = [None] * len(clusters)

        mileage_per_trip = [None] * len(clusters)
        duration_per_trip = [None] * len(clusters)
        speed_per_trip = [None] * len(clusters)

        # Process data for each cluster
        for idx, cluster in enumerate(clusters):
            # Filter data for the current cluster
            cluster_data = mob_data_ext.extract_cluster(cluster_id=cluster, copy=False)

            # Per day
            t_mob_char_day = self.calc_mob_char(cluster_data, 
                                                typedays=TypeDays(groups=[[0,1,2,3,4,5,6]]),
                                                grouping='day', 
                                                calc_share_at_locations=False)
            daily_mileage_per_day[idx] = t_mob_char_day.at[0,'daily_kilometrage']
            triptime_per_day[idx] = t_mob_char_day.at[0, 'daily_journey_time']
            n_trips_per_day[idx] = t_mob_char_day.at[0, 'number_journeys_per_day']

            # Per vehicle
            t_mob_char_vehicle = self.calc_mob_char(cluster_data, 
                                                typedays=TypeDays(groups=[[0,1,2,3,4,5,6]]),
                                                grouping='vehicle', 
                                                calc_share_at_locations=False)
            daily_mileage_per_vehicle[idx] = t_mob_char_vehicle.at[0,'daily_kilometrage']
            triptime_per_vehicle[idx] = t_mob_char_vehicle.at[0,'daily_journey_time']
            n_trips_per_vehicle[idx] = t_mob_char_vehicle.at[0,'number_journeys_per_day']

            # Per trip
            mask_trips = cluster_data.df['location'] == 0
            mileage_per_trip[idx] = cluster_data.df.loc[mask_trips,'distance'].tolist()
            duration_per_trip[idx] = (cluster_data.df.loc[mask_trips,'duration']).tolist()
            speed_per_trip[idx] = cluster_data.df.loc[mask_trips,'speed'].tolist()
        
        # Create the plot 
        fig = subplots.make_subplots(
            rows=3, cols=3, 
            horizontal_spacing=0.15, 
            vertical_spacing=0.2,
            subplot_titles=(
                'Daily kilometrage per day', 
                'Daily journey time per day', 
                'Number of journeys per day', 
                'Daily kilometrage per vehicle', 
                'Daily journey time per vehicle', 
                'Number of journeys per vehicle', 
                'Kilometrage per journey', 
                'Duration per journey', 
                'Speed per journey'
            )
        )

        # Update the formatting of subplot titles
        for i, annotation in enumerate(fig['layout']['annotations']):
            annotation['font'] = dict(size=self._font_size+5, family=self._font_family)  # Customize font size, color, family, and make bold
            annotation['y'] += 0.02  # Adjust the y-position to move the title higher

        # create histograms per day
        fig = self._plot_sub_hist(fig=fig, data=daily_mileage_per_day,  row=1, col=1, string_xlabel='Daily kilometrage in km', step=20)
        fig = self._plot_sub_hist(fig=fig, data=triptime_per_day, row=1, col=2, string_xlabel='Daily journey time in h')
        fig = self._plot_sub_hist(fig=fig, data=n_trips_per_day, row=1, col=3, string_xlabel='Number of journey')
        fig = self._plot_sub_hist(fig=fig, data=daily_mileage_per_vehicle, row=2, col=1, string_xlabel='Daily kilometrage in km', step=10)
        fig = self._plot_sub_hist(fig=fig, data=triptime_per_vehicle, row=2, col=2, string_xlabel='Daily journey time in h')
        fig = self._plot_sub_hist(fig=fig, data=n_trips_per_vehicle, row=2, col=3, string_xlabel='Number of journey')
        fig = self._plot_sub_hist(fig=fig, data=mileage_per_trip, row=3, col=1, string_xlabel='kilometrage per journey in km', step=20)
        fig = self._plot_sub_hist(fig=fig, data=duration_per_trip, row=3, col=2, string_xlabel='Duration per journey in h')
        fig = self._plot_sub_hist(fig=fig, data=speed_per_trip, row=3, col=3, string_xlabel='Speed per journey in km/h', step=10)
            
        # Update layout
        fig.update_layout(
            height=1000,
            width=1500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=self._font_size, family=self._font_family),
            showlegend=True,
            legend=dict(
                font=dict(
                size=self._font_size, 
                family=self._font_family
            )),
        )

        # Show the plot
        if self._show:
            fig.show()

        return fig


    def _plot_sub_hist(self, 
                       fig: go.Figure, 
                       data: list[np.ndarray], 
                       row: int, 
                       col: int, 
                       string_xlabel: Optional[str]=None, 
                       string_title:Optional[str]=None, 
                       step: Optional[int]=1) ->go.Figure:
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
            fig.add_trace(go.Scatter(
                x=extended_x,  # Extended X values with zero at start and end
                y=extended_y,  # Extended Y values with zero at start and end
                mode='lines',
                text=legends[i],  # Add legend text
                legendgroup=legends[i],
                textposition='top center',  # Position text above the subplot
                textfont=dict(size=self._font_size),  # Set font size for the text
                line=dict(color=f"rgb({self._rgb_color[i][0] * 255},{self._rgb_color[i][1] * 255},{self._rgb_color[i][2] * 255})", width=2), 
                showlegend=True if row == 1 and col == 1 else False,  # Show legend only for the first subplot
                name=legends[i],  # Use legend_groups for the name
                ), row=row, col=col
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
            range=[min_x-max_x*0.05, max_x*1.05], # Set maximum and mimum with 10% margin
            row=row, col=col
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
            range=[0, max_y*1.1],  # Set minimum to 0, maximum remains dynamic
            row=row, col=col
        )

        return fig
    
    def plot_location_profile_week(self,mob_data: MobData | MobDataExtended) -> go.Figure:
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
        # Parse mob_data to ensure it is MobDataExtended
        mob_data_ext = self._parse_mob_data(mob_data)
        mob_data_ext_df = mob_data_ext.df

        # Check clusters and legend of clusters
        self._get_legend_clusters(mob_data_ext)
        n_clusters = len(self._clusters)

        # Determine unique locations and set default location order
        unique_locations = mob_data_ext_df['location'].unique()
        self._unique_locations = unique_locations
        if self._location_order is None or len(self._location_order) == 0 or len(self._location_order) != len(unique_locations):
            self._location_order = [0] + list(unique_locations[(unique_locations != 0) & (unique_locations != 1)]) + [1]

        # Set default labels
        if  self._location_labels is None or len(self._location_labels) == 0 or len(self._location_labels) != len(self._location_order):
            self._location_labels = [f"Location = {str(loc)}" for loc in self._location_order]
            index_home = self._location_order.index(1)
            index_driving = self._location_order.index(0)
            self._location_labels[index_home] = "Home"
            self._location_labels[index_driving] = "Driving"

        # Initialize variables
        temp_res = self._temp_res
        n_timesteps_day = int(24/temp_res)
        n_timesteps_week = n_timesteps_day * 7
        n_locations = len(self._location_order)
        array_share_week = np.zeros((n_timesteps_week, n_locations))

        # Extend array_share_week to include an additional dimension for clusters
        array_share_week = np.zeros((n_timesteps_week, n_locations, n_clusters))

        mob_data_ext_df["start_index"] = get_week_index(mob_data_ext_df["start_dt"], temp_res)
        mob_data_ext_df["end_index"] = get_week_index(mob_data_ext_df["end_dt"], temp_res)

        # Vektorisierte Berechnung der Aufenthaltsmatrix für alle Cluster und Locations
        for cluster_idx, cluster in enumerate(self._clusters):
            cluster_data = mob_data_ext_df[mob_data_ext_df['cluster'] == cluster]
            starts = cluster_data["start_index"].values
            ends = cluster_data["end_index"].values-1
            ends[ends < 0] = n_timesteps_week-1
            locs = cluster_data["location"].values
            # Korrigiere End-Indizes, falls nötig (optional, je nach Datenstruktur)
            lengths = ends - starts + 1
            mask = lengths > 0
            if np.any(mask):
                all_indices = np.concatenate([np.arange(s, e+1) for s, e in zip(starts[mask], ends[mask])])
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
        df_share_loc = pd.DataFrame(self._array_share_week_norm[:,:,0], columns=self._location_labels)

        fig = px.area(
            df_share_loc, 
            x=df_share_loc.index, 
            y=df_share_loc.columns,
            color_discrete_sequence=[f"rgb({r*255},{g*255},{b*255})" for r, g, b in self._rgb_color]
        )

        # Remove lines from the area chart
        fig.update_traces(
            line=dict(width=0),
            opacity=1.0)   
        
                # Formatting the plot
        fig.update_layout(
            xaxis=dict(
                visible=True, 
                tickmode='array',
                showgrid=True, 
                tickvals=self._x_ticks,  # Ticks remain at 0, 24, 48, ...
                ticktext=[""] * len(self._x_ticks),  # Empty tick labels
                title=dict(
                    text = "Time of week",
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
                size=self._font_size  # Standard-Schriftgröße
            ),
            yaxis=dict(
                visible=True, 
                showgrid=True,
                gridcolor='white',  # Optional: Set gridline color
                gridwidth=1,  # Optional: Set gridline width
                title="Share in %",
                range=[0, 100], # Set y-axis range from 0 to 100%
                tickmode='array',
                tickvals=[0, 20, 40, 60, 80, 100],  # Define y-ticks
                ticktext=list(range(0, 101, 20)),   # Define y-tick labels
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
                font=dict(
                    size=self._font_size, 
                    family=self._font_family
                )
            ),
            width=1200,  # width of the plot in Pixel
            height=400,   # high of the plot in Pixel
            plot_bgcolor='white',  # background of the plot area
            paper_bgcolor='white',  # background of the entire figure
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
                font=dict(size=self._font_size,  family=self._font_family),
            )

        return fig

    def _plot_location_profile_week_ncluster(self) -> go.Figure:
        # Create subplots for each cluster
        n_timesteps_week = self._array_share_week_norm.shape[0]
        fig = subplots.make_subplots(
            rows=len(self._location_order), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[self._location_labels[i] for i in range(len(self._location_order))]
        )

        # Update the formatting of subplot titles
        for i, annotation in enumerate(fig['layout']['annotations']):
            annotation['font'] = dict(size=self._font_size+5, color='black', family=self._font_family)
            annotation['y'] += 0.012  # Adjust the y-position to move the title higher

        # Plot each location as a separate row
        for loc_idx, loc in enumerate(self._location_order):
            for cluster_idx, cluster in enumerate(self._clusters):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_timesteps_week)),
                        y=self._array_share_week_norm[:, loc_idx, cluster_idx],
                        mode='lines',
                        line=dict(
                            color=f"rgb({self._rgb_color[cluster_idx][0] * 255},{self._rgb_color[cluster_idx][1] * 255},{self._rgb_color[cluster_idx][2] * 255})",
                            width=2
                        ),
                        name=self._legend_clusters[cluster_idx] if self._legend_clusters else f"Cluster {cluster_idx + 1}",
                        legendgroup=f"Cluster {cluster_idx + 1}",
                        showlegend=(loc_idx == 0)  # Show legend only for the first row
                    ),
                    row=loc_idx + 1, col=1
                )

            # get the y-axis range from of share_loc
            y_min = 5 * np.floor(self._array_share_week_norm[:, loc_idx, :].min() / 5)
            y_max = 5 * np.ceil(self._array_share_week_norm[:, loc_idx, :].max() /5)
    
            fig.update_xaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                linewidth=1,
                tickvals=self._x_ticks,  # Ticks remain at 0, 24, 48, ...
                ticktext=[""] * len(self._x_ticks),  # Empty tick labels
                title=dict(
                    text = "Time of week",
                    standoff=35,  # Abstand zwischen Titel und Achse
                ),
                range=[0, n_timesteps_week],  # Set x-axis range from 0 to 168 (7 days * 24 hours)
                showticklabels=False,
                layer="above traces",
                title_font=dict(size=self._font_size), 
                row=loc_idx+1, col=1
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
                range=[y_min,y_max],
                row=loc_idx+1, col=1
            )

            # Add weekday labels at midpoints for each subplot
            for i, label_pos in enumerate(self._label_positions):
                    fig.add_annotation(
                        x=label_pos,
                        y=y_min, #y_min - (y_max-y_min)*0.01,  # Use the minimum from the y-axis range
                        yshift=-15,  # Adjust this value to position the labels below the x-axis
                        text=self._weekdays[i % 7],
                        showarrow=False,
                        yref=f"y{loc_idx + 1}",  # Reference the y-axis of the current subplot
                        xref="x",
                        font=dict(size=self._font_size,  family=self._font_family),
            )

        fig.update_layout(
            width=1200,  # width of the plot in Pixel
            height=250*len(self._location_order),   # high of the plot in Pixel
            plot_bgcolor='white',  # background of the plot area
            paper_bgcolor='white',  # background of the entire figure
            legend=dict(
                font=dict(
                size=self._font_size, 
                family=self._font_family
            ))
        )

        return fig
    
        # Plot the colors from _rgb_color
    def show_rgb_colors(self):
        df = pd.DataFrame({
            "Color": [f"Color {i+1}" for i in range(len(self._rgb_color))],
            "Value": [1] * len(self._rgb_color),
            "RGB": [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in self._rgb_color]
            })
        fig = px.bar(
            df,
            x="Color",
            y="Value",
            color="Color",
            color_discrete_sequence=df["RGB"],
            text="Color"
        )
        fig.update_traces(marker_line_color='black', marker_line_width=1, textposition='outside')
        fig.update_layout(
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            xaxis=dict(showgrid=False, zeroline=False),
            showlegend=False,
            title="RGB Colors",
            width=800,
            height=200,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        if self._show:
            fig.show()

    def _get_legend_clusters(self, mob_data_ext):
        """Get the legend labels for the clusters.
        """
        clusters = mob_data_ext.df['cluster'].unique()
        # Check legend_clusters
        if mob_data_ext.labels_clusters is None or len(mob_data_ext.labels_clusters) != len(clusters) or mob_data_ext.labels_clusters == []:
            legend_clusters = [f"Cluster {i}" for i in range(len(clusters))]
        else:
            legend_clusters = mob_data_ext.labels_clusters
        
        self._clusters = clusters
        self._legend_clusters = legend_clusters 




