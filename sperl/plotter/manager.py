import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sperl.generatedata.measures.all_measures import AllMeasures
from sperl.plotter.data_loader import DataLoader


class PlotterManager:
    LARGE_INT = 2**63 - 1

    def __init__(self, **kwargs):
        self.x_axis: str
        self.compare_variable: str
        self.folder: str
        self.input_file: List[str]
        self.output_file: str
        self.logscale: bool
        self.distance_range: Optional[Tuple[float, float]]
        self.distance: str
        self.resource_min: Optional[int]
        self.resource_max: Optional[int]
        self.canvas: Tuple[float, float]
        self.fontsize_axis: int
        self.fontsize_numbers: int
        self.fontsize_legend: int
        self.thicker_axis: bool
        self.print_resource_value: Optional[float]
        self.overlay_speedup: Optional[str]
        self.colors_alt: int

        self.min_resource_values = {}

        for attr, val in kwargs.items():
            setattr(self, attr, val)

        self.set_output_name()

    def set_output_name(self):
        self.output_file = (
            f"out_{self.output_file}_{self.distance}_{self.x_axis}"
        )
        if self.resource_min != 0:
            self.output_file += f"_min{self.resource_min}"
        if self.resource_max != self.LARGE_INT:
            self.output_file += f"_max{self.resource_max}"
        if self.logscale:
            self.output_file += "_log"
        self.output_file += ".pdf"

    def _print_compare_value(self, compare_value):
        if self.print_resource_value != -100.0:
            print("compare_value", compare_value)

    def _get_colors(self):
        if self.colors_alt == 1:
            return ['#da20e9', '#8681a4', '#47d0fc']
        elif self.colors_alt == 2:
            return  ['#d15df9', '#eea44f', '#73df5c']
        else:
            return ['#1f77b4', '#ff7f0e', '#2ca02c']

    def run(self):
        plt.rcParams.update({"font.size": self.fontsize_numbers})
        fig, ax = plt.subplots(figsize=self.canvas)
        if self.thicker_axis:
            plt.setp(ax.spines.values(), linewidth=1.4)  # type: ignore

        dl = DataLoader(
            self.folder,
            self.input_file,
            self.compare_variable,
            self.distance,
            self.x_axis,
        )
        cols = self._get_colors()
        for color, (compare_value, mean, std_err, x_axis_value) in zip(cols, dl.load_values()):
            self.min_resource_values[compare_value] = self._get_min_resource_values(mean, x_axis_value)
            self._print_compare_value(compare_value)
            self._print_resource_values(mean, x_axis_value)
            mean, std_err, x_axis_value = self._clip_to_range(
                mean, std_err, x_axis_value
            )
            ax.plot(
                x_axis_value,
                mean,
                label="{}{}".format(self.legend_text(), compare_value),
                linewidth=2,
                color=color
            )
            ax.fill_between(
                x_axis_value, mean - std_err, mean + std_err, alpha=0.5, facecolor = color, linewidth=0, edgecolor = (0,0,0,0.0)
            )
        
        if self.logscale:
            ax.set_yscale("log")

        ax.set_xlabel(
            self.x_axis_text(), fontsize=self.fontsize_axis
        )  # , fontname="DejaVu Serif")
        ax.set_ylabel(
            self.y_axis_text(), fontsize=self.fontsize_axis
        )  # , fontname="DejaVu Serif")
        self._overlay_speedup(ax)

        if self.distance_range:
            ax.set_ylim(
                bottom=float(self.distance_range[0]),
                top=float(self.distance_range[1]),
            )

        ax.legend(loc="best", prop={"size": self.fontsize_legend})

        fig.savefig(self.filepath, bbox_inches="tight")

    def _overlay_speedup(self, ax):
        col = "#262626"

        if self.overlay_speedup is None:
            return
        y = self.print_resource_value
        x1 = self.min_resource_values['MDRR']
        x2 = min(self.min_resource_values['RR'], self.min_resource_values['DRR'])
        # Drawing a two-sided arrow
        ax.annotate(
            '',                         # No text, just the arrow
            xy=(x2, y),                # Head of the arrow (end point)
            xytext=(x1, y),              # Tail of the arrow (start point)
            arrowprops=dict(arrowstyle="<->", lw=2, color=col),
            color=col
        )

        # Adding text under the arrow
        ax.text((x1 + x2) / 2, y-0.05 , self.overlay_speedup, ha='center', va='top', fontsize=self.fontsize_numbers-2, color=col)


    def _is_resource_value(self, prev_mean, mean):
        return prev_mean >= self.print_resource_value and self.print_resource_value >= mean

    def _print_resource_value(self, prev_mean, mean, v):
        if prev_mean is None:
            return
        if self._is_resource_value(prev_mean, mean):
            print("   resource value = ", self.print_resource_value)
            print("   hit at: ", v)

    def _print_resource_values(self, mean, x_axis_value):
        prev_mean = None
        for m, v in zip(mean, x_axis_value):
            self._print_resource_value(prev_mean, m, v)
            prev_mean = m
            
    def _get_min_resource_values(self, mean, x_axis_value):
        if self.print_resource_value == -100.0:
            return None
        prev_mean = None
        for m, v in zip(mean, x_axis_value):
            if prev_mean is not None:
                if self._is_resource_value(prev_mean, m):
                    return v
            prev_mean = m
        raise ValueError("No resource value found.")

    def _clip_to_range(
        self, mean, std_err, x_axis_value
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        if self.resource_min == 0 and self.resource_max == self.LARGE_INT:
            return mean, std_err, x_axis_value

        mean_clipped = []
        std_err_clipped = []
        x_axis_value_clipped = []
        for m, s, v in zip(mean, std_err, x_axis_value):  # type: ignore
            if v >= self.resource_min and v < self.resource_max:
                mean_clipped.append(m)
                std_err_clipped.append(s)
                x_axis_value_clipped.append(v)
        return (
            np.array(mean_clipped),
            np.array(std_err_clipped),
            x_axis_value_clipped,
        )

    def _print_diagnostic_info(self, compare_value, mean, x_axis_value):
        print("{} = {}".format(self.compare_variable, compare_value))
        print("{} = {}".format("  mean", mean))
        print("{} = {}".format("  x_axis_value", x_axis_value))

    def x_axis_text(self) -> str:
        if self.x_axis == "step":
            return "Round $t$"
        elif self.x_axis == "samples":
            return "#samples"
        elif self.x_axis == "retrainings":
            return "#retrainings"
        raise NotImplementedError()

    def y_axis_text(self) -> str:
        return AllMeasures.get_distance(self.distance).axis_text()

    @property
    def filepath(self) -> str:
        return os.path.join(self.folder, self.output_file)

    def legend_text(self) -> str:
        match self.compare_variable:
            case "beta":
                return "$\\beta=$"
            case "regularizer":
                return "$\\lambda=$"
            case "gamma":
                return "$\\gamma=$"
            case "num_ftrl_steps":
                return "$FTRL-steps=$"
            case "b":
                return "$b=$"
            case "k":
                return "$k=$"
            case "agent2_algorithm":
                return ""
            case "meta_policy":
                return ""
            case "v":
                return "$v=$"
        raise NotImplementedError(
            '"{}" is not supported.'.format(self.compare_variable)
        )
