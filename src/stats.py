from read_files import read_all_data

import json
import numpy as np

from bokeh.plotting import figure, show, save
from bokeh.palettes import Set2_5

from statsmodels.stats.inter_rater import fleiss_kappa


def _compute_avg_time(num_el, num_ann, data):
    time = [0 for _ in range(num_el)]

    for name in data.keys():
        for i, t in enumerate(data[name]["timeDiffs"]):
            time[i] += t

    return list(map(lambda x: x / num_ann, time))


def plot_time(data, notSimple=False):
    anns = list(data.keys())
    num_anns = len(anns)
    num_elems = len(data[anns[0]]["timeDiffs"])

    x = list(range(num_elems))
    avg_time = _compute_avg_time(num_elems, num_anns, data)
    colors = ["red", "green", "blue", "violet"]

    p = figure(width=400, height=400)
    if notSimple:
        for i, name in enumerate(anns):
            p.scatter(x, data[name]["timeDiffs"], size=5, alpha=0.5, color=colors[i])

    p.line(x, avg_time, color="black", line_width=2)

    p.xaxis.axis_label = "Question number"
    p.yaxis.axis_label = "Time spent"

    show(p)


def read_model_data(model_file):
    """Read model data from the file and return a dictionary with counts of each annotation."""
    model_dict = dict()
    labels = ["correct", "incorrect", "wrong"]
    try:
        with open(model_file, "r", encoding="utf-8") as reader:
            for line in reader:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"Skipping line due to invalid JSON: {line}")
                    continue
                model_name = data["model"]
                model_dict.setdefault(model_name, {k: 0 for k in labels})
                model_dict[model_name][data["annotation"]] += 1
    except FileNotFoundError:
        print(f"Could not open file: {model_file}")
    return model_dict


def create_stacked_bar_chart(model_dict):
    """Create a stacked bar chart from the model data."""
    models = list(model_dict.keys())
    bokeh_source_data = {
        "models": models,
        "correct": [model_dict[name]["correct"] for name in models],
        "incorrect": [model_dict[name]["incorrect"] for name in models],
        "invalid": [model_dict[name]["wrong"] for name in models],
    }

    plot = figure(width=400, height=400, x_range=models)
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]
    labels = ["correct", "incorrect", "wrong"]
    plot.vbar_stack(
        labels,
        x="models",
        width=0.9,
        source=bokeh_source_data,
        legend_label=labels,
        color=colors,
    )

    plot.y_range.start = 0
    plot.x_range.range_padding = 0.1
    plot.xgrid.grid_line_color = None
    plot.axis.minor_tick_line_color = None
    plot.xaxis.major_label_orientation = "vertical"
    plot.outline_line_color = None
    plot.legend.location = "center"
    plot.legend.orientation = "horizontal"
    plot.add_layout(plot.legend[0], "above")

    show(plot)


def model_error(model_file):
    """Read model data and visualize it with a stacked bar chart."""
    model_dict = read_model_data(model_file)
    create_stacked_bar_chart(model_dict)


def calculate_fleiss_kappa(data):
    """Calculate Fleiss' Kappa for models."""

    names = list(data.keys())
    models = set(data[names[0]]["models"])

    model_annotations = [np.zeros((100, 3)) for _ in models]
    model_map = {model: i for i, model in enumerate(models)}

    for name in names:
        answer_map = [0 for _ in models]
        for i, val in enumerate(data[name]["answers"]):
            model = data[name]["models"][i]
            is_wrong = bool(data[name]["wrong"][i])
            answer_index = answer_map[model_map[model]]
            answer_map[model_map[model]] += 1

            if is_wrong:
                model_annotations[model_map[model]][answer_index, 2] += 1
            elif val == "yes":
                model_annotations[model_map[model]][answer_index, 0] += 1
            elif val == "no":
                model_annotations[model_map[model]][answer_index, 1] += 1
            else:
                print(f'Unexpected value: {val}. Expected "yes", "no" or a boolean.')
                continue

    return model_map, model_annotations


def create_plot(model_map, model_annotations):
    """Create a bar chart with Fleiss' Kappa for each model."""

    models = list(model_map.keys())
    data = {
        "models": models,
        "fleiss": [
            fleiss_kappa(model_annotations[model_map[model]]) for model in models
        ],
        "color": list(Set2_5),
    }

    plot_figure = figure(height=400, width=400, x_range=models)
    plot_figure.vbar(x="models", top="fleiss", source=data, width=0.7, color="color")

    plot_figure.y_range.start = 0
    plot_figure.x_range.range_padding = 0.1
    plot_figure.xgrid.grid_line_color = None
    plot_figure.axis.minor_tick_line_color = None
    plot_figure.xaxis.major_label_orientation = "vertical"
    plot_figure.outline_line_color = None

    show(plot_figure)


def compute_model_agreement(data):
    model_map, model_annotations = calculate_fleiss_kappa(data)
    create_plot(model_map, model_annotations)


if __name__ == "__main__":
    # index, d = read_all_data('../manual_evaluation', debug=False)
    # plot_time(d, notSimple=False)
    model_error("../manual_evaluation/manual_evaluation_merge.jsonl")
    # compute_model_agreement(d)
