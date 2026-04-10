#!/usr/bin/env python3

import json
import pandas as pd
import matplotlib.pyplot as plt

XLABEL = "Quantization Type and Precision"

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

def create_figures(df):
    accuracy = df.plot.bar(x="title", y="accuracy", rot=0, xlabel=XLABEL, ylabel="Accuracy")
    latency = df.plot.bar(x="title", y="avg_latency", yerr="std_latency", rot=0, capsize=4, xlabel=XLABEL, ylabel="Mean latency")
    throughput = df.plot.bar(x="title", y="avg_throughput", yerr="std_throughput", rot=0, capsize=4, xlabel=XLABEL, ylabel="Mean throughput (Img/s)", legend=False, title="Pipeline Throughput")

    scatter_throughput = df.plot.scatter(x="avg_throughput", y="accuracy", xerr="std_throughput", xlabel="Pipeline Throughput (Img/s)", ylabel="Accuracy", title="Accuracy vs Pipeline Throughput")
    label(df, scatter_throughput, "avg_throughput", "accuracy")
    scatter_latency = df.plot.scatter(x="avg_latency", y="accuracy", xerr="std_latency", s=df["memory_used"] * 5, rot=0, xlabel="Latency (ms)", ylabel="Accuracy", title="Accuracy vs Latency")
    label(df, scatter_latency, "avg_latency", "accuracy")

def label(df, plot, x, y):
    for _, row in df.iterrows():
        plot.annotate(
            row["title"],
            (row[x], row[y]),
            xytext=(5, 5),
            textcoords="offset points"
        )

def main():
    name = input("Results file path: ")
    with open(name, "r") as f:
        data = json.load(f)
    
    rows = []
    
    # Base model
    base = data["base"]
    rows.append({
        "method": "base",
        "precision": "float32",
        "title": "float32",
        "accuracy": base["accuracy"],
        **base["bench"]
    })
    
    # Quantized models
    for method in ["ptq", "qat"]:
        for precision in ["float16", "int8"]:
            entry = data[method][precision]
            rows.append({
                "method": method,
                "precision": precision,
                "title": f"{method.upper()} {precision}",
                "accuracy": entry["accuracy"],
                **entry["bench"]
            })
    
    # method precision title accuracy avg_latency std_latency memory_used avg_throughput std_throughput
    df = pd.DataFrame(rows)
    df["title"] = pd.Categorical(df["title"], ["float32", "QAT float16", "PTQ float16", "QAT int8", "PTQ int8"], ordered=True)
    df = df.sort_values("title")
    
    print(df.to_csv(sep='\t', index=False))

    create_figures(df)

if __name__ == "__main__":
    main()
    plt.show()
