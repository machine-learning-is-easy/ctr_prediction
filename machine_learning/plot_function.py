# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def evaluation_plot(evaluation):

    labels = evaluation["content_spacy_id"]
    auc = evaluation["AUC"]
    top5recall = evaluation["Top 5 recall"]
    precision = evaluation["Top 5 precision"]
    f1 = evaluation["Top 5 f1"]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2*2, auc, width, label='AUC')
    rects2 = ax.bar(x + width/2, top5recall, width, label='Top 5 recall')
    rects3 = ax.bar(x - width/2, precision, width, label='Top 5 precision')
    rects4 = ax.bar(x + width/2*2, f1, width, label='Top 5 f1')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    fig.tight_layout()
    plt.show()
