import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle



def Plot_ROC_Curve():
    lw = 2
    cls = ['CNN', 'LSTM', 'ResNet', 'ResNet+LSTM', 'OPFA-OWMT-DNet']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')

    colors = cycle(["blue", "darkorange", "y", "cyan", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i])

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def plot_results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC', 'FOR', 'PT',
            'BA', 'FM', 'BM', 'MK', 'PLHR', 'lrminus', 'DOR', 'Prevalence', 'Threat Score']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 20]
    Algorithm = ['TERMS', 'SGO', 'MOA', 'GOA', 'PFOA', 'OPFA-OWMT-DNet']
    Method = ['TERMS', 'CNN', 'LSTM', 'ResNet', 'ResNet+LSTM', 'OPFA-OWMT-DNet']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('--------------------------------------------------  K-fold - Dataset', i + 1, 'Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Method[0], Terms)
        for j in range(len(Method) - 1):
            Table.add_column(Method[j + 1], value1[j+5, :])
        print('--------------------------------------------------  K-fold - Dataset', i + 1, 'Method Comparison ',
              '--------------------------------------------------')
        print(Table)

    kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 0], color=[0.9, 0.6, 0.6], width=0.15, label="SGO-OWMT-DNet")
            ax.bar(X + 0.15, Graph[:, 1], color='g', width=0.15, label="MOA-OWMT-DNet")
            ax.bar(X + 0.30, Graph[:, 2], color=[0.5, 0.7, 0.5], width=0.15, label="GOA-OWMT-DNet")
            ax.bar(X + 0.45, Graph[:, 3], color='c', width=0.15, label="PFOA-OWMT-DNet")
            ax.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label="OPFA-OWMT-DNet")
            plt.xticks(X + 0.30, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFold', fontsize=16)
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            # plt.ylim([83, 97])
            path1 = "./Results/Dataset_%s_%s_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            length = np.arange(5)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 5], color='#010fcc', linewidth=6, marker='*', markerfacecolor='red',  # 98F5FF
                    markersize=20,
                    label='CNN')
            ax.plot(length, Graph[:, 6], color='#08ff08', linewidth=6, marker='*', markerfacecolor='green',  # 7FFF00
                    markersize=20,
                    label='ResNet')
            ax.plot(length, Graph[:, 7], color='#fe420f', linewidth=6, marker='*', markerfacecolor='cyan',  # C1FFC1
                    markersize=20,
                    label='LSTM')
            ax.plot(length, Graph[:, 8], color='#f504c9', linewidth=6, marker='*', markerfacecolor='#fdff38',
                    markersize=20,
                    label='ResNet+LSTM')
            ax.plot(length, Graph[:, 9], color='k', linewidth=6, marker='*', markerfacecolor='r', markersize=20,
                    label='OPFA-OWMT-DNet')

            ax.fill_between(length, Graph[:, 5], Graph[:, 6], color='#ff8400', alpha=.5)
            ax.fill_between(length, Graph[:, 6], Graph[:, 7], color='#19abff', alpha=.5)
            ax.fill_between(length, Graph[:, 7], Graph[:, 8], color='#00f7ff', alpha=.5)
            ax.fill_between(length, Graph[:, 8], Graph[:, 9], color='#ecfc5b', alpha=.5)
            plt.xticks(length, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFold', fontsize=16)
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/%s_Cls_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plot_results()
    # Plot_ROC_Curve()
