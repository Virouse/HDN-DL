import numpy as np
from load_y import load_y
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

y_test_DO5 = load_y('y_test_DO12.csv')
y_test_DO5_HDN = load_y('y_pred_DO12_HDN.csv')
y_test_DO5_LSSVR = load_y('y_pred_DO12_LSSVR.csv')

y_test_PH5 = load_y('y_test_PH5.csv')
y_test_PH5_HDN = load_y('y_pred_PH5_HDN.csv')
y_test_PH5_LSSVR = load_y('y_pred_PH5_LSSVR.csv')

# 绘制图形
fig, axs = plt.subplots(2, 1, figsize=(15, 8))  # 创建两个小图
plt.subplots_adjust(hspace=15)

# 绘制第一个小图
axs[0].plot(y_test_PH5_HDN, label='Predicated by HDN-DL', color='red', linewidth=1)
axs[0].plot(y_test_PH5_LSSVR, label='Predicated by LSSVR', color='green', linewidth=1)
axs[0].plot(y_test_PH5, label='Real Values', color='blue', linewidth=1)
axs[0].set_ylabel('PH Value', fontname='Times New Roman', fontsize=25)
axs[0].set_xlim(0, len(y_test_DO5))  # 手动设置X轴范围
axs[0].set_ylim(7, 9)  # 手动设置Y轴范围
axs[0].set_xticks(np.arange(0, 1001, 100))
axs[0].set_yticks(np.arange(7, 9.5, 0.5))
legend_font = FontProperties(family='Times New Roman', size=20)
axs[0].legend(loc='upper left', prop=legend_font)
axs[0].set_xticklabels(axs[0].get_xticks(), fontname='Times New Roman', fontsize=25)
axs[0].set_yticklabels(axs[0].get_yticks(), fontname='Times New Roman', fontsize=25)
axs[0].tick_params(axis='both', direction='in', pad=10)

# 绘制第二个小图
axs[1].plot(y_test_DO5_HDN, label='Predicated by HDN-DL', color='red', linewidth=1)
axs[1].plot(y_test_DO5_LSSVR, label='Predicated by LSSVR', color='green', linewidth=1)
axs[1].plot(y_test_DO5, label='Real Values', color='blue', linewidth=1)
axs[1].set_ylabel('DO Value', fontname='Times New Roman', fontsize=25)
axs[1].set_xlim(0, len(y_test_PH5))  # 手动设置X轴范围
axs[1].set_ylim(0, 20)  # 手动设置Y轴范围
axs[1].set_xticks(np.arange(0, 1001, 100))
axs[1].set_yticks(np.arange(0, 25, 5))
axs[1].legend(loc='upper left', prop=legend_font)
axs[1].set_xticklabels(axs[1].get_xticks(), fontname='Times New Roman', fontsize=25)
axs[1].set_yticklabels(axs[1].get_yticks(), fontname='Times New Roman', fontsize=25)
axs[1].tick_params(axis='both', direction='in')
plt.subplots_adjust(hspace=0.5, bottom=0.1)

plt.tight_layout()  # 自动调整布局
plt.savefig("regression.svg", dpi=1000, format="svg")
plt.show()  # 显示图形
