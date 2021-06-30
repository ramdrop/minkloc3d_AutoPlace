# %%
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

font_path = '/home/kaiwen/Downloads/times.ttf'  # the location of the font file
my_font = fm.FontProperties(
    fname=font_path, size=20)  # get the font based on the font_path

fig, ax = plt.subplots()

ax.plot([0, 1, 2, 3], [2, 3, 4, 5], color='green')
ax.set_xlabel('Some text', fontproperties=my_font)
ax.set_ylabel('Some text', fontproperties=my_font)
ax.set_title('title', fontproperties=my_font)
for label in ax.get_xticklabels():
    label.set_fontproperties(my_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(my_font)