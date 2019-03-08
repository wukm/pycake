# coding: utf-8
with open('OUTPUT_DIR/compare_parameters_quality_0.json'. 'w') as f:
    json.dump(integral_scores, f, indent=True)
with open('OUTPUT_DIR/compare_parameters_quality_0.json', 'w') as f:
    json.dump(integral_scores, f, indent=True)
    
with open(f'{OUTPUT_DIR}/compare_parameters_quality_0.json', 'w') as f:
    json.dump(integral_scores, f, indent=True)
    
import json
with open(f'{OUTPUT_DIR}/compare_parameters_quality_0.json', 'w') as f:
    json.dump(integral_scores, f, indent=True)
    
integral_scores
I = [i[1:] for i in integral_scores]
cvrs = np.array(I)
cvrs.shape
cvrs.squeeze()
cvrs = cvrs.squeeze()
get_ipython().run_line_magic('who', '')
PARAMS
labels = [p['label'] for p in PARAMS]
fig, ax = plt.subplots(figsize=(12,6))
ax.set_title(r'Incidence of Vesselness Score along Traced Vessels (25 samples)')
ax.set_xlabel('Frangi parametrization type')
ax.set_ylabel('Cumulative Vesselness Ratio (CVR)')
axl = plt.setp(ax, xticklabels=labels)
plt.setp(axl, rotation=45)
plt.show()
fig, ax = plt.subplots(figsize=(12,6))
ax.boxplot(cvrs)
plt.show()
I_medians = np.median(I, axis=0)
I
I_medians
fig, ax = plt.subplots(figsize=(6.5,7.5))
boxplot_dict = ax.boxplot(data, labels=labels)
boxplot_dict = ax.boxplot(I, labels=labels)
boxplot_dict = ax.boxplot(cvrs, labels=labels)
axl = plt.setp(ax, xticklabels=labels)
plt.setp(axl, rotation=90)
ax.set_xlabel('Frantgi parametrization type')
ax.set_ylabel('Cumulative Vesselness Ratio (CVR)')
ax.set_title(r'Incidence of Vesselness Score along Traced Vessels (25 samples)')
for line, med in zip(boxplot_dict['medians'], I_medians):
    x, y = line.get_xydata()[1]  # right of median line
    plt.text(x, y, '%.2f' % med, verticalalignment='center')
    
for line, med in zip(boxplot_dict['medians'], I_medians.squeeze()):
    x, y = line.get_xydata()[1]  # right of median line
    plt.text(x, y, '%.2f' % med, verticalalignment='center')
    
plt.subplots_adjust(bottom=0.30)
plt.savefig(f'{OUTPUT_DIR}/cvr_boxplot.png')
plt.show()
