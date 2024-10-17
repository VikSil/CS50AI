# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%

# Read in the data
data = pd.read_csv('./sample.csv')

# %%

# ACCURACY analysis
accuracy = data['accuracy'].sort_values()
print(accuracy)

accuracy_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0] 
accuracy_freq_dist = (
    pd.cut(accuracy, bins=accuracy_bins, include_lowest=True, right=False).value_counts().sort_index()
)

plt.bar(accuracy_freq_dist.index.astype(str), accuracy_freq_dist, width=0.5, color='blue')
plt.xlabel('Validation Accuracy')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.show()

perc_90_cutoff = accuracy.quantile(0.90)
perc_90_count = len(accuracy[accuracy >= perc_90_cutoff])
print(f'90th percentile starts at: {perc_90_cutoff}')
print(f'90th percentile contains {perc_90_count} elements')

# Remove runs with accuracy less than 0.9761 form the sample
data.drop(data[data['accuracy'] < 0.9761].index, inplace=True)
data.sort_values(['accuracy'], ascending= False, inplace=True)
print(data)

# %%

# LOSS analysis

loss = data['loss'].sort_values()
print(loss)

loss_bins = [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]
loss_freq_dist = pd.cut(loss, bins=loss_bins, include_lowest=True, right=False).value_counts().sort_index()

plt.bar(loss_freq_dist.index.astype(str), loss_freq_dist, width=0.5, color='blue')
plt.xlabel('Validation Loss')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.show()

data.plot(x='accuracy', y='loss')

high_loss_count = len(loss[loss>=0.1])
print(high_loss_count)

# Remove runs with loss at least 1.0
data.drop(data[data['loss'] >= 0.1].index, inplace=True)
data.sort_values(['loss'], ascending=False, inplace=True)
print(data)


# %%

# CONVERGENCE analysis

convergence = data['acc_convergence'].sort_values()

convergence_epochs = convergence.value_counts()
plt.bar(convergence_epochs.index.astype(str), convergence_epochs, color='blue')
plt.xlabel('Epochs till Convergence')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.show()

data.sort_values(['accuracy'], ascending=False, inplace=True)
data.plot(x='accuracy', y='acc_convergence')

data.sort_values(['loss'], ascending=False, inplace=True)
data.plot(x='loss', y='acc_convergence')

data['accuracy_rank'] = data['accuracy'].rank(ascending = False)
data['loss_rank'] = data['loss'].rank()
data['rank'] = data['accuracy_rank'] * data['loss_rank']
print(data)

print(f"Lowest accuracy before pruning is: {data['accuracy'].min()}")
print(f"Highest accuracy before pruning is: {data['accuracy'].max()}")

print(f"Lowest loss before pruning is: {data['loss'].min()}")
print(f"Highest loss before pruning is: {data['loss'].max()}")

# Remove 100 lowest ranked runs with zero convergence
data.sort_values(['rank'], ascending = False, inplace = True)
data = data.iloc[100:]

print(f"Lowest accuracy after pruning is: {data['accuracy'].min()}")
print(f"Highest accuracy after pruning is: {data['accuracy'].max()}")

print(f"Lowest loss after pruning is: {data['loss'].min()}")
print(f"Highest loss after pruning is: {data['loss'].max()}")

# %%

# ACCURACY INCREASE analysis

accuracy_increase_sectors = data['acc_valid'].value_counts()
plt.pie(
    accuracy_increase_sectors,
    labels=accuracy_increase_sectors.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#ff9999', '#66b3ff'],
)
plt.title('Validation Accuracy Higher than 10th Epoch Accuracy')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle
plt.show()

# Remove runs that did not show accuracy increase at validation
data.drop(data[data['acc_valid'] == False].index, inplace=True)

print(data)

# %%

# LOSS DECREASE analysis

loss_decrease_sectors = data['loss_valid'].value_counts()
plt.pie(
    loss_decrease_sectors,
    labels=loss_decrease_sectors.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#ff9999', '#66b3ff'],
)
plt.title('Validation Loss Lower than 10th Epoch Loss')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle
plt.show()

# Remove runs that did not show loss decrease at validation
data.drop(data[data['loss_valid'] == False].index, inplace=True)
print(data)

# %%

# TRAINABLE PARAMETERS analysis

parameters = data['parameters'].sort_values()
print(parameters)

parameters_bins = np.linspace(parameters.min(), parameters.max(), 10)
parameters_freq_dist = (
    pd.cut(parameters, bins=parameters_bins, include_lowest=True, right=False).value_counts().sort_index()
)

plt.bar(parameters_freq_dist.index.astype(str), parameters_freq_dist, width=0.5, color='blue')
plt.xlabel('Trainable parameters')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.show()

data.sort_values(['accuracy'], ascending=False, inplace=True)
data.plot(x='accuracy', y='parameters')

data.sort_values(['loss'], ascending=False, inplace=True)
data.plot(x='loss', y='parameters')


# Remove runs with more than 1.2M trainable parameters
data.drop(data[data['parameters'] > 1_200_000].index, inplace=True)
print(data)


# %%

# SETUP analysis

filters = data['filters'].sort_values()

print(f"Lowest numer of filters is: {filters.min()}")
print(f"Highest number of filters is: {filters.max()}")
print()

filters_bins = np.linspace(filters.min(), filters.max(), 5)
filters_freq_dist = pd.cut(filters, bins=filters_bins, include_lowest=True, right=False).value_counts().sort_index()

plt.bar(filters_freq_dist.index.astype(str), filters_freq_dist, width=0.5, color='blue')
plt.xlabel('Number of Filters')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.show()


filter_size_sectors = data['filter_size'].value_counts()
plt.pie(
    filter_size_sectors,
    labels=filter_size_sectors.index,
    autopct='%1.1f%%',
    startangle=90,
)
plt.title('Filter Size')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle
plt.show()


pooling_size_sectors = data['pooling_size'].value_counts()
plt.pie(
    pooling_size_sectors,
    labels=pooling_size_sectors.index,
    autopct='%1.1f%%',
    startangle=90,
)
plt.title('Pooling Kernel Size')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle
plt.show()


hidden_nodes = data['hidden_nodes'].sort_values()

print(f"Lowest numer of hidden nodes is: {hidden_nodes.min()}")
print(f"Highest number of hidden nodes is: {hidden_nodes.max()}")
print()

hidden_nodes_bins = np.linspace(hidden_nodes.min(), hidden_nodes.max(), 5)
hidden_nodes_freq_dist = (
    pd.cut(hidden_nodes, bins=hidden_nodes_bins, include_lowest=True, right=False).value_counts().sort_index()
)

plt.bar(hidden_nodes_freq_dist.index.astype(str), hidden_nodes_freq_dist, width=0.5, color='blue')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.show()


dropout = data['dropout'].sort_values()

print(f"Lowest dropout is: {dropout.min()}")
print(f"Highest dropout is: {dropout.max()}")
print()

dropout_bins = np.linspace(dropout.min(), dropout.max(), 5)
dropout_freq_dist = pd.cut(dropout, bins=dropout_bins, include_lowest=True, right=False).value_counts().sort_index()

plt.bar(dropout_freq_dist.index.astype(str), dropout_freq_dist, width=0.5, color='blue')
plt.xlabel('Dropout')
plt.ylabel('Frequency')
plt.xticks(rotation=30, ha='right')
plt.show()

# %%

# FINAL PRUNING

criteria_cols = ['accuracy', 'loss', 'parameters']
small_data = pd.DataFrame(index = data.index)
small_data[criteria_cols] = data[criteria_cols]

print(small_data)

indices_to_prune = []
for index1, row1 in small_data.iterrows():
    for index2, row2 in small_data.iterrows():
        if row1['accuracy'] < row2['accuracy'] and row1['loss'] > row2['loss'] and row1['parameters'] > row2['parameters']:
            indices_to_prune.append(index1)
            break

print(sorted(indices_to_prune))

data.drop(index = indices_to_prune, inplace=True)
print(data)

