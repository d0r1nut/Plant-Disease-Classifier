import os
import pandas as pd
import matplotlib.pyplot as plt

data = []
root_dir = 'dataset'

for label in os.listdir(root_dir):
    label_path = os.path.join(root_dir, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            data.append([os.path.join(label_path, file), label])

df = pd.DataFrame(data, columns=['filepath', 'label'])

print("Total Images:", len(df))
class_counts = df['label'].value_counts()
print(class_counts)

plt.figure(figsize=(10,6))
class_counts.plot(kind='bar', color='green')
plt.title('Disease Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('distribution.png')
print("Distribution chart saved as distribution.png")