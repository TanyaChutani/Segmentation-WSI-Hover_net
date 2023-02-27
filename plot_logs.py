from matplotlib.pylab import plt
from numpy import arange
import json
  
# Opening JSON file
f = open('/home/ubuntu/aira/hover_net/logs/00/stats.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

# Load the training and validation loss dictionaries
 
epochs = range(1, len(data))
train_loss = []
# Generate a sequence of integers to represent the epoch numbers
for epoch in range(1, len(data)):
    train_loss.append(data[str(epoch)]["train-overall_loss"])

    
# Plot and label the training and validation loss values
plt.plot(epochs, train_loss, label='Training Loss')
 
# Add in a title and axes labels
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
# Set the tick locations
plt.xticks(arange(1, len(data), 2))
 
# Display the plot
plt.legend(loc='best')
plt.savefig('/home/ubuntu/aira/hover_net/train_logs.png')
plt.show()