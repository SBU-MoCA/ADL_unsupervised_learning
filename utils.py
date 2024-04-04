import os.path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(path, filename, training_loss, test_loss, test_accuracy, test_interval=10):
	plt.plot(training_loss, label='Training Loss')
	plt.plot(np.arange(1, len(test_loss)+1, 1) * test_interval, test_loss, label='Test Loss')
	plt.plot(np.arange(1, len(test_accuracy)+1, 1) * test_interval, test_accuracy, label='Test Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss & Accuracy')
	
	plt.legend()
	plt.grid()
	plt.title("loss & test accuracy")
	plt.savefig(os.path.join(path, filename))
	plt.cla()
	# plt.show()
