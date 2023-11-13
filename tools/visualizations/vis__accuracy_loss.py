import json
import matplotlib.pyplot as plt

# read json file
epoch_list = []
top_list = []
loss_list = []
for line in open('single.json', 'r', encoding='utf-8'):
    result = json.loads(line)
    top_1 = result.get('top-1')
    if top_1 is not None:
        epoch = result.get('epoch')
        loss = result.get('loss')

        epoch_list.append(epoch)
        top_list.append(top_1)
        loss_list.append(loss)


plt.plot(epoch_list, top_list, '-', label='Accuracy', color='blue')
plt.plot(epoch_list, loss_list, '-', label='Loss', color='red')

plt.xlabel("Number of epoch", fontsize=16)
plt.ylabel("Accuracy/Loss", fontsize=16)
plt.grid()
plt.legend()
# plt.savefig('loss.pdf')
plt.show()
